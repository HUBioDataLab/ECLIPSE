import torch
from torch_geometric.nn import Linear, HGTConv


#-----Define ECLIPSE model architecture----------

class ECLIPSE(torch.nn.Module):
    def __init__(self, data, config, prediction_layer, baseline=False):  
        """
        Parameters:
            data: PyG HeteroData object containing node features and edge indices for each node/edge type.
            config: dict
                Configuration dictionary with model hyperparameters, e.g.:
                    - hidden_channels: int, hidden dimension size for all layers
                    - num_heads: int, number of attention heads for HGTConv
                    - num_layers: int, number of HGTConv layers
                    - fc_layers: int, number of fully connected layers (for 'fc' prediction layer)
                    - coeff: int, scaling coefficient for hidden channels in fc layers
                    - dropout: float, dropout rate for fc layers
            prediction_layer: str, prediction head type ('dp' for dot product, 'fc' for fully connected)
            baseline: bool, if True disables HGTConv and uses only linear layers for compound/protein nodes
        """
        super().__init__()
        self.config = config
        self.prediction_layer = prediction_layer
        self.baseline = baseline

        # Linear transformation for each node type
        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(-1, self.config["hidden_channels"])
            for node_type in data.node_types
        })

        # Stack of HGTConv layers (if baseline is False)
        if not self.baseline:
            self.convs = torch.nn.ModuleList([
                HGTConv(self.config["hidden_channels"], self.config["hidden_channels"], data.metadata(), self.config["num_heads"], group='sum')
                for _ in range(self.config["num_layers"])
            ])
        else:
            self.convs = None

        self.left_linear = Linear(self.config["hidden_channels"], self.config["hidden_channels"])
        self.right_linear = Linear(self.config["hidden_channels"], self.config["hidden_channels"])

        #Method1: computing dot product of learned protein and compound embeddings
        if self.prediction_layer == "dp":           
            self.sqrt_hd  = self.config["hidden_channels"]**0.5  # For normalization of dot product output

           
        #Method2: applying fully connected layers upon concatenation of learned protein and compound embeddings
        elif self.prediction_layer == "fc":
            input_dim = self.config["hidden_channels"] * 2  # Input dimension for the first FC layer
            coeff = self.config["coeff"]  # Coefficient for hidden channels scaling
            dropout = self.config["dropout"]  # Dropout rate for FC layers

            # Build FC layers dynamically
            layers = []
            for i in range(1, self.config["fc_layers"] + 1):
                # determine dims: first from input_dim, last to 1
                in_dim  = input_dim if i == 1 else self.config["hidden_channels"] * coeff
                out_dim = 1 if i == self.config["fc_layers"] else self.config["hidden_channels"] * coeff
                layers.append(Linear(in_dim, out_dim))

                # add BN, activation, dropout after non-final layers
                if i < self.config["fc_layers"]:
                    layers.append(torch.nn.BatchNorm1d(self.config["hidden_channels"] * coeff))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout(dropout))

            self.fc = torch.nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown prediction layer: {self.prediction_layer}")



    def forward(self, x_dict, edge_index_dict, edge_label_index):
        row, col = edge_label_index

        # Project initial node features to hidden dimension
        if self.baseline:   # Linearize only compound and protein nodes for baseline
            x_dict = {
                "compound": self.lin_dict["compound"](x_dict["compound"]).relu_(),
                "protein": self.lin_dict["protein"](x_dict["protein"]).relu_()
            }
        else:
            x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
            }

        # Pass through HGTConv layers (if baseline is False)
        if not self.baseline and self.convs is not None:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)

        # Learned compound and protein embeddings 
        cmp = x_dict["compound"]  
        prt = x_dict["protein"]

        cmp_lin = self.left_linear(cmp)
        prt_lin = self.right_linear(prt)
        
        # Compute predicted binding affinities depending on the method selected
        if self.prediction_layer == "dp":
            out = (cmp_lin[row] * prt_lin[col]).sum(dim=-1)
            return out / self.sqrt_hd
        
        elif self.prediction_layer == "fc":
            cmp_prt = torch.cat([cmp_lin[row], prt_lin[col]], dim=-1)
    
            # MLP forward through FC layers with BatchNorm
            out = self.fc(cmp_prt)
            return out.view(-1)
    

  
        

