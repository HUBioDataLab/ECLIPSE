import argparse
import json
import torch
import torch.nn.functional as F 
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime


from torch_geometric.seed import seed_everything
from model import ECLIPSE
from data_loader import get_data_loaders
from utils import performance_scores, load_config

from pathlib import Path
# Resolve paths relative to repo root
ROOT_DIR = Path(__file__).resolve().parents[1]   # src → repo root

def parse_args():
    CONFIG_DIR = ROOT_DIR / "configs"

    parser = argparse.ArgumentParser(description="Train ECLIPSE model for binding affinity prediction")
    parser.add_argument("-s", "--split", type=str, required=True, choices=["fds", "dcs", "rs"],
                        help="Data split: fds (fully_dissimilar_split), dcs (dissimilar_compound_split), or rs (random_split)")
    parser.add_argument("-pl", "--prediction-layer", type=str, required=True, choices=["dp", "fc"],
                        help="Prediction layer: dp (dot_product) or fc (fully_connected)")
    parser.add_argument("-cr", "--compound-representation", type=str, required=True, choices=["ecfp4", "selformer"],
                        help="Compound representation: ecfp4 or selformer")
    parser.add_argument("-nw", '--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument("-nt", '--num-threads', type=int, default=2, help='Number of CPU threads')
    parser.add_argument("-o", '--output-dir', type=str, default= ROOT_DIR / 'outputs', help='Output directory')
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to config file (default is generated from other args)")
    parser.add_argument("-sm",'--save-model', action='store_true', help='Save trained model')
    parser.add_argument("-sp",'--save-predictions', action='store_true', help='Save predictions')
    parser.add_argument("-b", "--baseline", action='store_true',
                        help="Use baseline model (no HGT layers, only linear layers)")

    # First parse known args to get split/model/compound_rep
    args, _ = parser.parse_args()

    # If config_path is not given, set a default based on other args
    if args.config_path is None:
        config_path = CONFIG_DIR / f"{args.split}_config.yaml"
        args.config_path = config_path


    return args    

    
#-----Initialize lazy parameters via forwarding a single batch to the model----------
@torch.no_grad()
def init_params(model, loader, device, edge_name):
    batch = next(iter(loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict, batch[edge_name].edge_label_index)
    

#-----Define train and test functions----------
def train(model, optim, loader, device, epoch, edge_name):
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(loader,desc=f"Epoch{epoch}: "):
        batch = batch.to(device) 
        optim.zero_grad()

        out = model(batch.x_dict, batch.edge_index_dict, batch[edge_name].edge_label_index)
        
        true_label = batch[edge_name].edge_label  
        true_label_size = len(true_label)
        
        loss = F.mse_loss(out, true_label)
        
        loss.backward()
        optim.step()
        
        total_examples += true_label_size
        total_loss += float(loss) * true_label_size
        
    return total_loss / total_examples


@torch.no_grad()
def test(model,test_loader, device, edge_name):
    model.eval()
    
    preds = []
    true_labels = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        pred = model(batch.x_dict, batch.edge_index_dict, batch[edge_name].edge_label_index)
        true_label = batch[edge_name].edge_label

        preds.append(pred)
        true_labels.append(true_label)

    preds = torch.cat(preds, dim=0).cpu()   
    true_labels = torch.cat(true_labels, dim=0).cpu()

    return [true_labels, preds]



#-----Train and evaluate the model---------

def run_model(model,
              train_loader,
              test_loader,
              config, 
              optimizer,
              device,
              model_name, 
              threshold,  
              edge_name
            ):
    
    start = datetime.now()
    print(f"Starting training run: {model_name}")

    for epoch in range(1, config.epochs+1):
        # Training phase
        ep_tr_loss = train(model,optimizer,train_loader,device,epoch, edge_name)

    # Test phase
    ts_out = test(model,test_loader, device, edge_name)

    ts_true, ts_pred = ts_out
    ts_score = performance_scores(ts_true, ts_pred, threshold)

    loss, rmse, pear, spear, mcc, acc, prec, rec, f1 = ts_score

    ts_score_dict = {
        "loss": loss,
        "rmse": rmse,
        "pearson": pear,
        "spearman": spear,
        "mcc": mcc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    # Log metrics
    total_time = datetime.now() - start
    print(f'Training completed for {model_name} in {total_time}')
    print(f'\nTrain loss: {ep_tr_loss:.4f}  -  Test loss: {loss:.4f}')

    # Save test results
    with open(fr'{config.output_dir}/{model_name}_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(ts_score_dict, f, indent=4)


    if config.save_model:
        torch.save(model.state_dict(), ROOT_DIR / "saved_models" / fr"{model_name}.pt")

    if config.save_predictions:       
        df_preds = pd.DataFrame({"true_values":ts_true.tolist(), "predictions":ts_pred.tolist()})
        df_preds.to_csv(fr"{config.output_dir}/{model_name}_test_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tsv", sep="\t", index=None)
           



def main():
    # Parse command line arguments
    args = parse_args()

    # Limits CPU threads
    torch.set_num_threads(args.num_threads) 

    # For reproducibility
    seed_everything(42)
    torch.backends.cudnn.deterministic = True

    # Device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Run name
    if args.baseline:
        model_name = f'{args.split}_{args.prediction_layer}_{args.compound_representation}'
    else:
        model_name = f'{args.split}_eclipse_{args.prediction_layer}_{args.compound_representation}'
 
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    print(f"Configuration:\n{model_name}\n")
    model_cfg = config["model"][model_name]

    # Load and prepare data
    print("Loading CROssBAR KG...")
    crossbar_kg = torch.load(ROOT_DIR / fr"data/integrated_crossbar_kg_{args.compound_representation}.pt")   

    print("Removing edges between protein and compound nodes to prevent data leakage...")
    del crossbar_kg["protein","rev_Chembl","compound"]
    crossbar_kg["compound","Chembl","protein"].edge_index = crossbar_kg["compound","Chembl","protein"].edge_index[:,:0]

    print(fr"Integrating {args.split} train/test edge indices and labels...") 
    
    # Load train/test edges
    tr_ts_data = torch.load(ROOT_DIR / fr"data/{args.split}_train_test_edges.pt")
    
    tr_edge_index, tr_edge_label = tr_ts_data["tr_edge_index"], tr_ts_data["tr_edge_label"]
    ts_edge_index, ts_edge_label = tr_ts_data["ts_edge_index"], tr_ts_data["ts_edge_label"]
    print(f"Train edges: {tr_edge_index.shape[1]}, Test edges: {ts_edge_index.shape[1]}")
    
    # Assign train/test edges to the CROssBAR KG
    crossbar_kg[edge_name].train_data, crossbar_kg[edge_name].train_label = tr_edge_index, tr_edge_label
    crossbar_kg[edge_name].test_data, crossbar_kg[edge_name].test_label = ts_edge_index, ts_edge_label

    print(f"CROssBAR KG loaded with {crossbar_kg.num_nodes} nodes and {crossbar_kg.num_edges} edges.\n")
    print(f"CROssBAR KG:\n {crossbar_kg}\n") 

    # Preparing data loaders
    print("Preparing data loaders...")
    edge_name = ("compound", "Chembl", "protein") 
    train_loader, test_loader = get_data_loaders(crossbar_kg, model_cfg["tr_batch_size"], args.num_workers, edge_name)    
        
   # Start training
    print('----------------------Starting training----------------------\n')
    
    model = ECLIPSE(crossbar_kg, model_cfg, args.prediction_layer, args.baseline)  # Initialize the model
    print(model)
    model.to(device)
    
    init_params(model,train_loader, device, edge_name)  # Initialize model parameters
    print("Model parameters initialized.\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.wd)

    tr_pchembl_median = np.median(crossbar_kg[edge_name].cpu().train_label)


    run_model(model, train_loader, test_loader, model_cfg, optimizer, device, model_name, tr_pchembl_median)

    print(datetime.now())



if __name__ == "__main__":
    main()