# 🕸️ ECLIPSE 

**ECLIPSE (Exploration of Complex Ligand-Protein Interactions through Learning from Systems-level Heterogeneous Biomedical Knowledge Graphs)** is an AI-powered framework for predicting compound–protein interactions. By combining advanced graph modeling, comprehensive biomedical knowledge, and pre-trained embeddings, it uncovers hidden relationships within complex biological networks, offering a practical tool for researchers in drug discovery and computational biology.

**ECLIPSE combines:**
- **Large-scale heterogeneous biomedical knowledge graphs (KGs):** We built this integrated KG using our in-house [CROssBAR platform](https://crossbar.kansil.org/about.php), capturing entities, including genes, proteins, drugs, compounds, pathways, diseases, and phenotypes, and their multi-layered interactions.  
- **Feature embeddings from language and graph models:** Each biological entity is represented using learned embeddings, enabling richer context and better predictions.  
- **Heterogeneous Graph Transformer (HGT):** Unlike standard GNNs, HGT leverages node and edge types with type-specific attention, effectively modeling complex and diverse relationships.

<p align="center">
  <img src="workflow.png" alt="The schematic representation of the ECLIPSE framework" width="800"/>
</p>

**The schematic representation of the ECLIPSE framework.** ECLIPSE is a systems-level framework for predicting compound–protein bioactivity. **The Integrated CROssBAR KG module** provides a multi-relational biomedical graph of proteins, compounds, drugs, pathways, phenotypes, and diseases, serving as the structural foundation for representation learning. From this graph, sampled subgraphs are processed in the **Node Projection on Sampled Subgraphs module**, where type-specific MLP layers project heterogeneous input node features into fixed-size representations. These embeddings are then passed into stacked **HGT Layers**, which apply heterogeneous mutual attention, message passing, and target-specific aggregation with residual connections to generate contextualized node embeddings. Finally, the **Prediction Layer** combines updated compound and protein embeddings, which are first refined through separate MLPs, either through vector concatenation with a fully connected network or via dot product, to predict bioactivity values.

---

## 📚 Contents

- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Training the ECLIPSE Model](#️-training-the-eclipse-model)
- [Making Predictions](#-making-predictions)
- [License](#-license)

## 📁 Repository Structure

The ECLIPSE repository is organized as follows:

```text
ECLIPSE/
│
├── data/                     # Input datasets and knowledge graph resources
│   ├── node_index/             # Node indexing files
│   ├── train_test_samples/     # Train/test splits for CPI benchmark datasets
│   └── crossbar_kg/            # Preprocessed knowledge graph and feature tensors
│
├── saved_models/                    # Trained ECLIPSE models
│   └── dcs_eclipse_dp_selformer.pt    # Dot-product ECLIPSE model with SELFormer embeddings, trained on dissimilar-compound split 
│
├── configs/                  # Configuration files with optimized hyperparameters and training settings
│   ├── rs_config.yaml          # Config for random-split based ECLIPSE and baseline models
│   ├── dcs_config.yaml         # Config for dissimilar-compound-split based ECLIPSE and baseline models
│   └── fds_config.yaml         # Config for fully-dissimilar-split based ECLIPSE and baseline models
│
├── src/                      # Source code
│   ├── data_loader.py          # Data loading & preprocessing functions
│   ├── model.py                # HGT-based model architecture
│   ├── train.py                # Training pipeline script
│   ├── predict.py              # Prediction script
│   └── utils.py                # Utility/helper functions
│
├── outputs/                 # Model outputs (predictions, performance scores etc.)
│
├── requirements.txt           # Python dependencies (pip-based setup)
├── environment.yml            # Conda environment specification
├── workflow.png               # Workflow diagram of the ECLIPSE framework
├── README.md                  # Project documentation (this file)
└── LICENSE                    # License information
```
## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/HUBioDataLab/ECLIPSE.git
cd ECLIPSE
```

**2. Set up the environment**

Option 1: Using conda (recommended)
```bash
conda env create -f environment.yml
conda activate eclipse
```

Option 2: Using pip
```bash
pip install -r requirements.txt
```

## ⚙️ Training the ECLIPSE Model

⚠️ Ensure that the graph files are correctly placed in the `data/crossbar_kg/` directory before starting training. For detailed instructions, see `data/README.md`.

To train the ECLIPSE model, run the `train.py` script with an example command:

```bash
python train.py -s dcs -pl dp -cr selformer -sm -sp
```
**Arguments:**
- `-s, --split`: Data split -> `fds` (fully_dissimilar_split), `dcs` (dissimilar_compound_split), or `rs` (random_split)
- `-pl, --prediction-layer`: Prediction layer -> `dp` (dot_product) or `fc` (fully_connected)
- `-cr, --compound-representation`: Compound representation -> `ecfp4` or `selformer`
- `-nw, --num-workers`: Number of data loading workers (default: 2)
- `-nt, --num-threads`: Number of CPU threads (default: 2)
- `-o, --output-dir`: Output directory (default: `outputs/`)
- `-c, --config`: Path to config file (default: generated from other args)
- `-sm,--save-model`: Save trained model to `saved_models/` if flagged
- `-sp,--save-predictions`: Save test set predictions to `--output-dir` if flagged
- `-b, --baseline`: Use baseline model (no HGT layers, only linear layers) if flagged

Test set performance results will be saved to the specified `--output-dir`.

## 🎯 Making Predictions

To generate bioactivity value predictions using a trained ECLIPSE model, run the `predict.py` script with the desired split, prediction layer, and compound representation.

An example command:

```bash
python predict.py -s dcs -pl dp -cr selformer -p P11309
```
**Arguments:**
- `-s, --split`: Data split -> `fds` (fully_dissimilar_split), `dcs` (dissimilar_compound_split), or `rs` (random_split)
- `-pl, --prediction-layer`: Prediction layer -> `dp` (dot_product) or `fc` (fully_connected)
- `-cr, --compound-representation`: Compound representation -> `ecfp4` or `selformer`
- `-o, --output-dir`: Output directory (default: `outputs/`)

Use only **one** of the following options:
- `-pid, --protein_id`: UniProt ID for protein-centric prediction (predict bioactivity values for the given protein against all compounds in the CROssBAR KG)
- `-cid", --compound_id`: Compound ID for compound-centric prediction (predict bioactivity values for the given compound against all proteins in the CROssBAR KG)
- `-c, --custom`: Path to a CSV file for a custom set (predict bioactivity values for the specified protein-compound pairs in the CROssBAR KG). The file must have two columns with headers: `compound_id`, `protein_id`

Predictions will be saved as a CSV file in the specified `--output-dir`. 


## 📄 License
Copyright (C) 2025 HUBioDataLab

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.



