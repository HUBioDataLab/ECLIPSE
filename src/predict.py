import argparse
import torch
import pandas as pd
import json
from model import ECLIPSE
from utils import load_config
from datetime import datetime

from pathlib import Path
# Resolve paths relative to repo root
ROOT_DIR = Path(__file__).resolve().parents[1]   # src → repo root


def parse_arguments():
    parser = argparse.ArgumentParser(description="ECLIPSE prediction script\n\nInput protein ids and compounds ids should be present in the training graph.\n")
    parser.add_argument("-s", "--split", type=str, required=True, choices=["fds", "dcs", "rs"],
                        help="Data split: fds (fully_dissimilar_split), dcs (dissimilar_compound_split), or rs (random_split)")
    parser.add_argument("-pl", "--prediction-layer", type=str, required=True, choices=["dp", "fc"],
                        help="Prediction layer: dp (dot_product) or fc (fully_connected)")
    parser.add_argument("-cr", "--compound-representation", type=str, required=True, choices=["ecfp4", "selformer"],
                        help="Compound representation: ecfp4 or selformer")
    parser.add_argument("-o", '--output-dir', type=str, default= ROOT_DIR / 'outputs', help='Directory to save prediction results')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-pid", "--protein_id", type=str, help="UniProt ID for protein-centric prediction" 
                       "Use only one option among --protein_id, --compound_id, or --custom")
    group.add_argument("-cid", "--compound_id", type=str, help="Compound ID for compound-centric prediction"
                       "Use only one option among --protein_id, --compound_id, or --custom")
    group.add_argument("-c", "--custom", type=str, metavar='CSV_FILE',
                       help="Path to a CSV file for custom prediction. The file must have two columns with headers: compound_id, protein_id"
                       "Use only one option among --protein_id, --compound_id, or --custom")
    return parser.parse_args()



def load_node_json(compound_representation):
    node_json_map = {
        "selformer": "node_index_cmp-selformer.json",
        "ecfp4": "node_index_cmp-ecfp4.json",
    }
    json_file = node_json_map.get(compound_representation)
    if json_file is None:
        raise ValueError(f"No node json mapping for compound_representation: {compound_representation}")
    with open(ROOT_DIR / "data" / "node_index" / json_file) as js:
        return json.load(js)


def get_model(model_name, config, graph, pred_layer, device):
    # Load configuration
    print(f"Loading configuration from {config}")
    config = load_config(config)
    model_cfg = config["model"][model_name]

    # Initialize the model and load weights
    model = ECLIPSE(graph, model_cfg, pred_layer)  

    model.load_state_dict(torch.load(ROOT_DIR / "saved_models" / f'{model_name}.pt', map_location=device))
    model.eval()
    return model


def predict_protein(model_name, protein_id, node_js, graph, model, output_dir):
    if protein_id not in node_js:
        raise ValueError(f"Protein ID {protein_id} not found in node json.")
    protein_idx = node_js[protein_id]
    compound_ids = [k for k in node_js if k.startswith(("CHEMBL", "CID"))]
    compound_indices = [node_js[cid] for cid in compound_ids]
    query_edge_index = torch.tensor([compound_indices, [protein_idx]*len(compound_indices)], dtype=torch.long)
    graph[("compound", "Chembl", "protein")].query_edge_index = query_edge_index

    with torch.no_grad():
        preds = model(graph.x_dict, graph.edge_index_dict, graph[("compound", "Chembl", "protein")].query_edge_index)
    df = pd.DataFrame({
        "compound_id": compound_ids,
        "predicted_activity (-log[M])": preds.numpy().round(2)
    })
    out_path = output_dir / f"{model_name}_{protein_id}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved predictions to {out_path}")


def predict_compound(model_name, compound_id, node_js, graph, model, output_dir):
    if compound_id not in node_js:
        raise ValueError(f"Compound ID {compound_id} not found in node json.")
    compound_idx = node_js[compound_id]
    protein_ids = [k for k in node_js if (k.startswith("H0Y") or (not k.startswith(("CHEMBL", "CID", "DB", "HP:", "hsa", "R-", "Orphanet:", "EFO:", "MONDO:", "H0"))))]
    protein_indices = [node_js[pid] for pid in protein_ids]
    print(f"Predicting for compound {compound_id} on {len(protein_ids)} proteins.")
    query_edge_index = torch.tensor([[compound_idx]*len(protein_indices), protein_indices], dtype=torch.long)
    graph[("compound", "Chembl", "protein")].query_edge_index = query_edge_index
    
    with torch.no_grad():
        preds = model(graph.x_dict, graph.edge_index_dict, graph[("compound", "Chembl", "protein")].query_edge_index)
    df = pd.DataFrame({
        "protein_id": protein_ids,
        "predicted_activity (-log[M])": preds.numpy().round(2)
    })
    out_path = output_dir / f"{model_name}_{compound_id}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved predictions to {out_path}")


def predict_custom(model_name, csv_file, node_js, graph, model, output_dir):
    df = pd.read_csv(csv_file)
    if "compound_id" not in df.columns or "protein_id" not in df.columns:
        raise ValueError("CSV file must contain 'compound_id' and 'protein_id' columns.")
    compound_ids = df["compound_id"].tolist()
    protein_ids = df["protein_id"].tolist()

    if len(compound_ids) != len(protein_ids):
        raise ValueError("Size of compound_ids and protein_ids must be equal.")
    for cid in compound_ids:
        if cid not in node_js:
            raise ValueError(f"Compound ID {cid} not found in node index json file.")
    for pid in protein_ids:
        if pid not in node_js:
            raise ValueError(f"Protein ID {pid} not found in node index json file.")
    compound_indices = [node_js[cid] for cid in compound_ids]
    protein_indices = [node_js[pid] for pid in protein_ids]
    query_edge_index = torch.tensor([compound_indices, protein_indices], dtype=torch.long)
    graph[("compound", "Chembl", "protein")].query_edge_index = query_edge_index

    with torch.no_grad():
        preds = model(graph.x_dict, graph.edge_index_dict, graph[("compound", "Chembl", "protein")].query_edge_index)
    df = pd.DataFrame({
        "compound_id": compound_ids,
        "protein_id": protein_ids,
        "predicted_activity (-log[M])": preds.numpy().round(2)
    })
    out_path = output_dir / f'{model_name}_custom_compound_protein_pairs_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tsv'
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved predictions to {out_path}")

def main():
    args = parse_arguments()

    # Device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model_name = fr'{args.split}_eclipse_{args.prediction_layer}_{args.compound_representation}'
    config_path = ROOT_DIR / "configs" / f"{args.split}_config.yaml"

    # Load data, model, and node json
    crossbar_kg = torch.load(ROOT_DIR / "data" / "crossbar_kg" / f"crossbar-kg_{args.compound_representation}.pt", map_location=device)
    node_js = load_node_json(args.compound_representation)
    model = get_model(model_name, config_path, crossbar_kg, args.prediction_layer, device)

    if args.protein_id:
        predict_protein(model_name, args.protein_id, node_js, crossbar_kg, model, args.output_dir)
    elif args.compound_id:
        predict_compound(model_name, args.compound_id, node_js, crossbar_kg, model, args.output_dir)
    elif args.custom:
        predict_custom(model_name, args.custom, node_js, crossbar_kg, model, args.output_dir)

if __name__ == "__main__":
    main()





















