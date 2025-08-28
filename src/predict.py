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
    model = eclipse.ECLIPSE(graph, model_cfg, pred_layer)  

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
    out_path = output_dir / f"{model_name}_{protein_id}_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tsv"
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
    out_path = output_dir / f"{model_name}_{compound_id}_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tsv"
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






















#-------



def use_case(prot_label, comp_edge_ind, node_js_prot_ind, node_js_cmp=None):
    print(f"\n---{prot_label} predictions---\n")
    prot_edge_index = torch.as_tensor([comp_edge_ind,[node_js_prot_ind]*len(comp_edge_ind)], device=device)
    print(prot_edge_index[0][:4])
    print(prot_edge_index[1][:4])
    
    print(data["compound"].x[prot_edge_index[0][:4]])
    print(data["protein"].x[prot_edge_index[1][:4]])


    data["compound","Chembl","protein"].prot_edge_index = prot_edge_index
    print(data["compound","Chembl","protein"].prot_edge_index.size())
    
    # prot_loader = LinkNeighborLoader(data, num_neighbors=[-1], shuffle=False, directed=False, 
    #                             edge_label_index=(("compound", "Chembl", "protein"), data[("compound", "Chembl", "protein")].prot_edge_index), batch_size=len(comp_edge_ind))

    # for batch in prot_loader:
    #     # prot_pred = model(batch.x_dict, batch.edge_index_dict, batch[("compound", "Chembl", "protein")].edge_label_index)
    #     prot_pred = model(batch.x_dict, batch.edge_index_dict, batch[("compound", "Chembl", "protein")].prot_edge_index)
       
    #     print(prot_pred.size())
    #     print(prot_pred[:10])

    #     print(batch["compound"].x[batch[("compound", "Chembl", "protein")].prot_edge_index[0][:4]])
    #     print(batch["protein"].x[batch[("compound", "Chembl", "protein")].prot_edge_index[1][:4]])
        
    #     # print(batch["compound"].x[batch[("compound", "Chembl", "protein")].edge_label_index[0][:4]])
    #     # print(batch["protein"].x[batch[("compound", "Chembl", "protein")].edge_label_index[1][:4]])

 
    prot_pred = model(data.x_dict, data.edge_index_dict, data[("compound", "Chembl", "protein")].prot_edge_index)
    print(prot_pred.size())
    print(prot_pred[:10])

    if node_js_cmp == None:
        df_prot_pred = pd.DataFrame({"compound_index":prot_edge_index[0], "protein_index":prot_edge_index[1], f"{prot_label}_predictions":prot_pred.numpy()})
    else:
        df_prot_pred = pd.DataFrame({"compound_id":[k for k,v in node_js_cmp.items() if v in prot_edge_index[0]], "compound_index":prot_edge_index[0], "protein_index":prot_edge_index[1], f"{prot_label}_predictions":prot_pred.numpy()})
    # df_prot_pred.to_csv(fr"{path}/model/{split}/{kg}/{protfam}/use_case/{kg}_hgt_tr-ap-ks_btq-gpd_selformer_{prot_label}-predictions_{protfam}_{split}_wd0_v1_030525_full-graph.tsv", sep="\t", index=None)
    df_prot_pred.to_csv(fr"{path}/RS/hgt_DP-RightLeftLin_self_lr0.001-hid128-hd16-lyr1-batch1024-wd0-ep250_ReLu_LNL-1_{prot_label}-predictions_{protfam}_{split}_130525.tsv", sep="\t", index=None)





split = "random_split" #"fully_dissimilar_split" #"dissimilar_compound_split"
kg = "merged-graph-davis"
#path = r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction"
path = r"/media/ubuntu/8TB/heval/Chapter3_DTI_Prediction/"
protfam = "transferases"

#-----call HeteroData() object--------
data = torch.load(fr"{path}/model/{split}/{kg}/{protfam}/data/{kg}_heterodata_unscaled_transformer-apaac-ksep_bioteque-gpd_{protfam}.pt", map_location=torch.device('cpu'))
del data["protein","rev_Chembl","compound"]
data["compound","Chembl","protein"].edge_index = data["compound","Chembl","protein"].edge_index[:,:0]

print(data)
print(data.num_nodes)
print(data.num_edges)

import json
with open (fr"{path}/model/{kg}/data/node_index_transformer-apaac-ksep_bioteque-gpd.json") as js:
    node_js = json.load(js) 
node_js_comps = {key: node_js[key] for key in node_js.keys() if key.startswith(("CHEMBL","CID"))}
print(len(node_js_comps))


# test_data = data[("compound", "Chembl", "protein")].test_data
# print(len(test_data[0]), len(set(test_data[0].tolist())))


# print(test_data[0][:4])
# print(test_data[1][:4])

# print(data["compound"].x[[20511, 20323, 20958, 21523]])
# print(data["protein"].x[[10099,  4245, 11605, 11605]])

# print(data["compound"].x[test_data[0][:4]])
# print(data["protein"].x[test_data[1][:4]])


# use_case_comps = {key: node_js_comps[key] for key in node_js_comps.keys() if node_js_comps[key] in test_data[0]}
# print(len(use_case_comps))
# rev_use_case_comps = {(v,k) for k,v in use_case_comps.items()}

import x4_hgt_model_train_prediction_hgt as eclipse

model = eclipse.ECLIPSE(data, hidden_channels=128, num_heads=16, num_layers=1, method=["DP", "RightLeftLin"], dropout=0)
#model = HGT(data, hidden_channels=32, num_heads=8, num_layers=3)

model.load_state_dict(torch.load(fr"{path}/RS/hgt_DP-RightLeftLin_self_lr0.001-hid128-hd16-lyr1-batch1024-wd0-ep250_ReLu_LNL-1.pt", map_location=torch.device('cpu'))) 
#model.load_state_dict(torch.load(fr"{path}/model/{split}/{kg}/{protfam}/{kg}_hgt_tr-ap-ks_btq-gpd_selformer_{protfam}_{split}_wd0_v1.pt", map_location=torch.device('cpu'))) 
model.eval()
with torch.no_grad():
    # test_loader = LinkNeighborLoader(data, num_neighbors=[-1], shuffle=False, directed=False, edge_label=data[("compound", "Chembl", "protein")]["test_label"],
    #                             edge_label_index=(("compound", "Chembl", "protein"), data[("compound", "Chembl", "protein")]["test_data"]), batch_size=len(data[("compound", "Chembl", "protein")]["test_label"]))#,num_workers=2,pin_memory=True)

    # print("\n---Test set predictions---\n")
    # # for batch in test_loader:
    # #     ts_pred = model(batch.x_dict, batch.edge_index_dict, batch[("compound", "Chembl", "protein")].edge_label_index)
    # #     print(ts_pred.size())
    # #     print(ts_pred[:10])
        
    # #     print(batch["compound"].x[batch[("compound", "Chembl", "protein")].edge_label_index[0][:4]])
    # #     print(batch["protein"].x[batch[("compound", "Chembl", "protein")].edge_label_index[1][:4]])


    # # ts_pred = model(data.x_dict, data.edge_index_dict, test_data)

    # # df_ts_pred = pd.DataFrame({"compound_index":test_data[0], "protein_index":test_data[1], "true_values":data[("compound", "Chembl", "protein")].test_label, "predictions":ts_pred})
    # # df_ts_pred.to_csv(fr"{path}/model/{split}/{kg}/{protfam}/predictions/{kg}_hgt_tr-ap-ks_btq-gpd_selformer_test_predictions-local_{protfam}_{split}_wd0_v1-2.tsv", sep="\t", index=None)

    # ts_pred = model(data.x_dict, data.edge_index_dict, test_data)
    # print(ts_pred.size())
    # print(ts_pred[:20])
    
    
        
    "use-case part"
    sorted_comps = sorted(list(node_js_comps.values()))
    
    # tp53_preds = use_case("tp53", sorted_comps, node_js["P04637"])
    her3_preds = use_case("her3", sorted_comps, node_js["P21860"])
    # trkb_preds = use_case("trkb", sorted_comps, node_js["Q16620"])
    pim1_preds = use_case("pim1", sorted_comps, node_js["P11309"]) 
    # oat_preds = use_case("oat", sorted_comps, node_js["P04181"]) 
    # cpt2_preds = use_case("cpt2", sorted_comps, node_js["P23786"]) 

    
#     df_merge = pd.read_csv(r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction/model/dissimilar_compound_split/merged-graph-davis/transferases/use_case/transferases_dissimilar_compound_split_O60674-P23458_common_training_datapoints.tsv", sep="\t")
#     node_js_df_merge_comps = [node_js[key] for key in df_merge.compound_id]
#     print(len(node_js_df_merge_comps))
#     print(node_js_df_merge_comps[:10])
    
#     O60674_preds = use_case("O60674", node_js_df_merge_comps, node_js["O60674"])
#     P23458_preds = use_case("P23458", node_js_df_merge_comps, node_js["P23458"])

# print(df_pred.iloc[2,4][:20])
# print(model.state_dict()['right_linear.weight'])

# df = pd.read_csv(r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction/datasets/human_bioactivity_datasets/dissimilar_compound_split/transferases_test_human.tsv", sep="\t")
# # df = pd.read_csv(r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction/datasets/human_bioactivity_datasets/dissimilar_compound_split/transferases_train_human.tsv", sep="\t")
# df_x_grp = df.groupby(by = ["target_id"]).count().reset_index().sort_values(by="pchembl_value", ascending=False).reset_index(drop=True)
# print(df_x_grp.head(10))
# # print(len(df), len(df_x_grp))

# unique_comps = []
# for prot in df_x_grp[:10].target_id.unique():
#     df_prot = df.loc[df.target_id==prot]
#     print(len(df_prot),len(df_prot.compound_id.unique()))
#     unique_comps.append(set(df_prot.compound_id.unique()))

# pchembl_diff = []
# for i in range(6,7):#len(unique_comps)):  #2,3):#
#     for j in range(7,8):#i+1, len(unique_comps)):  #9,10):#
#         cmp_common = unique_comps[i] & unique_comps[j]
#         print(f"{i}: {len(unique_comps[i])}, {j}: {len(unique_comps[j])}, common-cmps: {len(cmp_common)}")
#         if len(cmp_common) != 0:
#             df_prot1 = df.loc[df.target_id==df_x_grp.iloc[i,0]]
#             df_prot2 = df.loc[df.target_id==df_x_grp.iloc[j,0]]
#             df_merge = df_prot1.merge(df_prot2, on="compound_id")
#             df_merge.loc[:,"pchembl_diff"] = [x-y for x,y in zip(df_merge.pchembl_value_x,df_merge.pchembl_value_y)]
#             print(len(df_merge))
#             print(df_merge.loc[:5,["compound_id","pchembl_value_x","pchembl_value_y","pchembl_diff"]])
#             pchembl_diff.extend(df_merge.pchembl_diff)
#             df_merge.to_csv(r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction/model/dissimilar_compound_split/merged-graph-davis/transferases/use_case/transferases_dissimilar_compound_split_P48730-P49674_common_test_datapoints.tsv", sep="\t", index=None)
            
# #             # break
        
# # print(len(pchembl_diff))
# plt.hist(pchembl_diff, bins=100, alpha=0.80)
# # plt.show()


# train_label = data[("compound", "Chembl", "protein")].train_label
# tr_pchembl_median = np.median(train_label)
# print(tr_pchembl_median)


'train datapoints'
# df_merge = pd.read_csv(r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction/model/dissimilar_compound_split/merged-graph-davis/transferases/use_case/transferases_dissimilar_compound_split_O60674-P23458_common_training_datapoints.tsv", sep="\t")
# df_O60674 = pd.read_csv(r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction/model/dissimilar_compound_split/merged-graph-davis/transferases/use_case/merged-graph-davis_hgt_tr-ap-ks_btq-gpd_selformer_O60674-predictions_transferases_dissimilar_compound_split_wd0_v1-2.tsv", sep="\t")
# df_P23458 = pd.read_csv(r"/Users/hevalatas/Desktop/HA/Education/PhD/Thesis_Study/Chapter3_DTI_Prediction/model/dissimilar_compound_split/merged-graph-davis/transferases/use_case/merged-graph-davis_hgt_tr-ap-ks_btq-gpd_selformer_P23458-predictions_transferases_dissimilar_compound_split_wd0_v1-2.tsv", sep="\t")

# pchembl_diff1 = df_merge.pchembl_diff
# pchembl_diff2 = [i-j for i,j in zip(df_O60674.O60674_predictions,df_P23458.P23458_predictions)]
# for pchembl_diff in [pchembl_diff2]:#,pchembl_diff2]:
#     plt.hist(pchembl_diff, bins=100, alpha=0.80)

# 'O60674'
# true_labels = df_merge.pchembl_value_x.values
# predictions = df_O60674.O60674_predictions.values

# ts_loss = rmse(true_labels, predictions)**2
# ts_rmse = rmse(true_labels, predictions)
# ts_spearman = spearman(true_labels, predictions)
# ts_mcc = mcc(true_labels, predictions, tr_pchembl_median)

# print(ts_loss, ts_rmse, ts_spearman, ts_mcc)

# 'P23458'
# true_labels = df_merge.pchembl_value_y.values
# predictions = df_P23458.P23458_predictions.values

# ts_loss = rmse(true_labels, predictions)**2
# ts_rmse = rmse(true_labels, predictions)
# ts_spearman = spearman(true_labels, predictions)
# ts_mcc = mcc(true_labels, predictions, tr_pchembl_median)

# print(ts_loss, ts_rmse, ts_spearman, ts_mcc)
