# 📂 Data Directory

This folder contains all data resources required to train and evaluate ECLIPSE models. The structure is organized as follows:
```
data/
│
├── node_index/
│   ├── node_index_cmp-ecfp4.json
│   └── node_index_cmp-selformer.json
│
├── train_test_samples/
│   ├── fds_train_test_edges_ecfp4.pt
│   ├── fds_train_test_edges_selformer.pt
│   ├── dcs_train_test_edges_ecfp4.pt
│   ├── dcs_train_test_edges_selformer.pt
│   ├── rs_train_test_edges_ecfp4.pt
│   └── rs_train_test_edges_selformer.pt
│
└── crossbar_kg/   (must be downloaded separately)
    ├── crossbar-kg_ecfp4.pt
    └── crossbar-kg_selformer.pt
```

**The integrated biomedical knowledge graph** objects can be downloaded from this [CROssBAR KG](https://drive.google.com/file/d/1AqVshFKkrNoEZstLILiLzuDeIqMkgpji/view?usp=sharing) link.

Once downloaded, place the `crossbar_kg/` folder (with the .pt files inside) into the `data/` directory. This ensures that train.py and predict.py can access the graph data correctly.

