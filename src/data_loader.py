from torch_geometric.loader import LinkNeighborLoader


def get_data_loaders(
    graph_data,
    train_batch_size,
    num_workers,
    edge_name,
    num_neighbors = [-1]
):


    # Train loader
    train_loader = LinkNeighborLoader(
        graph_data,
        num_neighbors=num_neighbors,  
        shuffle=True,
        directed=False,
        edge_label=graph_data[edge_name]["train_label"],
        edge_label_index=(edge_name, graph_data[edge_name]["train_data"]),
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test loader
    test_loader = LinkNeighborLoader(
        graph_data,
        num_neighbors=num_neighbors,
        shuffle=False,
        directed=False,
        edge_label=graph_data[edge_name]["test_label"],
        edge_label_index=(edge_name, graph_data[edge_name]["test_data"]),
        batch_size=len(graph_data[edge_name]["test_label"]),
    )

    return train_loader, test_loader



