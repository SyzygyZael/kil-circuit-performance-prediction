import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, GCNConv, SAGEConv, GINConv
import torch

class GCNwMLPHead(nn.Module):
    def __init__(self, dataset, num_classes = 1):
        super().__init__()

        self.layer1 = GCNConv(dataset.num_node_features, 128)
        self.layer2 = GCNConv(128, 64)

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x, batch, edge_index):
        x = self.layer1(x, edge_index).relu()
        x = self.layer2(x, edge_index).relu()

        x = global_mean_pool(x, batch)

        return self.mlp(x)

class GATwMLPHead(nn.Module):
    def __init__(self, dataset, num_classes = 1):
        super().__init__()

        self.layer1 = GATConv(dataset.num_node_features, 128)
        self.layer2 = GATConv(128, 64)

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x, batch, edge_index):
        x = self.layer1(x, edge_index).relu()
        x = self.layer2(x, edge_index).relu()

        x = global_mean_pool(x, batch)

        return self.mlp(x)

class SAGEwMLPHead(nn.Module):
    def __init__(self, dataset, num_classes = 1):
        super().__init__()

        self.layer1 = SAGEConv(dataset.num_node_features, 128)
        self.layer2 = SAGEConv(128, 64)

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x, batch, edge_index):
        x = self.layer1(x, edge_index).relu()
        x = self.layer2(x, edge_index).relu()

        x = global_mean_pool(x, batch)

        return self.mlp(x)

class GINwMLPHead(nn.Module):
    def __init__(self, dataset, num_classes = 1):
        super().__init__()

        nn1 = nn.Sequential(nn.Linear(dataset.num_node_features, 128), nn.ReLU())
        nn2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

        self.layer1 = GINConv(nn1)
        self.layer2 = GINConv(nn2)

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x, batch, edge_index):
        x = self.layer1(x, edge_index).relu()
        x = self.layer2(x, edge_index).relu()

        x = global_mean_pool(x, batch)

        return self.mlp(x)