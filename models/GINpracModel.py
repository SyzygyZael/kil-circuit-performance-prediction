from torch_geometric.nn import GINConv, GlobalAttention
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch

# GINE models take edge features into account for datasets that provide them.
# This means that they are able to see relationships between nodes.

class GINModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.mlp1 = nn.Sequential(nn.Linear(dataset.num_node_features, 128), nn.ReLU(), nn.Linear(128, 128))
        self.layer1 = GINConv(self.mlp1)

        self.mlp2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.layer2 = GINConv(self.mlp2)

        # creates an importance score for each node
        self.pooling_gate = nn.Linear(256, 1)
        self.attention_pool = GlobalAttention(gate_nn=self.pooling_gate)

        self.lin = nn.Linear(128*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x1 = self.layer1(x, edge_index).relu()
        x1 = F.dropout(x1, p=0.2, training = self.training)

        x2 = self.layer2(x1, edge_index).relu()
        x2 = F.dropout(x2, p=0.2, training = self.training)

        x_cat = torch.cat([x1, x2], dim=1)

        x_pooled = self.attention_pool(x_cat, batch)

        return self.lin(x_pooled)


