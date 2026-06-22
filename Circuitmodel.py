import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import torch

class CircuitModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.layer1 = GATConv(dataset.num_node_features, 128, heads=4, concat=True)
        self.layer2 = GATConv(128*4, 256, heads=4, concat=True)

        self.fc1 = nn.Linear((256*4) + (128*4), 128)
        self.fc2 = nn.Linear(128, 3)

        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index, batch):
        # implement layers
        x1 = self.relu(self.layer1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = self.relu(self.layer2(x1, edge_index))

        xComb = torch.cat([x1, x2], dim=1)

        # pooling
        xPooled = global_mean_pool(xComb, batch)

        # map out to raw regression predictions
        out = self.relu(self.fc1(xPooled))
        out = self.dropout(out)
        prediction = self.fc2(out)

        return prediction
    
    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif (isinstance(layer, nn.Sequential)):
                for sub_layer in layer:
                    if hasattr(sub_layer, 'reset_parameters'):
                        sub_layer.reset_parameters()

