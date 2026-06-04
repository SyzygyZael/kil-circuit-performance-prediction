import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import torch

class GATModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.layer1 = GATConv(dataset.num_node_features, 64, heads=8, concat=True)
        # because we have 8 heads that will concat, the ouput becomes 128*8
        self.layer2 = GATConv(64*8, 128, heads = 1, concat=True)

        # multiply by 2 because of xCombined concatenating one last time
        self.lin = nn.Linear((64*8 + 128) * 2, dataset.num_classes)
    
    def forward(self, x, edge_index, batch):
        x1 = self.layer1(x, edge_index).relu()
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x2 = self.layer2(x1, edge_index).relu()
        x2 = F.dropout(x2, p=0.2, training=self.training)

        xCombined = torch.cat([x1, x2], dim=1)

        # pooling that is aware of the levels of importance of different nodes
        xMean = global_mean_pool(xCombined, batch)
        xMax = global_max_pool(xCombined, batch)

        xFinal = torch.cat([xMean, xMax], dim = 1)

        return self.lin(xFinal)
    
    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif (isinstance(layer, nn.Sequential)):
                for sub_layer in layer:
                    if hasattr(sub_layer, 'reset_parameters'):
                        sub_layer.reset_parameters()
