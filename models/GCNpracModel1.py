from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import DataLoader
from GCNmsgPass import GCNMsgPass

# model definition
class GCN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        
        # create layers
        self.layer1 = GCNConv(dataset.num_node_features, 64)
        self.layer2 = GCNConv(64, 64)
        self.layer3 = GCNMsgPass(64, 64)

        # output layer
        self.lin = nn.Linear(64, dataset.num_classes)
    
    def forward(self, x, edge_index, batch):
        # pass features to layers + dropout
        x = self.layer1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training = self.training)

        x = self.layer2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training = self.training)

        x = self.layer3(x, edge_index).relu()

        # one embedding per node --> one embedding per graph
        # basically averages all node embeddings belonging to the same graph
        x = global_mean_pool(x, batch)

        return self.lin(x)
    
