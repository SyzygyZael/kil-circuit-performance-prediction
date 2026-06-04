import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNMsgPass(MessagePassing):
    def __init__(self, inChannels, outChannels):
        super().__init__(aggr = 'add')

        # scale input dim "inChannels" to target output dim (outChannels)
        self.lin = Linear(inChannels, outChannels, bias = False)

        # create bias
        self.bias = Parameter(torch.empty(outChannels))

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edgeIndex):
        # x shape [N, inChannels]
        # edge_index shape [2, E]

        # add self-loops to the adjacency matrix
        edgeIndex, _ = add_self_loops(edgeIndex, num_nodes=x.size(0))

        # linearly transform node feature matrix
        x = self.lin(x)

        # compute normalization factor
        row, col = edgeIndex
        deg = degree(col, x.size(0), dtype=x.dtype)
        degInvSqrt = deg.pow(-0.5)
        degInvSqrt[degInvSqrt == float('inf')] = 0
        norm = degInvSqrt[row] * degInvSqrt[col]

        # begin message propagation
        out = self.propagate(edgeIndex, x = x, norm=norm)

        # apply final bias vector
        out = out + self.bias

        return out
    
    def message(self, x_j, norm):
        # x_j shape [E, outChannels] and is the feature vector of the source node

        # normalize node features
        return norm.view(-1, 1) * x_j