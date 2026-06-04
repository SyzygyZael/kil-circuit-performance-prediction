import pandas as pd
from pathlib import Path
import torch
import pickle
from torch_geometric.data import Data
import torch.nn.functional as F

# This class is responsible for parsing the subgraph data from the CktGNN dataset. It reads the pickled data and CSV file, 
# and provides methods to access the training and validation data, as well as a method to summarize the circuit information 
# in a human-readable format.

SUBG_NODE = {
            0: ['In'],
            1: ['Out'],
            2: ['R'],
            3: ['C'],
            4: ['R','C'],
            5: ['R','C'],
            6: ['+gm+'],
            7: ['-gm+'],
            8: ['+gm-'],
            9: ['-gm-'],
            10: ['C', '+gm+'],
            11: ['C', '-gm+'],
            12: ['C', '+gm-'],
            13: ['C', '-gm-'],
            14: ['R', '+gm+'],
            15: ['R', '-gm+'],
            16: ['R', '+gm-'],
            17: ['R', '-gm-'],
            18: ['C', 'R', '+gm+'],
            19: ['C', 'R', '-gm+'],
            20: ['C', 'R', '+gm-'],
            21: ['C', 'R', '-gm-'],
            22: ['C', 'R', '+gm+'],
            23: ['C', 'R', '-gm+'],
            24: ['C', 'R', '+gm-'],
            25: ['C', 'R', '-gm-']
        }

class CktGNNSubGParser:
    def __init__(self, data_path):
        with open(data_path + '/ckt_bench_101.pkl', "rb") as f:
            self.data = pickle.load(f)
        self.df = pd.read_csv(data_path + "/perform101.csv")
    
        self.train_data = self.data[0]
        self.val_data = self.data[1]
    
    def getRawTrainingData(self):
        return self.train_data
    
    def getRawValidationData(self):
        return self.val_data
    
    def getRawData(self):
        return self.train_data + self.val_data
    
    def getAuthorFormat(self, dataSplit, circuitID):
        g_subg = dataSplit[circuitID][0]
        g_comp = dataSplit[circuitID][1]

        num_subgraphs = g_subg.vcount()
        num_nodes = g_comp.vcount()
        over = [num_subgraphs, num_nodes]

        nodeSummaries = []
        for j in range(num_subgraphs):
            subg_type = g_subg.vs[j]['type']
            predecessors = g_subg.predecessors(j)
            feats = g_subg.vs[j]['subg_nfeats']

            nodeSummaries.append([
                subg_type,
                j,
                len(predecessors),
                predecessors,
                -1,
                feats[1:-1],
                -1
            ])
        
        circuitPerformance = self.df.iloc[circuitID].values.tolist()

        return [over, nodeSummaries, circuitPerformance[2:5]]
    
    def summarizeCircuit(self, dataSplit, circuitID):
        nodeSummaries = self.getAuthorFormat(dataSplit, circuitID)
        yvals = [self.df.iloc[circuitID].values.tolist()]
        ylabs = self.df.columns.tolist()
        yDF = pd.DataFrame(yvals, columns=ylabs)
        
        print(f"Number of Subgraphs: {nodeSummaries[0][0]}")
        print(f"Number of Nodes: {nodeSummaries[0][1]}\n")
        
        for i, node in enumerate(nodeSummaries[1]):
            subg_type = node[0]
            components = SUBG_NODE[subg_type]
            feats = node[5]
            print(f"Subgraph {i}:")
            print(f"  Type: {components}")
            print(f"  Incoming Edges: {node[3]}")
            for k, comp in enumerate(components):
                val = feats[k] if k < len(feats) else 'N/A'
                print(f"  {comp} value: {val}")
            print()

        print(f"Circuit Specs:")
        print(yDF)
        
        return ''
    
    def toDataObject(self, authorFormat, performanceTargets=None):
        nodesList = authorFormat[1]
        if (len(nodesList) == 2):
            nodesList = nodesList[1]
        nodeFeatures = []

        for node in nodesList:
            subgType = node[0]
            nodeID = node[1]
            inputSources = node[3]
            specs = node[5]

            # one hot encode the subgraph types
            subgTensor = torch.tensor([subgType], dtype=torch.long)
            subgOneHot = F.one_hot(subgTensor, num_classes=len(SUBG_NODE)).float().squeeze(0)

            # every node should a have the same parameter width
            paramTensor = torch.tensor(specs, dtype=torch.float)
            if (len(paramTensor) < 3):
                paddingSize = 3 - len(paramTensor)
                paramTensor = F.pad(paramTensor, (0, paddingSize), "constant", 0.0)
            
            # combine to make the final node features
            nodeFeatureRow = torch.cat([subgOneHot, paramTensor], dim=0)
            nodeFeatures.append(nodeFeatureRow)

        # PyTorch Geometric Tensors
        x = torch.stack(nodeFeatures, dim=0)

        # edge direction matrix of size [2, numEdges]
        edgeIndex = torch.tensor([inputSources, [nodeID for i in range(len(inputSources))]], dtype=torch.long)

        # add simulation labels
        if (performanceTargets is not None):
            y = torch.tensor(performanceTargets, dtype=torch.float).view(1, -1)
        else:
            y = torch.zeros((1, 4), dtype=torch.float)
        
        return Data(x = x, edge_index=edgeIndex, y = y)
    
    
