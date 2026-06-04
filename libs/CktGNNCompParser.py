import pandas as pd
import torch_geometric.data.data as pyg_data
from pathlib import Path
import torch
import pickle

# This class is responsible for parsing the component data from the CktGNN Pygraph dataset. It reads the pickled data and CSV file, 
# and provides methods to access the training and validation data, as well as a method to summarize the circuit information 
# in a human-readable format.

class CktGNNCompParser:
    def __init__(self, data_path):
        with open(data_path + '/ckt_bench_101_pygraph.pkl', "rb") as f:
            self.data = pickle.load(f)
        self.df = pd.read_csv(data_path + "/perform101.csv")
        
        self.NODE_TYPE = {
            0: 'R',
            1: 'C',
            2: '+gm+',
            3: '-gm+',
            4: '+gm-',
            5: '-gm-',
            6:'sudo_in',
            7:'sudo_out',
            8: 'In',
            9:'Out'
        }

        self.train_data = self.data[0]
        self.val_data = self.data[1]


    def getTrainingData(self):
        return self.train_data
    
    def getValidationData(self):
        return self.val_data

    def getRawCircuitData(self, data, row):
        circuit = data[row]
        r = object.__getattribute__(circuit, '__dict__')
        label = self.df.iloc[row].values.tolist()

        return {
            'x':             r['x'],
            'edge_index':    r['edge_index'],
            'bi_layer_index': r['bi_layer_index'],
            'y':             torch.tensor(label, dtype=torch.float)
        }

    def getAuthorFormat(self, data, row):
        circuit_data = self.getRawCircuitData(data, row)
        bi = circuit_data['bi_layer_index'].tolist()
        ei = circuit_data['edge_index'].tolist()
        x = circuit_data['x'].tolist()

        sourceNodes = ei[0]
        endNodes = ei[1]

        node_ids = bi[0][0]
        num_components = len(set(node_ids))
        numNodes = len(x)

        over = [num_components, numNodes]

        componentTypes = [i.index(1) for i in x]
        nodeSummaries = []
        for curNode in range(numNodes):
            directedEdges = []
            for i, nodeID in enumerate(endNodes):
                if (nodeID == curNode):
                    directedEdges.append(sourceNodes[i])
            directedEdges.append(numNodes - 1)

            nodeSummaries.append([componentTypes[curNode], curNode, len(directedEdges), directedEdges, -1, x[curNode][10], -1])


        return [over, nodeSummaries]


    def summarizeCircuit(self, data, row):
        nodeSummaries = self.getAuthorFormat(data, row)
        yvals = [self.df.iloc[row].values.tolist()]
        ylabs = self.df.columns.tolist()
        yDF = pd.DataFrame(yvals, columns=ylabs)

        print(f"Number of Subgraphs: {nodeSummaries[0][0]}")
        print(f"Number of Nodes: {nodeSummaries[0][1]}\n")
        for i, node in enumerate(nodeSummaries[1]):
            print(f"Node {i}:")
            print(f"  Component Type: {self.NODE_TYPE[node[0]]}")
            print(f"  Number of Directed Edges: {node[2]}")
            print(f"  Directed Edges: {node[3]}")
            print(f"  Node Value: {node[5]}\n")

        print(f"Specs")
        print(yDF)
        
        return ''
    