import argparse
from torch_geometric.datasets import TUDataset
import torch.nn.functional
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm
from Circuitmodel import CircuitModel
import pandas as pd
from CktGNNSubGParser import CktGNNSubGParser as cph
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
from VarModels import GATwMLPHead, GCNwMLPHead, SAGEwMLPHead
import os

parser = argparse.ArgumentParser(description="Train a practice GCN model on Enzyme data")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--reps', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.0)
parser.add_argument('--graph', type=bool, default=False)

args = parser.parse_args()

autoStop = [args.threshold]

# model = CircuitModel(convertedData[0])
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lossFn = torch.nn.MSELoss()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# training loop
def train(loader, curEpoch, model, optimizer):
    # switch model to training mode
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Training {curEpoch + 1}/{args.epochs}", leave=False)
    
    for i in pbar:
        # clears previous gradient
        optimizer.zero_grad()

        # runs a forward pass through the model
        pred = model(i.x, i.batch, i.edge_index)

        # performs MSE loss
        loss = lossFn(pred, i.y)

        # computes gradients of the loss with respect to each weight
        loss.backward()

        # updates weights
        optimizer.step()

        total_loss += loss.item() * i.num_graphs

        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
    
    return total_loss / len(loader.dataset)

# test loop
def test(loader, model):
    # switch model to evaluation mode
    model.eval()
    totalLoss = 0

    with torch.no_grad():
        for i in loader:
            pred = model(i.x, i.batch, i.edge_index)
            loss = lossFn(pred, i.y)
            totalLoss += loss.item() * i.num_graphs
    
    return totalLoss / len(loader.dataset)

def shuffleDataset(data):
    # shuffle data
    # torch.manual_seed(42)
    random.shuffle(data)
    trainSplitPercent = int(len(data)*0.85)
    

    # datasplit
    trainSet = data[:trainSplitPercent]
    testSet = data[trainSplitPercent:]

    # combines multiple graphs into one batch
    train_loader = DataLoader(trainSet, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(testSet, batch_size=args.batch, shuffle=False)

    return train_loader, test_loader

def initGAT(data):
    model = GATwMLPHead(data[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    return model, optimizer, scheduler

def initGCN(data):
    model = GCNwMLPHead(data[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    return model, optimizer, scheduler

def initSAGE(data):
    model = SAGEwMLPHead(data[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    return model, optimizer, scheduler

def main():
    print(f"--epochs {args.epochs}\n--lr {args.lr}\n--batch {args.batch}\n--reps {args.reps}\n--threshold {autoStop[0]}\n\n")

    with open("config.json") as f:
        config = json.load(f)

    cktPath = config["data_path"]

    df = pd.read_csv(cktPath + "/perform101.csv")

    specs = ['gain', 'bw', 'pm', 'fom']
    modelTypes = ['GCN', 'GAT', 'SAGE']
    summary = {
        'gain':[], 
        'bw':[], 
        'pm':[],
        'fom':[]
    }

    for modelName in modelTypes:
        folderName = "graphs"
        modelPath = os.path.join(folderName, modelName)

        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
            print(f"File Created at {modelPath}\n\n")
        else:
            print(f"{modelPath} already exists")

    for spec in specs:
        print('\n\n')

        print('='*50)
        print(f"Spec: {spec.upper()}")
        print('='*50)

        rawPerfMatrix = df[[spec]].values
        perfMeans = rawPerfMatrix.mean(axis=0)
        perfSTD = rawPerfMatrix.std(axis=0)

        normalizedPerfs = (rawPerfMatrix - perfMeans) / perfSTD

        subGParser = cph(cktPath)
        rawData = subGParser.getRawData()
        parsedDataset = [subGParser.getAuthorFormat(dataSplit=rawData, circuitID=i) for i in range(len(rawData))]
        # convertedData = [subGParser.toDataObject(authorFormat=circuit, performanceTargets=circuit[2]) for circuit in parsedDataset]

        # appends statistical properties of each node before processing
        pre_transform = T.LocalDegreeProfile()

        convertedData = []
        for i, circuit in enumerate(parsedDataset):
            convertedData.append(subGParser.toDataObject(authorFormat=circuit, performanceTargets=normalizedPerfs[i]))
        

        results = {'GCN':[], 'GAT':[], 'SAGE':[]}

        for rep in range(args.reps):
            print(f"\nRep: {rep + 1}")
            print('_'*50)

            models = [GCNwMLPHead(convertedData[0]), GATwMLPHead(convertedData[0]), SAGEwMLPHead(convertedData[0])]

            for i, model in enumerate(models):
                print(f"\nModel: {modelTypes[i]}")

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

                epochLoss = []
                epochLst = []
                testLossTmp = []
                consec = 0
                consecInc = 0
                trainLoader, testLoader = shuffleDataset(convertedData)

                # if (i > 0):
                #     model.reset_weights() 

                for epoch in range(args.epochs):
                    trainLoss = train(trainLoader, epoch, model, optimizer)
                    testLoss = test(testLoader, model)

                    # cuts lr by half if model does not improve after 10 epochs
                    scheduler.step(testLoss)

                    if (args.epochs > 10) or ((epoch + 1) == 1):
                        if ((epoch + 1) % 5 == 0) or ((epoch + 1) == 1):
                            print(f"Epoch {epoch + 1}/{args.epochs}: Train Loss {trainLoss:.4f}, Test Loss {testLoss:.4f}")
                    if ((epoch + 1) == args.epochs):
                        print(f"Epoch {epoch + 1}/{args.epochs}: Train Loss {trainLoss:.4f}, Test Loss {testLoss:.4f}")
                        results[modelTypes[i]].append(testLoss)

                    epochLoss.append(testLoss)
                    epochLst.append(epoch + 1)

                    # break after change is minimal 3 consecutive times
                    if (autoStop[0] > 0.0):
                        if (len(testLossTmp) <= 1):
                            testLossTmp.append(testLoss)
                        else:
                            if (testLossTmp[1] > testLossTmp[0]):
                                consecInc +=1
                            else:
                                consecInc = 0
                            
                            change = abs(testLossTmp[1] - testLossTmp[0])
                            testLossTmp = []
                            if (change <= 0.001):
                                consec += 1
                            else:
                                consec = 0
                        if (consec >= 5 or consecInc >= 7):
                            print(f"Epoch {epoch + 1}/{args.epochs}: Train Loss {trainLoss:.4f}, Test Loss {testLoss:.4f}")
                            print(f"Autostop applied due to {'consecutive loss increase' if (consecInc >= 7) else 'threshold'}")
                            print(f"Lowest Test Loss: {min(epochLoss)}")
                            results[modelTypes[i]].append(testLoss)
                            break
                
                if (args.graph):
                    fileName = f"{modelTypes[i]} Test Loss Rep {rep + 1}.png"

                    filePath = os.path.join(folderName, modelTypes[i], fileName)

                    plt.figure(figsize=(8, 6))
                    plt.title(f"{modelTypes[i]} Test Loss Rep {rep + 1}", fontsize = 18)
                    plt.ylabel("Loss", fontsize=18)
                    plt.xlabel("Epoch", fontsize=18)
                    
                    sns.regplot(x = epochLst, y = epochLoss, order=2 if (args.epochs >= 10) else 1)
                    plt.savefig(filePath, dpi=300)
                    # plt.show()

        for i in range(len(results)):
            summary[spec].append(sum(results[modelTypes[i]]) / len(results[modelTypes[i]]))

    # for spec in summary.keys():
    #     print('\n' + spec.upper())
    #     for model in summary[spec].keys():
    #         print(f"{model} Average Test Loss: {summary[spec][model]}")
    
    print('\n\n')
    print('='*50)
    print("RESULTS")
    print('='*50)

    dataTable = pd.DataFrame(data=summary, index=modelTypes)
    print(dataTable)

    print('='*50)

    if (args.graph):
        path = os.path.join(folderName, "Model Comparison.png")

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        sns.barplot(x=dataTable['gain'].index, y=dataTable['gain'].values, ax=axes[0][0], palette='viridis', hue=dataTable['gain'].index, legend=False)
        axes[0][0].set_title("Gain", fontsize=14)
        g_min = dataTable['gain'].values.min() - 0.05
        g_max = dataTable['gain'].values.max() + 0.05
        axes[0][0].set_ylim(g_min, g_max)
        axes[0][0].set_xlabel("Architecture", fontsize=10)
        axes[0][0].set_ylabel("Loss", fontsize=10)

        sns.barplot(x=dataTable['bw'].index, y=dataTable['bw'].values, ax=axes[0][1], palette='viridis', hue=dataTable['bw'].index, legend=False)
        axes[0][1].set_title("BW", fontsize=14)
        b_min = dataTable['bw'].values.min() - 0.05
        b_max = dataTable['bw'].values.max() + 0.05
        axes[0][1].set_ylim(b_min, b_max)
        axes[0][1].set_xlabel("Architecture", fontsize=10)
        axes[0][1].set_ylabel("Loss", fontsize=10)

        sns.barplot(x=dataTable['pm'].index, y=dataTable['pm'].values, ax=axes[1][0], palette='viridis', hue=dataTable['pm'].index, legend=False)
        axes[1][0].set_title("PM", fontsize=14)
        p_min = dataTable['pm'].values.min() - 0.05
        p_max = dataTable['pm'].values.max() + 0.05
        axes[1][0].set_ylim(p_min, p_max)
        axes[1][0].set_xlabel("Architecture", fontsize=10)
        axes[1][0].set_ylabel("Loss", fontsize=10)

        sns.barplot(x=dataTable['fom'].index, y=dataTable['fom'].values, ax=axes[1][1], palette='viridis', hue=dataTable['fom'].index, legend=False)
        axes[1][1].set_title("FoM", fontsize=14)
        f_min = dataTable['fom'].values.min() - 0.05
        f_max = dataTable['fom'].values.max() + 0.05
        axes[1][1].set_ylim(f_min, f_max)
        axes[1][1].set_xlabel("Architecture", fontsize=10)
        axes[1][1].set_ylabel("Loss", fontsize=10)

        plt.tight_layout()
        
        plt.savefig(path, dpi=300)
        plt.close()
        print("\n\nSuccess")


if (__name__ == "__main__"):
    main()
