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

parser = argparse.ArgumentParser(description="Train a practice GCN model on Enzyme data")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--reps', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.0)
parser.add_argument('--graph', type=bool, default=False)

args = parser.parse_args()

autoStop = [args.threshold]

# appends statistical properties of each node before processing
pre_transform = T.LocalDegreeProfile()

cktPath = "C:/Users/Kevin Nesbitt/Documents/Coding/Python/KIL/CktGNN Clone/CktGNN/OCB/CktBench101"

df = pd.read_csv(cktPath + "/perform101.csv")
rawPerfMatrix = df[['gain', 'bw', 'pm']].values
perfMeans = rawPerfMatrix.mean(axis=0)
perfSTD = rawPerfMatrix.std(axis=0)

normalizedPerfs = (rawPerfMatrix - perfMeans) / perfSTD

convertedData = []
subGParser = cph(cktPath)
rawData = subGParser.getRawData()
parsedDataset = [subGParser.getAuthorFormat(dataSplit=rawData, circuitID=i) for i in range(len(rawData))]
# convertedData = [subGParser.toDataObject(authorFormat=circuit, performanceTargets=circuit[2]) for circuit in parsedDataset]

for i, circuit in enumerate(parsedDataset):
    convertedData.append(subGParser.toDataObject(authorFormat=circuit, performanceTargets=normalizedPerfs[i]))

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
        pred = model(i.x, i.edge_index, i.batch)

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
            pred = model(i.x, i.edge_index, i.batch)
            loss = lossFn(pred, i.y)
            totalLoss += loss.item() * i.num_graphs
    
    return totalLoss / len(loader.dataset)

def shuffleDataset():
    # shuffle data
    # torch.manual_seed(42)
    random.shuffle(convertedData)
    trainSplitPercent = int(len(convertedData)*0.85)
    

    # datasplit
    trainSet = convertedData[:trainSplitPercent]
    testSet = convertedData[trainSplitPercent:]

    # combines multiple graphs into one batch
    train_loader = DataLoader(trainSet, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(testSet, batch_size=args.batch, shuffle=False)

    return train_loader, test_loader

def main():
    results = []
    print(f"--epochs {args.epochs}\n--lr {args.lr}\n--batch {args.batch}\n--reps {args.reps}\n--threshold {autoStop[0]}\n\n")

    for i in range(args.reps):
        model = CircuitModel(convertedData[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        epochLoss = []
        epochLst = []
        testLossTmp = []
        consec = 0
        consecInc = 0
        trainLoader, testLoader = shuffleDataset()

        # if (i > 0):
        #     model.reset_weights() 

        print(f"\nRep: {i + 1}")
        for epoch in range(args.epochs):
            trainLoss = train(trainLoader, epoch, model, optimizer)
            testLoss = test(testLoader, model)

            # cuts lr by half if model does not improve after 10 epochs
            scheduler.step(testLoss)

            if (args.epochs > 10) or ((epoch + 1) == 1):
                if ((epoch + 1) % 5 == 0) or ((epoch + 1) == 1):
                    print(f"Epoch {epoch + 1}/{args.epochs}: Train Loss {trainLoss:.4f}, Test Loss {testLoss:.4f}")
            if ((epoch + 1) == args.epochs):
                results.append([trainLoss, testLoss])

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
                    results.append([trainLoss, testLoss])
                    break

        if (args.graph):
            plt.figure(figsize=(8, 6))
            plt.title("Test Loss", fontsize = 18)
            plt.ylabel("Loss", fontsize=18)
            plt.xlabel("Epoch", fontsize=18)
            
            sns.regplot(x = epochLst, y = epochLoss, order=2)
            plt.savefig(f"Test Loss Rep {i}", dpi=300)
            # plt.show()

    
    print('\n')
    print('='*50)
    print("RESULTS")
    print('='*50)
    for i, lst in enumerate(results):
        print(f"Rep {i + 1}:\n    Final Train Loss: {lst[0]}\n    Final Test Loss: {lst[1]}")

    print('\n')
    print(f"Avg Train Loss: {sum([i[0] for i in results])/len(results)}\nAvg Test Loss: {sum([i[1] for i in results])/len(results)}")
    print('='*50)


if (__name__ == "__main__"):
    main()