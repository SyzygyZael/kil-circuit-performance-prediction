import argparse
import torch.nn.functional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from CSVDataLoader import CSVDataLoader as cldr
from PricePredictorModel import LaptopPricePredictor as lpr

parser = argparse.ArgumentParser(description="Train a model on laptop specs and their prices.")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--reps', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.0)
parser.add_argument('--graph', type=bool, default=False)

args = parser.parse_args()

autoStop = [args.threshold]

lossFn = torch.nn.MSELoss()

def train(loader, curEpoch, model, optimizer):
    model.train()
    totalLoss = 0

    pbar = tqdm(loader, desc=f"Training {curEpoch + 1}/{args.epochs}", leave=False)

    for xBatch, yBatch in pbar:
        optimizer.zero_grad()
        pred = model(xBatch)
        loss = lossFn(pred, yBatch.view(-1, 1))
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    return totalLoss / len(loader.dataset)
    
def test(loader, model):
    model.eval()
    totalLoss = 0

    with torch.no_grad():
        for xBatch, yBatch in loader:
            pred = model(xBatch)
            loss = lossFn(pred, yBatch.view(-1, 1))

            totalLoss += loss.item()
        
    return totalLoss / len(loader.dataset)
    
def main():
    results = []
    print(f"--epochs {args.epochs}\n--lr {args.lr}\n--batch {args.batch}\n--reps {args.reps}\n--threshold {autoStop[0]}\n\n")

    for i in range(args.reps):
        csvLoader = cldr("C:/Users/Kevin Nesbitt/Documents/Coding/Python/Deep Learning/Laptop Specs/data/laptop_data.csv", args.batch)
        daddy = csvLoader.toDataLoaderObject()

        model = lpr(daddy[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        epochLoss = []
        epochLst = []
        testLossTmp = []
        consec = 0
        consecInc = 0

        trainLoader = daddy[0]
        testLoader = daddy[1]

        # if (i > 0):
        #     model.reset_weights() 

        print(f"\nRep: {i + 1}")
        for epoch in range(args.epochs):
            trainLoss = train(trainLoader, epoch, model, optimizer)
            testLoss = test(testLoader, model)

            # cuts lr by half if model does not improve after 10 epochs
            scheduler.step()

            if (args.epochs > 10) or ((epoch + 1) == 1):
                if ((epoch + 1) % 10 == 0) or ((epoch + 1) == 1):
                    print(f"Epoch {epoch + 1}/{args.epochs}: Train Loss {trainLoss:.4f}, Test Loss {testLoss:.4f}")
            if ((epoch + 1) == args.epochs):
                results.append([trainLoss, testLoss])
                print(f"Lowest Test Loss: {min(epochLoss)}")

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
    print(f"Avg Test Loss in Dollars: ${csvLoader.UnNormalize(loss=sum([i[1] for i in results])/len(results)):.2f}")
    print('='*50)

if (__name__ == "__main__"):
    main()