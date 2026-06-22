import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch as t
import math

class CSVDataLoader():
    def __init__(self, path, batchSize):
        self.df = pd.read_csv(path)
        self.df["Price"] = self.df["Price"] * 0.011
        self.batchSize = batchSize
        self.ySTD = 0

    def toDataLoaderObject(self):
        # drop irrelevant columns
        self.df = self.df.drop(['Weight'], axis=1)

        # normalize features
        numericCols = self.df.select_dtypes(include="number").columns.tolist()
        numericCols.remove("Price")
        self.df[numericCols] = (self.df[numericCols] - self.df[numericCols].mean()) / self.df[numericCols].std()

        # one hot encoding of all colimns
        self.df = pd.get_dummies(self.df, drop_first=True)
        self.df = self.df.astype(float)
        
        # [num_samples, num_features]
        x = t.tensor(self.df.drop(['Price'], axis=1).values, dtype=t.float)

        # normalize target
        y = t.tensor(self.df['Price'].values, dtype=t.float) # [num_samples]
        yMean = y.mean()
        yStd = y.std()
        self.setySTDnorm(yStd)
        y = (y - yMean) / yStd

        dataset = TensorDataset(x, y)

        trainSize = math.floor(len(dataset) * 0.85)
        testSize = len(dataset) - trainSize

        trainSet, testSet = random_split(dataset, [trainSize, testSize])

        trainLoader = DataLoader(trainSet, batch_size=self.batchSize, shuffle=True)
        testLoader = DataLoader(testSet, batch_size=self.batchSize, shuffle=True)
        
        return [trainLoader, testLoader, x.shape[1]]
    
    def setySTDnorm(self, yStd):
        self.ySTD = yStd
    
    def UnNormalize(self, loss):
        rmseNorm = math.sqrt(loss)
        rmseRev = rmseNorm * self.ySTD

        return rmseRev

