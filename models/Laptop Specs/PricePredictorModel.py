import torch.nn as nn

class LaptopPricePredictor(nn.Module):
    def __init__(self, inpSize):
        super().__init__()

        self.layer1 = nn.Linear(inpSize, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layerOut = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)

        return self.layerOut(x)
    
    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif (isinstance(layer, nn.Sequential)):
                for sub_layer in layer:
                    if hasattr(sub_layer, 'reset_parameters'):
                        sub_layer.reset_parameters()


