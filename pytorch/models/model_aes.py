from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv1d(8, 32, 1),
                nn.BatchNorm1d(32),
                nn.ReLU()
                )
        
        self.block2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(32, 32, 3, padding='same'),
                nn.BatchNorm1d(32),
                nn.ReLU(),

                nn.Conv1d(32, 32, 3, padding='same'),
                nn.BatchNorm1d(32),
                nn.ReLU(),

                ) for _ in range(10)])

        self.block3 = nn.Sequential(
                nn.Flatten(),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Linear(64, 1),
                nn.Sigmoid()
                )
    
    def forward(self, x):

        # Preparing input
        x = torch.reshape(x, (-1,8,4))
        # x = x.permute((0,2,1))

        # Applying all layers
        x = self.block1(x)

        for block in self.block2:
            x = block(x) + x

        x = self.block3(x)


        return x.reshape((-1,))	

