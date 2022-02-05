import torch
import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self, channels=3, joints=13, num_cls=4):
        super(FCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels*joints, channels*joints*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels*joints*2, num_cls)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    
if __name__ == "__main__":
    import numpy as np
    
    sample_input = torch.tensor(np.zeros((16, 13, 3))).float()
    model = FCNet()
    print(model)
    print(model(sample_input))
    