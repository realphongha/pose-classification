import torch
import torch.nn as nn


class FCNet2(nn.Module):
    def __init__(self, channels=3, joints=13, num_cls=4):
        super(FCNet2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels*joints, channels*joints)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(channels*joints, channels*joints*2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(channels*joints*2, num_cls)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    
if __name__ == "__main__":
    import numpy as np
    
    sample_input = torch.tensor(np.zeros((16, 13, 3))).float()
    model = FCNet2()
    print(model)
    print(model(sample_input))
    