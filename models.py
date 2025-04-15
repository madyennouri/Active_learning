import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x = self.fc3(x2)
        return x, [x1, x2]


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[64, 128], interm_dim=256):
        super(LossNet, self).__init__()
        self.FC1 = nn.Linear(feature_sizes[0], interm_dim)
        self.FC2 = nn.Linear(feature_sizes[1], interm_dim)
        self.linear = nn.Linear(2 * interm_dim, 1)

    def forward(self, features):
        out1 = torch.relu(self.FC1(features[0]))
        out2 = torch.relu(self.FC2(features[1]))
        out_f = self.linear(torch.cat((out1, out2), 1))
        return out_f