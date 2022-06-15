import torch
import torch.nn as nn


class KinematicPredictor(torch.nn.Module):
    def __init__(self):
        super(KinematicPredictor, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(3, 50), nn.LeakyReLU(), nn.Linear(50, 100), nn.LeakyReLU())
        self.norm = nn.BatchNorm1d(100)
        self.layer2 = nn.Sequential(nn.Linear(100, 50), nn.LeakyReLU(), nn.Linear(50, 20), nn.LeakyReLU())
        self.fc = nn.Linear(20, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x


class ShapePredictor(torch.nn.Module):
    def __init__(self):
        super(ShapePredictor, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(3, 10), nn.LeakyReLU(), nn.Linear(10, 25), nn.Linear(25, 40), nn.LeakyReLU(), nn.Linear(40, 50), nn.LeakyReLU())
        self.norm = nn.BatchNorm1d(50)
        self.layer2 = nn.Sequential(nn.Linear(50, 20), nn.LeakyReLU(), nn.Linear(20, 20), nn.LeakyReLU())
        self.fc = nn.Linear(20, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x
