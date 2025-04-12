from typing import List

import torch
from torch import nn
# from d2l import torch as d2l


class LetNet(nn.Module):
    def __init__(self):
        super(LetNet, self).__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
        for l in self.net:
            X = l(X)
            print((l.__class__.__name__, 'output shape: ', X.shape))


if __name__ == '__main__':
    net = LetNet()
    net.forward(torch.randn(1, 1, 28, 28))
