from __future__ import print_function
import torch
import torch.nn as nn
import os

class LeNet_300_100(nn.Module):
    def __init__(self, bias_flag, cfg):
        if cfg == None:
            cfg = [300,100]
        super(LeNet_300_100, self).__init__()
        self.ip1 = nn.Linear(28*28, cfg[0], bias=bias_flag)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = nn.Linear(cfg[0], cfg[1], bias=bias_flag)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = nn.Linear(cfg[1], 10, bias=bias_flag)
        return

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        return x