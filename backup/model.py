from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tools import resnet
import numpy as np

class my_net_res50(nn.Module):
    def __init__(self):
        super(my_net_res50, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        out = self.pretrained_model(x)
        return out
