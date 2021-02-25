from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import resnet
import numpy as np

class prob_res50(nn.Module):
    def __init__(self, is_prob=False):
        super(prob_res50, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(2048, 240)
        self.is_prob=is_prob

    def forward(self, x):
        prob = self.pretrained_model(x)
        return prob if self.is_prob else self.get_predict_age(prob)

    def get_predict_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb,dim=1).cuda()
        age = torch.linspace(0,239,240).cuda()
        predict_age = torch.mv(predict_age_pb,age)
        return predict_age


class my_net_res50(nn.Module):
    def __init__(self):
        super(my_net_res50, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        out = self.pretrained_model(x)
        return out
