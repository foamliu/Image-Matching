import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torchscope import scope
from torchvision import models

from config import num_classes


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MatchMobile(nn.Module):
    def __init__(self):
        super(MatchMobile, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Remove linear layer
        modules = list(mobilenet.children())[:-1]
        self.model = nn.Sequential(*modules,
                                   DepthwiseSeparableConv(1280, 1280, kernel_size=7, padding=0),
                                   Flatten(),
                                   nn.Linear(1280, 512),
                                   )
        self.output = nn.Sigmoid()

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        prob = self.output(s1 - s2)
        return prob

    def predict(self, input):
        s = self.model(input)
        return self.output(s)


class ArcMarginModel(nn.Module):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


if __name__ == "__main__":
    from config import device

    model = MatchMobile().to(device)
    scope(model, input_size=[(3, 224, 224)])
