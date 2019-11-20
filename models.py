from torch import nn
from torchsummary import summary
from torchvision import models


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


if __name__ == "__main__":
    from config import device

    model = RankNetMobile().to(device)
    summary(model, input_size=[(3, 224, 224), (3, 224, 224)])
