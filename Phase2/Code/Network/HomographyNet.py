'''
Implementation of homography network as formulated in https://arxiv.org/abs/1606.03798

Author: Aneesh Dandime
Email: aneeshd@umd.edu
'''
import torch.nn as nn
from torchsummary import summary


class HomographyNet(nn.Module):
    def __init__(self) -> None:
        super(HomographyNet, self).__init__()
        self.layer1 = self.__make_layer(2, 64, pool=False)
        self.layer2 = self.__make_layer(64, 64)
        self.layer3 = self.__make_layer(64, 64, pool=False)
        self.layer4 = self.__make_layer(64, 64)
        self.layer5 = self.__make_layer(64, 128, pool=False)
        self.layer6 = self.__make_layer(128, 128)
        self.layer7 = self.__make_layer(128, 128, pool=False)
        self.layer8 = self.__make_layer(128, 128, pool=False)
        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, 8)
    
    def __make_layer(self, in_channels, out_channels, kernel_size=3, pool=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(-1, 32768)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = HomographyNet()
    summary(model, (2, 128, 128))

