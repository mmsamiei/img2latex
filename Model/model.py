from torch import nn
import torch

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3)

    def forward(self, x):
        temp = x
        temp = self.conv1(temp)
        temp = self.conv2(temp)
        temp = self.pool1(temp)
        temp = self.conv3(temp)
        temp = self.conv4(temp)
        temp = self.pool2(temp)
        return temp

model = CNNEncoder()
test = torch.rand((200,30))
test = test.unsqueeze(0)
test = test.unsqueeze(1)
print(model(test).shape)
