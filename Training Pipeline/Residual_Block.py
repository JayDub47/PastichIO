import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.padding1 = nn.ReflectionPad2d(1)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.padding2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.batch_norm2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.padding1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.padding2(x)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        #residual = F.pad(residual, (-2, -2, -2, -2)) #Match x size to out size after convolutions
        out = out + residual

        return out
