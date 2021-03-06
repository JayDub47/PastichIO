import torch
import torch.nn as nn
import torch.nn.functional as F
from Residual_Block import Residual_Block

class Image_Transform_Net(nn.Module):
    '''This class is the Image Transform Network as set out by Johnson et al,
       It follow their archtiecture almost exactly with a few small changes.'''


    def __init__(self):
        super(Image_Transform_Net, self).__init__()
        self.padding2 = nn.ReflectionPad2d(4)
        self.conv1 = nn.Conv2d(3, 32, 9, stride=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.padding3 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.batch2 = nn.BatchNorm2d(64)
        self.padding4 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.batch3 = nn.BatchNorm2d(128)
        self.residual1 = Residual_Block(128, 128, 3)
        self.residual2 = Residual_Block(128, 128, 3)
        self.residual3 = Residual_Block(128, 128, 3)
        self.residual4 = Residual_Block(128, 128, 3)
        self.residual5 = Residual_Block(128, 128, 3)

        '''Deconvolution is implemented in two steps, an upsample followed by
           a standard convolution. This is done because the standard
           ConvTranspose aproach leads to checkerboarding artifacts.'''
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.padding5 = nn.ReflectionPad2d(1)
        self.deconv1 = nn.Conv2d(128, 64, 3, stride=1)
        self.batch4 = nn.BatchNorm2d(64)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.padding6 = nn.ReflectionPad2d(1)
        self.deconv2 = nn.Conv2d(64, 32, 3, stride=1)
        self.batch5 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 3, 9, stride=1)
        self.padding7 = nn.ReflectionPad2d(4)
        self.batch6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.padding2(x)
        out = self.conv1(out)
        out = self.batch1(out)
        out = self.relu(out)
        out = self.padding3(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu(out)
        out = self.padding4(out)
        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu(out)
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.residual3(out)
        out = self.residual4(out)
        out = self.residual5(out)
        out = self.upsample1(out)
        out = self.padding5(out)
        out = self.deconv1(out)
        out = self.batch4(out)
        out = self.relu(out)
        out = self.upsample2(out)
        out = self.padding6(out)
        out = self.deconv2(out)
        out = self.batch5(out)
        out = self.relu(out)
        out = self.padding7(out)
        out = self.conv4(out)
        out = self.batch6(out)
        out = self.relu(out)

        return out
