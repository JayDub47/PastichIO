import torch
import torch.nn as nn

from Gram_Matrix import Gram_Matrix

class Style_Loss(nn.Module):

    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = Gram_Matrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G = self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
