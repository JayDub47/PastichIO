import torch
import torch.nn as nn

class Content_Loss(nn.Module):

    def __init__(self, target, weight):
        '''The class is initialised with the target it will eventually calculate
           the loss from'''
        super(Content_Loss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        #The forward does not affect its input but uses it to calculate the loss
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
