import torch
import torch.nn as nn

class Gram_Matrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size() #a = batch size, b = number of feature maps, (c,d) = dimensions of feature map
        features = input.view(a * b, c * d)
        G = torch.mm(features, torch.t(features))
        return G
