import copy
import torch
import torch.nn as nn
import torchvision.models as models
from Content_Loss import Content_Loss
from Style_Loss import Style_Loss
from Gram_Matrix import Gram_Matrix

class Loss_Network():

    def __init__(self, style_image, content_weight, style_weight, content_layer, style_layers, use_cuda):
        self.style_image = style_image
        self.content_layer = content_layer
        self.content_weight = content_weight
        self.style_layers = style_layers
        self.style_weight = style_weight
        self.use_cuda = use_cuda
        self.network = models.vgg19(pretrained=True).features
        
        if use_cuda:
            self.network = self.network.cuda()

    def get_model_and_losses(self,content_image):
        network = copy.deepcopy(self.network)

        content_losses = []
        style_losses = []

        model = nn.Sequential()
        gram = Gram_Matrix()

        if self.use_cuda:
            model = model.cuda()
            gram = gram.cuda()

        i = 1
        for layer in list(network):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                model.add_module(name, layer)

                if name in self.content_layer:
                    target = model(content_image).clone()
                    content_loss = Content_Loss(target, self.content_weight)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)
                
                if name in self.style_layers:
                    target_feature = model(self.style_image).clone()
                    target_feature_gram = gram(target_feature)
                    style_loss = Style_Loss(target_feature_gram, self.style_weight)
                    model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                model.add_module(name, layer)
                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride,
                                        padding=layer.padding)
                model.add_module(name, avgpool)
        
        return model, content_losses, style_losses