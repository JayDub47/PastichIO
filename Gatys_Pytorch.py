import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy
import sys

from PIL import Image
from torch.autograd import Variable

i = 0

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size() #a = batch size, b = number of feature maps, (c,d) = dimensions of feature map
        features = input.view(a * b, c * d)
        G = torch.mm(features, torch.t(features)) # Compute the gram product
        return G.div(a * b * c * d) # Normalize G by dividing by the number of feature elements


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0) #adds fake batch dimension to the image tensor
    return image

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

content_layer = ["conv_4"]
style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

def get_style_model_and_losses(cnn, style_image, content_image,
                                style_weight=1000, content_weight=1,
                                content_layer=content_layer, style_layers=style_layers):
    cnn = copy.deepcopy(cnn)

    #losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()
    gram = GramMatrix()

    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()
    
    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layer:
                target = model(content_image).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)
            
            if name in style_layers:
                target_feature = model(style_image).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
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

    return model, style_losses, content_losses

def get_input_param_optimizer(input_image):
    input_param = nn.Parameter(input_image.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(cnn, content_image, style_img, input_image, num_steps=50,
                        style_weight=1000, content_weight=1, regularization_weight=1000):
    print("building the style transfer model")
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_image, content_image, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_image)

    print("Beginning Optimization")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_param.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += (1 / len(style_losses)) * sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            
            regularization_score = regularization_weight * \
                (torch.sum(torch.abs(input_param.data[:, :, :, :-1] - input_param.data[:, :, :, 1:])) +\
                torch.sum(torch.abs(input_param.data[:, :, :-1, :]) - input_param.data[:, :, 1:, :]))

            run[0] += 1

            return style_score + content_score + regularization_score

        optimizer.step(closure)
    
    input_param.data.clamp_(0, 1)
    return input_param.data

content_layer = ["conv_4"]
style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
imsize = 512 if use_cuda else 128

unloader = transforms.ToPILImage()

def imshow(tensor):
    image = tensor.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    plt.imshow(image)
    plt.pause(0.001)
    return image


loader = transforms.Compose([
    transforms.Resize(imsize), #scale imported image to size
    transforms.ToTensor() #transforms image to a tensor
])

style_image = image_loader(sys.argv[1])
content_image = image_loader(sys.argv[2])

assert style_image.size() == content_image.size(), \
    "Style and Content images must be of the same size"

cnn = models.vgg19(pretrained=True).features
if use_cuda:
    cnn = cnn.cuda()

#input_image = content_image.clone()
input_image = Variable(torch.randn(content_image.size())).type(dtype)

output_image = run_style_transfer(cnn, content_image, style_image, input_image, num_steps=30)
save_image = imshow(output_image)
save_image.save("output.png")
plt.show()