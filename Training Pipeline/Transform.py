import sys

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from Image_Transform_Net import Image_Transform_Net
from Loss_Network import Loss_Network

transform_network = Image_Transform_Net().cuda()
transform_network.load_state_dict(torch.load(sys.argv[1]))
transform_network.eval()

loader = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

image = Image.open(sys.argv[2])
image = loader(image).unsqueeze(0).cuda()
pastiche = transform_network(image)
print(pastiche.shape)
pastiche = pastiche.clone().cpu()
pastiche = pastiche.view(3, 512, 512)
unloader = transforms.ToPILImage()
pastiche = unloader(pastiche.data)
pastiche.save("output_evaluate.jpg")
dummy_input = Variable(torch.randn(1, 3, 512, 512)).cuda()
torch.onnx.export(transform_network, dummy_input, "Trained_Network.proto")
