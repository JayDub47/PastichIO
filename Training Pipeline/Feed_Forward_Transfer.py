import sys

import torch
import torch.onnx
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from PIL import Image
from torch.autograd import Variable
from Image_Transform_Net import Image_Transform_Net
from Loss_Network import Loss_Network


class Feed_Forward_Transfer():

    def __init__(self, content_folder_path, style_image_path):
        self.content_folder_path = content_folder_path
        self.style_image_path = style_image_path
        self.transform_network = Image_Transform_Net()
        self.optimizer = optim.Adam(self.transform_network.parameters(), lr=0.001)
        self.content_layer = ["conv_4"]
        self.content_weight = 4500 
        self.content_loss_points = []
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
        self.style_weight = 1 
        self.style_loss_points = []
        self.tv_weight = 0.0001
        self.use_cuda = torch.cuda.is_available()
        self.im_size = 400 
        self.batch_size = 4
        self.epochs = 4 

        if self.use_cuda:
            self.transform_network = self.transform_network.cuda()

        self.loader = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor()
        ])

    def image_loader(self, im_path):
        image = Image.open(im_path)
        image = self.loader(image)
        return image

    def transform_image(self, im_path):
        image = self.image_loader(im_path).unsqueeze(0)
        if self.use_cuda:
            image = image.cuda()
        pastiche = self.transform_network(image)
        pastiche = pastiche.clone().cpu()
        pastiche = pastiche.view(3, self.im_size, self.im_size)
        unloader = transforms.ToPILImage()
        pastiche = unloader(pastiche.data)
        return pastiche

    def get_model(self):
        return self.transform_network

    def save_model_weights(self, filepath):
        torch.save(self.transform_network.state_dict(), filepath)

    def train(self):
        content = datasets.ImageFolder(root=self.content_folder_path, loader=self.image_loader)
        content_loader = torch.utils.data.DataLoader(content, batch_size=self.batch_size, shuffle=True)
        print("Training start:")

        style_image = self.image_loader(self.style_image_path).unsqueeze(0)
        style_image = Variable(torch.cat((style_image, style_image, style_image, style_image), 0))
        if self.use_cuda:
            style_image = style_image.cuda()

        loss_network = Loss_Network(style_image, self.content_weight, self.style_weight,
                                    self.content_layer, self.style_layers, self.use_cuda)
        
        for epoch in range(self.epochs):
            print("Beginning Epoch " + str(epoch))
            for i, content_batch in enumerate(content_loader):
                input = content_batch[0]
                input = Variable(input)

                if self.use_cuda:
                    input = input.cuda()

                pastiches = self.transform_network.forward(input)

                model, content_losses, style_losses = loss_network.get_model_and_losses(input)
                
                def closure():                        
                    self.optimizer.zero_grad()
                    content_loss = 0
                    style_loss = 0
                    tv_loss = 0
                    model(pastiches)

                    for sl in style_losses:
                        style_loss += (1 / len(self.style_layers)) * sl.backward()
                    for cl in content_losses:
                        content_loss += cl.backward()
                    
                    tv_loss = self.tv_weight * \
                                    (torch.sum(torch.abs(pastiches[:, :, :, :-1] - pastiches[:, :, :, 1:])) +\
                                    torch.sum(torch.abs(pastiches[:, :, :-1, :]) - pastiches[:, :, 1:, :]))
                    

                    total_loss = style_loss + content_loss + tv_loss

                    if i % 50 == 0:
                        print(str(i) + "th batch iteration")
                        print("content loss: " + str(content_loss.data[0]))
                        print("style loss: " + str(style_loss.data[0]))

                    self.content_loss_points.append((i, content_loss.data[0]))
                    self.style_loss_points.append((i, style_loss.data[0]))

                    return total_loss

                self.optimizer.step(closure)
        
        with open('content_loss.pickle', 'wb') as f:
            pickle.dump(self.content_loss_points, f)

        with open('style_loss.pickle', 'wb') as f:
            pickle.dump(self.style_loss_points, f)


if __name__ == "__main__":
    network = Feed_Forward_Transfer(sys.argv[1], sys.argv[2])
    network.train()
    model = network.get_model()
    network.save_model_weights("weights.pt")
    model = model.eval()
    test_pastiche = network.transform_image(sys.argv[3])
    test_pastiche.save("Output.png")
    dummy_input = Variable(torch.randn(1, 3, 512, 512)).cuda()
    torch.onnx.export(model, dummy_input, "Trained_Network.proto")