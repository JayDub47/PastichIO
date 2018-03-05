import sys
import keras.backend as k
import numpy as np
import os
from keras.applications import vgg19
from keras.preprocessing import image
from keras.losses import mean_squared_error
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

class Gatys:

    def __init__(self, content_image_path, style_image_path, output_folder, iterations):
        self.content_weight = 0.00001
        self.style_weight = 1.0
        self.total_variation_weight = 1.0
        self.output_folder = output_folder
        self.iterations = int(iterations)
        self.content_layer = 'block4_conv2'
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.out_dim = 400
        self.content_img = self.load_image(content_image_path)
        self.style_img = self.load_image(style_image_path)

        #Placeholder values for the generated image
        self.generated_img = k.placeholder((1, self.out_dim, self.out_dim, 3))

        #Run tensor through model
        input_tensor = k.concatenate([self.content_img, self.style_img, self.generated_img], axis=0)
        model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')

        #Save outputs in dictionary with keys
        outputs = dict([(layer.name, layer.output) for layer in model.layers])

        #Extract content layer output and evaluate current content loss
        total_loss = k.variable(0.0)
        content_layer_output = outputs[self.content_layer]
        base_content_layer_output = content_layer_output[0, :, :, :]
        generated_content_layer_output = content_layer_output[2, :, :, :]
        total_loss = self.content_weight * self.calculate_content_loss(base_content_layer_output, generated_content_layer_output)

        #Extract style layers output and evaluate the style loss
        temp_style_loss = k.variable(0.0)
        style_layer_weight = 1 / len(self.style_layers)
        for layer in self.style_layers:
            layer_output = outputs[layer]
            style_image_output = layer_output[1, :, :, :]
            generated_image_output = layer_output[2, :, :, :]
            layer_style_loss = self.calculate_style_loss(style_image_output, generated_image_output)
            temp_style_loss += style_layer_weight * layer_style_loss

        style_loss = self.style_weight * temp_style_loss
        total_loss += style_loss

        total_loss += self.total_variation_weight * self.calculate_total_variation_loss(self.generated_img)

        #Compute Gradients
        gradients = k.gradients(total_loss, self.generated_img)

        outputs = [total_loss]

        if type(gradients) in {list, tuple}:
            outputs += gradients
        else:
            outputs.append(gradients)

        self.loss_gradients_function = k.function([self.generated_img], outputs)

    def load_image(self, image_path):
    #Load the images from the given path
        loaded_image = image.load_img(image_path, target_size=(self.out_dim, self.out_dim))
        loaded_image = image.img_to_array(loaded_image)
        loaded_image = np.expand_dims(loaded_image, axis=0)
        loaded_image = vgg19.preprocess_input(loaded_image)
        loaded_image = k.variable(loaded_image)
        return loaded_image

    def gram_matrix(self, input_matrix):
        #Calculate and return the gram matrix of the given input matrix
        matrix = k.batch_flatten(k.permute_dimensions(input_matrix, (2, 0, 1)))
        return k.dot(matrix, k.transpose(matrix))

    def calculate_content_loss(self, base_image_output, generated_image_output):
        #Returned squared loss between the two arrays
        return k.sum(k.square(generated_image_output - base_image_output))

    def calculate_style_loss(self, style_image_output, generated_image_output):
        #Return the weighted difference between the Gram Matrices of the two arrays

        #Get weight parameters
        layer_shape = style_image_output.get_shape()
        n = layer_shape.dims[-1].__int__()
        m = layer_shape.dims[1].__int__() * layer_shape.dims[2].__int__()
        style_gram = self.gram_matrix(style_image_output)
        generated_gram = self.gram_matrix(generated_image_output)
        factor = 1.0 / (4 * n**2 * m**2)
        return factor * (k.sum(k.square(generated_gram - style_gram)))

    def calculate_total_variation_loss(self, generated_image_output):
        #Calculates the Total Variation loss of the generated image
        #implementation adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb
        total_variation_loss = k.sum(k.abs(generated_image_output[:, 1:, :, :]) - k.abs(generated_image_output[:, :-1, :, :])) + \
               k.sum(k.abs(generated_image_output[:, :, 1:, :]) - k.abs(generated_image_output[:, :, :-1, :]))

        return total_variation_loss

    def loss(self, img):
        #Calculate Loss
        img = img.reshape((1, self.out_dim, self.out_dim, 3))
        out = self.loss_gradients_function([img])
        return out[0]

    def gradient(self, img):
        #Calculate Gradeints
        img = img.reshape((1, self.out_dim, self.out_dim, 3))
        out = self.loss_gradients_function([img])
        return np.array(out[1:]).flatten().astype('float64')

    def deprocess_image(self, img):
        iteration_result = img.reshape((self.out_dim, self.out_dim, 3))
        # Remove zero-center by mean pixel
        iteration_result[:, :, 0] += 103.939
        iteration_result[:, :, 1] += 116.779
        iteration_result[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        iteration_result = iteration_result[:, :, ::-1]
        iteration_result = np.clip(iteration_result, 0, 255).astype('uint8')
        return iteration_result

    def generate_styled_image(self):
        #create output folder
        os.makedirs(self.output_folder)

        #generate random image to serve as initial image
        output_image = np.random.uniform(0, 255, (1, self.out_dim, self.out_dim, 3))

        for i in range(0, self.iterations):
            print("Beginning Iteration" + str(i))
            output_image, y, z = fmin_l_bfgs_b(self.loss, output_image.flatten(), fprime=self.gradient, maxfun=20)
            iteration_result = self.deprocess_image(output_image.copy())
            imname = self.output_folder + "/" + "output_" + str(i) + ".jpg"
            imsave(imname, iteration_result)

if __name__ == "__main__":
    styler = Gatys(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    styler.generate_styled_image()