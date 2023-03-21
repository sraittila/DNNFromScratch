import tensorflow as tf
import numpy as np
from NeuralNetwork import NeuralNetwork
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_images = 60000
num_pixels = 28*28
#transforms picture matrix into a vector
x_train_flat = np.zeros((num_images, num_pixels))
for i in range(num_images):
    x_train_flat[i,:] = x_train[i,:,:].flatten()

#initializes neural network, first layer with sigmoid
#and output layer with linear activation
nn = NeuralNetwork([784,32,10],["sigmoid","linear"])


#Setting weights from files
#These weights are trained with Tensor Flow
#Purpose is to use them directly to see if 
#implementation of forward propagation works
nn.setWeightsFromFile("layer1_weights.txt",1)
nn.setBiasFromFile("layer1_biases.txt",1)
nn.setWeightsFromFile("layer2_weights.txt",2)
nn.setBiasFromFile("layer2_biases.txt",2)

#checks how many numbers neural network guesses correct
#Expected 90.98%
i = 0
correct = 0
for x in x_train_flat:
    #x is a row vector and needs to be transposed
    xtransp = np.transpose(np.array([x]))
    #output is 10x1 vector
    outputlist = nn.forwardPropagate(xtransp)
    #print(outputlist)
    out = outputlist[-1][1]
    #argmax returns the index which has the highest value
    value = np.argmax(out)
    if value == y_train[i]:
        correct += 1
    i += 1


correctProcent = correct/60000.0*100
print()
print("The neural network got " + str(round(correctProcent,2)) + "% correct.")

