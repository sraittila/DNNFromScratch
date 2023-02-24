import tensorflow as tf
import numpy as np
from NeuralNetwork import NeuralNetwork
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_images = 60000
num_pixels = 28*28

x_train_flat = np.zeros((num_images, num_pixels))
for i in range(num_images):
    x_train_flat[i,:] = x_train[i,:,:].flatten()


nn = NeuralNetwork([784,32,10],["sigmoid","linear"])
#print(nn.network[1][1])

print("matrix")
nn.setWeightsFromFile("layer1_weights.txt",1)
nn.setBiasFromFile("layer1_biases.txt",1)
nn.setWeightsFromFile("layer2_weights.txt",2)
nn.setBiasFromFile("layer2_biases.txt",2)
i = 0
correct = 0
for x in x_train_flat:
    xtransp = np.transpose(np.array([x]))
    out = nn.forwardPropagate(xtransp)
    value = np.argmax(out)
    if value == y_train[i]:
        correct += 1
    i += 1

print(correct/60000.0*100)

