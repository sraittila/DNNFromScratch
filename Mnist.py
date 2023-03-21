import numpy as np
import tensorflow as tf
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


nn = NeuralNetwork([784,32,10],["sigmoid","sigmoid"])

nn.setWeightsFromFile("l1_weights.txt",1)
nn.setBiasFromFile("l1_bias.txt",1)
nn.setWeightsFromFile("l2_weights.txt",2)
nn.setBiasFromFile("l2_bias.txt",2)

i = 0
correct = 0
for i in range(60000):
    #x is a row vector and needs to be transposed
    xtransp = np.transpose(np.array([x_train_flat[i]]))
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

j = 0
while j < 5:

    i = 0
    for i in range(60000):
        #x is a row vector and needs to be transposed
        xtransp = np.transpose(np.array([x_train_flat[i]]))
        ytrain = np.zeros((10,1))
        ytrain[y_train[i]] = 1
        nn.stochasticGradDesc(xtransp,ytrain,0.1)
        i += 1

    i = 0
    mse = 0
    for i in range(60000):
        xtransp = np.transpose(np.array([x_train_flat[i]]))
        ytrain = np.zeros((10,1))
        ytrain[y_train[i]] = 1
        outputlist = nn.forwardPropagate(xtransp)
        #print(ytrain)
        #print(outputlist[-1][1])
        error = ytrain - outputlist[-1][1]
        mse += (np.square(error)).mean()
        i += 1
    
    print("Epoch " +str(j+1)+ " MSE: " + str(mse))
    print()
    i = 0
    correct = 0
    for i in range(60000):
        #x is a row vector and needs to be transposed
        xtransp = np.transpose(np.array([x_train_flat[i]]))
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
    
    
    j += 1
    



print(nn)
#checks how many numbers neural network guesses correct
#Expected 90.98%
i = 0
correct = 0
for i in range(60000):
    #x is a row vector and needs to be transposed
    xtransp = np.transpose(np.array([x_train_flat[i]]))
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
        

    
