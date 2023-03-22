import numpy as np
import tensorflow as tf
from NeuralNetwork import NeuralNetwork

class Mnist:

    def __init__(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        num_images = 60000
        num_pixels = 28*28
        #transforms picture matrix into a vector
        x_train_flat = np.zeros((num_images, num_pixels))
        for i in range(num_images):
            x_train_flat[i,:] = x_train[i,:,:].flatten()
        #same for test data
        x_test_flat = np.zeros((10000, num_pixels))
        for i in range(10000):
            x_test_flat[i,:] = x_test[i,:,:].flatten()
        
        self.xtrain = x_train_flat
        self.xtest = x_test_flat
        self.ytrain = y_train
        self.ytest = y_test
        self.nn = NeuralNetwork([784,32,10],["sigmoid","sigmoid"])
        
        self.nn.setWeightsFromFile("l1_weights.txt",1)
        self.nn.setBiasFromFile("l1_bias.txt",1)
        self.nn.setWeightsFromFile("l2_weights.txt",2)
        self.nn.setBiasFromFile("l2_bias.txt",2)


    def calculateEpoch(self):
        
        i = 0
        for i in range(len(self.xtrain)):
            #x is a row vector and needs to be transposed
            xtransp = np.transpose(np.array([self.xtrain[i]]))
            yvector = np.zeros((10,1))
            yvector[self.ytrain[i]] = 1
            self.nn.stochasticGradDesc(xtransp,yvector,0.1)
            i += 1
        
    def calculateAccuracy(self,command):
        xdataset = 0
        ydataset = 0

        if command == "test":
            xdataset = self.xtest
            ydataset = self.ytest
        elif command == "train":
            xdataset = self.xtrain
            ydataset = self.ytrain
        
        i = 0
        correct = 0
        for i in range(len(xdataset)):
            #x is a row vector and needs to be transposed
            xtransp = np.transpose(np.array([xdataset[i]]))
            #output is 10x1 vector
            outputlist = self.nn.forwardPropagate(xtransp)
            out = outputlist[-1][1]
            #argmax returns the index which has the highest value
            value = np.argmax(out)
            if value == ydataset[i]:
                correct += 1
            i += 1


        return correct/len(xdataset)*100
    
    def calculateMSE(self, command):
        xdataset = 0
        ydataset = 0

        if command == "test":
            xdataset = self.xtest
            ydataset = self.ytest
        elif command == "train":
            xdataset = self.xtrain
            ydataset = self.ytrain
        
        i = 0
        mse = 0
        for i in range(len(xdataset)):
            xtransp = np.transpose(np.array([xdataset[i]]))
            yvector = np.zeros((10,1))
            yvector[ydataset[i]] = 1
            outputlist = self.nn.forwardPropagate(xtransp)
            #print(ytrain)
            #print(outputlist[-1][1])
            error = yvector - outputlist[-1][1]
            mse += (np.square(error)).mean()
            i += 1
        
        return mse
    
    

        


"""







print(len(x_train_flat))
print(len(x_test_flat))


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
        
"""
    
