import numpy as np
#Implements neural network forward propagation
class NeuralNetwork:

    def __init__(self, layerSizeArray, activationArray):
        self.layerSizeArray = layerSizeArray #size of input, size of each layer, size of output
        self.activationArray = activationArray #actiovation function for every layer including the last. 
                                                #Assumes that input doesnt have activation
        self.network = self.createNetwork()   #initialize network weights
    
    #initializes network with random weights
    def createNetwork(self):
        network = []
        for i in range(1,len(self.layerSizeArray)):
            weights = np.random.rand(self.layerSizeArray[i],self.layerSizeArray[i-1])
            bias = np.random.rand(self.layerSizeArray[i],1)
            activation = self.activationArray[i-1]
            network.append([weights,bias,activation])

        return network
    
    #sigmoid activation function
    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))
    
    #sets weights from a textfile.
    #It is assumed that text file is written as transpose of weight matrix
    def setWeightsFromFile(self, filename, layer):
        file = open(filename, "r")
        i = 0
        for row in file:
            j = 0
            rowvector = row.split()
            for weight in rowvector:
                w = float(weight)
                self.network[layer-1][0][j,i] = w
                j += 1

            i += 1
        
            
    #sets bias from a textfile
    #bias is assumed to be written as a column vector
    def setBiasFromFile(self, filename, layer):
        
        file = open(filename, "r")
        i = 0
        for row in file:
            b = float(row)
            self.network[layer-1][1][i] = b
                
            i += 1
        

    #forward propagates through network
    #returns the output vector
    def forwardPropagate(self,xinput):
        a = xinput
        for layer in self.network:
            weights = layer[0]
            bias = layer[1]
            activation = layer[2]
            z = np.matmul(weights, a) + bias
            if activation == "sigmoid": 
                a = self.sigmoid(z)
            elif activation == "linear":
                a = z
        
        return a
    
    def __str__(self):
        dimensionText = ""
        for i in range(len(self.layerSizeArray)):
            if i == 0:
                dimensionText = "Input dimensions: " + str(self.layerSizeArray[0])+ "x1\n"
                continue
            #if i == len(self.layerSizeArray())
            dimensionText += ("Weight matrix " + str(i) + " dimensions: " + str(self.layerSizeArray[i]) + "x" + str(self.layerSizeArray[i-1]) +  "\n")
            #dimensionText += self.network[i-1][0].__str__() + "\n"
            dimensionText += "Activation: " + self.activationArray[i-1] + "\n"
            if i == len(self.layerSizeArray)-1:
                dimensionText += "Output dimensions: " + str(self.layerSizeArray[i])+ "x1\n"

        return dimensionText






        

    
