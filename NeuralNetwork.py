import numpy as np

class NeuralNetwork:

    def __init__(self, layerSizeArray, activationArray):
        self.layerSizeArray = layerSizeArray
        self.activationArray = activationArray
        self.network = self.createNetwork()
    
    def createNetwork(self):
        network = []
        for i in range(1,len(self.layerSizeArray)):
            weights = np.random.rand(self.layerSizeArray[i],self.layerSizeArray[i-1])
            bias = np.random.rand(self.layerSizeArray[i],1)
            activation = self.activationArray[i-1]
            network.append([weights,bias,activation])

        return network
    
    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))
    
    def setWeightsFromFile(self, filename, layer):
        print(filename)
        
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
        
        #print(self.network[layer-1][0])
        
    
    def setBiasFromFile(self, filename, layer):
        print(filename)
        
        file = open(filename, "r")
        i = 0
        for row in file:
            b = float(row)
            self.network[layer-1][1][i] = b
                
            i += 1
        
        #print(self.network[layer-1][1])


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

   






        

    
