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
            weights = np.random.uniform(-1,1,(self.layerSizeArray[i],self.layerSizeArray[i-1]))
            bias = np.random.uniform(-1,1,(self.layerSizeArray[i],1))
            activation = self.activationArray[i-1]
            network.append([weights,bias,activation])

        return network
    
    #sigmoid activation function
    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))
    
    def sigmoidDer(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
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
        layerOutputs = []
        a = xinput
        layerOutputs.append([a,a])
        for layer in self.network:
            weights = layer[0]
            bias = layer[1]
            activation = layer[2]
            z = np.matmul(weights, a) + bias
            if activation == "sigmoid": 
                a = self.sigmoid(z)
            elif activation == "linear":
                a = z
            layerOutputs.append([z,a])
        
        return layerOutputs
    
    def backPropagate(self, xTraining, yTraining):
        
        #forward propage and get the a's and z's of network
        outputlist=self.forwardPropagate(xTraining)
        
        #index of current layer, starting from top
        i=len(self.network)
        #print(i)
        gradients = [0]*i
        deltas = [0]*i
        #print(gradients)
        #the output of whole network
        yEstimate=outputlist[i][1]
        #print("yestimate")
        #print(yEstimate)
        #gL is the g-term for the output layer
        dJdy = np.transpose(yEstimate-yTraining)
        #print(dJdy)
        dadz = self.diagDerivativeMatrix(self.network[i-1][2],outputlist[i][0])
        #print(dadz)
        gL = np.matmul(dJdy,dadz)
        

        gtransp = np.transpose(gL)
        weightGrad = np.matmul(gtransp,np.transpose(outputlist[i-1][1]))
        biasGrad = gtransp
        
        gradients[i-1] = [weightGrad,biasGrad]
        deltas[i-1] = gL
        
        
        i -= 1
        g=gL
        #print(outputlist[i][1])
        #print(self.network[i][0])
        while i > 0:
            middle = np.matmul(g,self.network[i][0])
            diag = self.diagDerivativeMatrix(self.network[i-1][2],outputlist[i][0])
            g=np.matmul(middle,diag)
            gtransp = np.transpose(g)
            weightGrad = np.matmul(gtransp,np.transpose(outputlist[i-1][1]))
            biasGrad = gtransp
            gradients[i-1] = [weightGrad,biasGrad]
            deltas[i-1] = g
            i -= 1
        
        return gradients
    
    def stochasticGradDesc(self, xTraining, yTraining, learningRate):
        gradients = self.backPropagate(xTraining,yTraining)
        for i in range(len(gradients)):
            weightGrad = -1*learningRate*gradients[i][0]
            #print(weightGrad)
            biasGrad = -1*learningRate*gradients[i][1]
            self.network[i][0] = np.add(self.network[i][0],weightGrad)
            self.network[i][1] = np.add(self.network[i][1],biasGrad)
            
        

    def diagDerivativeMatrix(self, activation, z):
        size = len(z)
        diagMatrix = np.zeros((size,size))
        if activation == "sigmoid":
            for i in range(size):
                diagMatrix[i,i] = self.sigmoidDer(z[i,0])
        if activation == "linear":
            for i in range(size):
                diagMatrix[i,i] = 1
        
        return diagMatrix
    
    def __str__(self):
        dimensionText = ""
        for i in range(len(self.layerSizeArray)):
            if i == 0:
                dimensionText = "Input dimensions: " + str(self.layerSizeArray[0])+ "x1\n"
                continue
            #if i == len(self.layerSizeArray())
            dimensionText += ("Weight matrix " + str(i) + " dimensions: " + str(self.layerSizeArray[i]) + "x" + str(self.layerSizeArray[i-1]) +  "\n")
            dimensionText += self.network[i-1][0].__str__() + "\n"
            dimensionText += self.network[i-1][1].__str__() + "\n"
            dimensionText += "Activation: " + self.activationArray[i-1] + "\n"
            if i == len(self.layerSizeArray)-1:
                dimensionText += "Output dimensions: " + str(self.layerSizeArray[i])+ "x1\n"

        return dimensionText






        

    
