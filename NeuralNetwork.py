import numpy as np
#Implements neural network forward propagation
class NeuralNetwork:

    def __init__(self, layerSizeArray, activationArray):
        self.layerSizeArray = layerSizeArray #size of input, size of each layer, size of output
        self.activationArray = activationArray #actiovation function for every layer including the last. 
                                                #Assumes that input doesnt have activation
        self.network = self.createNetwork()   #initialize network weights
        self.prevGradients = []
    
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
    
    #derivative of sigmoid
    def sigmoidDer(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    #sets weights from a textfile.
    #It is assumed that text file is written as transpose of weight matrix
    #because it is the format of Tensorflow
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
        #creates an empty array to store gradients
        gradients = [0]*i
        
        #the output of whole network
        yEstimate=outputlist[i][1]
        
        #gL is the g-term of the output layer
        dJdy = np.transpose(yEstimate-yTraining)
        dadz = self.diagDerivativeMatrix(self.network[i-1][2],outputlist[i][0])
        gL = np.matmul(dJdy,dadz)

        #weight and bias gradients of the output layer
        gtransp = np.transpose(gL)
        weightGrad = np.matmul(gtransp,np.transpose(outputlist[i-1][1])) #g(l)*a(l-1)
        biasGrad = gtransp
        
        #stores gradients of output layer at the last index
        gradients[i-1] = [weightGrad,biasGrad]
        
        #loops through the rest of the network and calculates gradients for each layer
        i -= 1
        g=gL
        
        while i > 0:
            #g(l+1)*W(l+1)
            middle = np.matmul(g,self.network[i][0])
            #diag(derActivation(z(l)))
            diag = self.diagDerivativeMatrix(self.network[i-1][2],outputlist[i][0])
            g=np.matmul(middle,diag)
            gtransp = np.transpose(g)
            weightGrad = np.matmul(gtransp,np.transpose(outputlist[i-1][1])) #g(l)*a(l-1)
            biasGrad = gtransp
            gradients[i-1] = [weightGrad,biasGrad]
    
            i -= 1
        
        return gradients
    
    def stochasticGradDesc(self, xTraining, yTraining, learningRate, momentum):
        gradients = self.backPropagate(xTraining,yTraining)

        if len(self.prevGradients) == 0:
            for i in range(len(gradients)):
                self.prevGradients.append([0,0])
            
            for i in range(len(gradients)):
                self.network[i][0] = np.add(self.network[i][0],-1*learningRate*gradients[i][0])
                self.network[i][1] = np.add(self.network[i][1],-1*learningRate*gradients[i][1])
                self.prevGradients[i][0] = gradients[i][0]
                self.prevGradients[i][1] = gradients[i][1]
        else:
            for i in range(len(gradients)):
                
                gradWithMomentum = np.add(momentum*self.prevGradients[i][0] ,gradients[i][0])
                biasGradWithMomentum = np.add(momentum*self.prevGradients[i][1] ,gradients[i][1])

                self.network[i][0] = np.add(self.network[i][0],-1*learningRate*gradWithMomentum)
                self.network[i][1] = np.add(self.network[i][1],-1*learningRate*biasGradWithMomentum)
                self.prevGradients[i][0] = gradWithMomentum
                self.prevGradients[i][1] = biasGradWithMomentum

        
            
        

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






        

    
