import numpy as np
from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([2,4,4,1],["sigmoid","sigmoid","sigmoid"])

print(nn)

x_train = [[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]]]
y_train = [[[1]],[[1]],[[0]],[[0]]]

for x in x_train:
    xtrain = np.array(x)
    
    out = nn.forwardPropagate(x)
    print(out[-1][1])

j = 0
while j < 5000:

    i = 0
    for x in x_train:
        xtrain = np.array(x)
        ytrain = np.array(y_train[i])
        nn.stochasticGradDesc(xtrain,ytrain,1)
        i += 1
    
    j += 1


print(nn)
for x in x_train:
    xtrain = np.array(x)
    
    out = nn.forwardPropagate(x)
    print(out[-1][1])