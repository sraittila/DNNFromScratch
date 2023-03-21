import numpy as np

def createWeights(filename, m , n):
    f = open(filename, "w")
    matrix = np.random.normal(0, 1, size=(m, n))
    for i in range(m):
        row = ""
        for j in range(n):
            row += str(matrix[i][j]) + " "
        
        f.write(row + "\n")

    f.close()


createWeights("l1_weights.txt",784,32)
createWeights("l1_bias.txt",32,1)
createWeights("l2_weights.txt",32,10)
createWeights("l2_bias.txt",10,1)




