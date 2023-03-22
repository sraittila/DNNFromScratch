from Mnist import Mnist
import csv

def saveToFile(fileName,trainMSE, testMSE, trainAccuracy, testAccuracy, iterations):

    with open(fileName, "a") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter = ";")
        csv_writer.writerow([str(iterations),str(trainMSE),str(testMSE),str(trainAccuracy),str(testAccuracy)])



mnist = Mnist()

saveToFile("mnist.csv","MSE training", "MSE test", "Accuracy training","Accuracy test","Iterations")

accTrain = round(mnist.calculateAccuracy("train"),3)
accTest = round(mnist.calculateAccuracy("test"),3)
mseTrain = round(mnist.calculateMSE("train"),3)
mseTest = round(mnist.calculateMSE("test"),3)

print()
print("Initial training accuracy is " + str(accTrain) + "%")
print("Initial test accuracy is " + str(accTest) + "%")
print("Initial training MSE is " + str(mseTrain))
print("Initial test MSE is " + str(mseTest))
print()
saveToFile("mnist.csv",mseTrain, mseTest, accTrain,accTest,0)

epochs = 0
while True:
    command = input("Calculate an epoch?(y/n) ")
    if command == "n":
        print("Good bye!")
        break
    
    epochs += 1
    mnist.calculateEpoch()

    accTrain = round(mnist.calculateAccuracy("train"),3)
    accTest = round(mnist.calculateAccuracy("test"),3)
    mseTrain = round(mnist.calculateMSE("train"),3)
    mseTest = round(mnist.calculateMSE("test"),3)
    saveToFile("mnist.csv",mseTrain, mseTest, accTrain,accTest,epochs)
    
    print()
    print("Values after epoch " + str(epochs)+": ")
    print("Training accuracy is " + str(accTrain) + "%")
    print("Test accuracy is " + str(accTest) + "%")
    print("Training MSE is " + str(mseTrain))
    print("Test MSE is " + str(mseTest))
    print()






