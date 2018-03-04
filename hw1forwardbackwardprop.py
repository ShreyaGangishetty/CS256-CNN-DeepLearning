import numpy as np
import csv

class NeuralNetwork(object):
    def __init__(self):
        self.inputLayerSize=2;
        self.hiddenLayerSize=5;
        self.outputLayerSize=2;
        self.WinputHidden=np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.WhiddenOutput=np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forwardProprgation(self,input):
        self.z2=np.dot(input,self.WinputHidden)
        self.a2=self.sigmoid(self.z2)
        self.z3=np.dot(self.a2,self.WhiddenOutput)
        self.yPredicted=self.sigmoid(self.z3)
        return self.yPredicted

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def softmax_norm(self):
        denom=np.exp(self.yPredicted[0][0])+np.exp(self.yPredicted[0][1])
        op1=np.exp(self.yPredicted[0][0])/denom
        op2=np.exp(self.yPredicted[0][1])/denom
        op=np.array([[op1,op2]])
        return op;

    def cross_entropy(self,output):
        softmax_op=self.softmax_norm()
        op=softmax_op.reshape(1,2)-output.reshape(1,2)
        return op;

    def costFunction(self,input,output):
        self.yPredicted = self.forwardProprgation(input)
        cost = 0.5*sum((output-self.yPredicted)**2)
        return cost

    def costFunctionPrime(self,input,output):
        self.yPredicted=self.forwardProprgation(input)
        deltatoinputlayer =np.multiply(-(output-self.yPredicted),self.sigmoidPrime(self.z3))
        DJdW2 = np.dot(self.a2.T,deltatoinputlayer)
        deltatohidden = np.dot(deltatoinputlayer,self.WhiddenOutput.T)*self.sigmoidPrime(self.z2)
        DJdW1 = np.dot(input.T,deltatohidden)
        return DJdW1,DJdW2


#total_data=np.loadtxt(open("dataset_csv.csv"),delimiter=",",skiprows=0)
data = open("dataset_csv.csv","r")
total_data = csv.reader(data)
print("total_data ",total_data)
x1=0
x2=0
y1=0
y2=0
for i in range(2):
    print("epoch : ",i)
    for num, row in enumerate(total_data):
        if (num != 0 and len(row)==3):
            x1 = float(row[0])
            x2 = float(row[1])
            y1 = float(row[2])
            if(y1==0 or y1=='0'):
                y2=float(1)
            elif(y1==1 or y1=='1'):
                y2=float(0)
        if(num<6):
            print(x1, x2, y1, y2)
        input = np.array([[x1, x2]])
        output = np.array([[y1, y2]])
        NN = NeuralNetwork()
        testInput = np.arange(-6, 6, 0.01)
        yHat = NN.forwardProprgation(input)
        cost1=NN.cross_entropy(output)
        if (num < 6):
            ## to display first 6 values
            print("error::",cost1)
        rc1, rc2 = NN.costFunctionPrime(input, output)
        learning_rate = 3
        NN.WinputHidden = NN.WinputHidden + learning_rate * rc1
        NN.WhiddenOutput = NN.WhiddenOutput + learning_rate * rc2
        cost2=NN.cross_entropy(output)
        rc1, rc2 = NN.costFunctionPrime(input, output)
        NN.WinputHidden = NN.WinputHidden - learning_rate * rc1
        NN.WhiddenOutput = NN.WhiddenOutput - learning_rate * rc2
        cost3 = NN.cross_entropy(output)
    data = open("dataset_csv.csv", "r")
    total_data = csv.reader(data)