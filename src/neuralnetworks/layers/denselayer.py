import numpy as np


class DenseL:
    def __init__(self,inputSize, outputSize) -> None:
        self.A = np.random.randn(outputSize,inputSize)
        self.b = np.random.randn(outputSize,1)

    def setA(self,newA):
        if(newA.shape == self.A.shape):
            self.A = newA
    def setb(self,newb):
        if(newb.shape == self.b.shape):
            self.b = newb
    def getA(self):
        return self.A
    def getb(self):
        return self.b
    

    def forward(self,input):
        self.input = input
        self.output = np.dot(self.A,input) + self.b
        return self.output
    
    def backward(self,output_gradient):
        self.dA = np.dot(self.A.T,output_gradient)
        self.db = output_gradient
        return np.dot(self.A.T,output_gradient)

