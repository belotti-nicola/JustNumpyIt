import numpy as np

class LayerDimensionsException(Exception):
    pass

class DenseL:
    def __init__(self,inputSize, outputSize) -> None:
        self.A = np.random.randn(outputSize,inputSize)
        self.b = np.random.randn(outputSize,1)

    def setA(self,newA):
        if(newA.shape == self.A.shape):
            self.A = newA
        else: 
            raise LayerDimensionsException("old shape: {0} - new shape {1}".format(self.A.shape,newA.shape))
    def setb(self,newb):
        if(newb.shape == self.b.shape):
            self.b = newb
        else: 
            raise LayerDimensionsException
    def getA(self):
        return self.A
    def getb(self):
        return self.b
    

    def forward(self,x):
        self.input = x
        self.output = np.dot(self.A,x) + self.b
        return self.output
    
    def backward(self,output_gradient):
        self.dA = np.dot(output_gradient,self.input.T)
        self.db = output_gradient
        return np.dot(self.A.T,output_gradient)

