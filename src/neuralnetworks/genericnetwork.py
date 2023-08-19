import numpy as np

class NeuralNetwork:
    def __init__(self,l,f):
        self.layers = l
        self.cost = f
        
    def forward(self,data):
        self.input = data
        
        tmp = data
        for layer in self.layers:
            tmp = layer.forward(tmp)
        self.output = tmp
        return tmp
    
    def getLayers(self):
        return self.layers
    def getReversedLayers(self):
        return self.layers[::-1]

    
    def backward(self,y):
        output = self.layers[-1].output
        self.cost.forward(output,y)

        grad = self.cost.backward(output,y)
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

