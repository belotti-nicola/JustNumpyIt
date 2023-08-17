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
        return tmp
    
    def backward(self,desired_output):
        output = self.layers[-1].output
        self.cost.compute(output,desired_output)

        grad = self.cost.backward()
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

    def evaluate(self,TESTDATA,TESTLABELS):
        correct_guesses = 0
        for sample,label in zip(TESTDATA,TESTLABELS):
            output = self.forward(sample)
            predict = np.argmax(output)
            correct_guesses += 1 if predict == label else 0

        return correct_guesses
