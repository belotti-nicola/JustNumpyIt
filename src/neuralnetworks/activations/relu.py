import numpy as np
from src.math_functions.relu import ReLU

class ReLUL:
    def forward(self,input):
        self.input = input
        self.output = ReLU.compute(input)
        return self.output
    
    def backward(self,output_gradient):
        shape = output_gradient.shape
        return np.greater(self.input, 0).astype(int)