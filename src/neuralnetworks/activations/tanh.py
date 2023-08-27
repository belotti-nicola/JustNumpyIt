import numpy as np
from src.math_functions.tanh import Tanh

class TanhL:
    def forward(self,input):
        self.input  = input
        self.output = Tanh.compute(input)
        return self.output
    def backward(self,output_gradient):
        shape = self.input.shape
        
        tanh_der = np.ones(shape) - np.multiply(self.output,self.output)


        return np.multiply(tanh_der,output_gradient)