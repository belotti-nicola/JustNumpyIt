import numpy as np
from src.math_functions.tanh import Tanh

class TanhL:
    def forward(self,input):
        self.input  = input
        self.output = Tanh.compute(input)
        return self.output
    def backward(self,output_gradient):
        shape = self.input.shape
        tanh_square = np.multiply(
            self.input,
            self.input
        )
        return np.multiply(
            output_gradient,
            np.ones(shape)-tanh_square
        )
