import numpy as np
from src.math_functions.softmax import SoftMax

class SoftMaxL:

    def forward(self,input):
        self.input = input
        self.output = SoftMax.compute(input) 
        return self.output

    def backward(self,output_gradient):
        shape = output_gradient.shape
        ones = np.ones(shape)
        return np.multiply(self.output,
                           (ones-self.output)
                           )

