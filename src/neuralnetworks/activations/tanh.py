import numpy as np
from src.math_functions.tanh import Tanh

class SoftMaxL:
    def forward(self,input):
        self.output = Tanh.compute(input)
        return self.output
    def backward(self,output_gradient):
        pass    

