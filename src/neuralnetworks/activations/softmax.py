import numpy as np
from src.math_functions.softmax import SoftMax

class SoftMaxL:
    @staticmethod
    def forward(input):
        return SoftMax.compute(input)
    @staticmethod
    def backward(self,output_gradient):
        pass    

