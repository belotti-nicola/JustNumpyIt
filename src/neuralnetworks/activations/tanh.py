import numpy as np
from src.math_functions.tanh import Tanh

class SoftMaxL:
    @staticmethod
    def forward(input):
        return Tanh.compute(input)
    @staticmethod
    def backward(self,output_gradient):
        pass    

