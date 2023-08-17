import numpy as np
from src.math_functions.relu import ReLU

class ReLUL:
    @staticmethod
    def forward(input):
        return ReLU.compute(input)
    @staticmethod
    def backward(self,output_gradient):
        pass    

