import numpy as np
from src.math_functions.matrix_enums import A

class Sigmoid():
    @staticmethod
    def compute(x):
        y = 1/(1 + np.exp(-x))
        return y
        

