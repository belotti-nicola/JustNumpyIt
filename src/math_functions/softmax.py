import numpy as np
from src.math_functions.matrix_enums import A

class SoftMax():
    @staticmethod
    def compute(x):
        e_x = np.exp(x - np.max(x))
        y = e_x / e_x.sum(axis=0)
        return y
        

