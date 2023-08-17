import numpy as np
from src.math_functions.matrix_enums import A

class ReLU():
    @staticmethod
    def compute(x):
        y = np.maximum(0,x)
        return y

