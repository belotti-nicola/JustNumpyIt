import numpy as np

class ReLU():
    @staticmethod
    def compute(x):
        y = np.maximum(0,x)
        return y

