import numpy as np

class Sigmoid():
    @staticmethod
    def compute(x):
        y = 1/(1 + np.exp(-x))
        return y
        

