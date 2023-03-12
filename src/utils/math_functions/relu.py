import numpy as np

class ReLU():
    @staticmethod
    def fun(x):
        return np.maximum(x,0)
    
    @staticmethod
    def der(x):
        return (x > 0) * 1