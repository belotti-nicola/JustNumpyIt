import numpy as np

class SoftMax():
    @staticmethod
    def fun(x):
        x = x - x.max(axis=None, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=None, keepdims=True)
    
    @staticmethod
    def der(x):
        return np.dot(fun(x),1-fun(x))   