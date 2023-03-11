import numpy as np

class SoftMax():
    @staticmethod
    def fun(x):
        x = x - x.max(axis=None, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=None, keepdims=True)
    
    @staticmethod
    def der(x):       
        return np.dot(SoftMax.fun(x),1-SoftMax.fun(x).T)   