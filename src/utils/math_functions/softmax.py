import numpy as np

class SoftMax():
    @staticmethod
    def fun(x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())
    
    @staticmethod
    def der(x):       
        return np.multiply(
                    SoftMax.fun(x),
                    SoftMax.fun(1-x)
                )

