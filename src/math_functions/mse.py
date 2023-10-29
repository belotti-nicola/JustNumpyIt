import numpy as np

class MSE():
    @staticmethod
    def compute(prediction,desired):
        n = len(prediction)
        diff = prediction - desired
        mse = 0.5 * (np.square(diff)).mean(axis=0) / n
        return mse 

