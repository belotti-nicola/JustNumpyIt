import numpy 

class MSE():
    @staticmethod
    def fun(y_hat:numpy.ndarray, y_desired:numpy.ndarray):
        return (y_hat -y_desired ) * (y_hat -y_desired).T
        
    def der(x):
        return 2 