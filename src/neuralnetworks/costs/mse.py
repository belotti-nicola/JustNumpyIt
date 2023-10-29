from src.math_functions.mse import MSE


class MSEC:
    def forward(self,y,y_hat):
        self.input = y_hat
        self.output = MSE.compute(y,y_hat)
        return self.output

    def backward(self,y):
        return y-self.input