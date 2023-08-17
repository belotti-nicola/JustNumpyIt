from src.math_functions.mse import MSE


class CrossEntropyC:
    def forward(y,y_hat):
        for prediction,expected in zip(y,y_hat):
            MSE.compute(prediction,expected)

    def backward(prediction,expected):
        return prediction-expected