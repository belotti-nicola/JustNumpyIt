import pytest
import numpy as np

from src.math_functions.mse import MSE


testdata = [
    {
        "prediction":[1,2,3,4],
        "concrete":[1,2,3,4],
        "result":0
        },
    {
        "prediction":[1,2,3,4],
        "concrete":[0,1,2,3],
        "result":0.125
        },
]

def test_MSE():

    for t in testdata:
        y = np.array(t["concrete"],np.double)
        y_hat = np.array(t["prediction"],np.double)
        expected = t["result"]
        assert MSE.compute(y,y_hat) == expected

