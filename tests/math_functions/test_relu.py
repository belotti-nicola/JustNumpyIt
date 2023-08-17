import pytest
import numpy as np

from src.math_functions.relu import ReLU


testdata = [
    {
        "x":[1,2,3,4],
        "y":[1,2,3,4],
        },
    {
        "x":[1,-2,3,-4],
        "y":[1, 0,3, 0],
        },
]

def test_ReLU():

    for t in testdata:
        x = np.array(t["x"],np.double)
        y = np.array(t["y"],np.double)
        assert np.linalg.norm(ReLU.compute(x) - y) < .01

