import pytest
import numpy as np

from src.math_functions.softmax import SoftMax


testdata = [
    {
        "x":[1,2,3,4],
        "y":[.032,.087,.236,.643],
        },
    {
        "x":[1,-2,3,-4],
        "y":[.11, .005, .87, 0],
        },
]

def test_sigmoid():

    for t in testdata:
        x = np.array(t["x"],np.double)
        y = np.array(t["y"],np.double)
        assert np.linalg.norm(SoftMax.compute(x) - y) < .01

