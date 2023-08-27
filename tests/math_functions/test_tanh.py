import pytest
import numpy as np

from src.math_functions.tanh import Tanh

testdata = [
    {
        "x":[1,2,3,4],
        "y":[.76,.96,.995,.999],
        },
    {
        "x":[1,-2,3,-4],
        "y":[.76, -.96, .995, -.999],
        },
]

def test_tanh():

    for t in testdata:
        x = np.array(t["x"],np.double)
        y = np.array(t["y"],np.double)
        assert np.linalg.norm(Tanh.compute(x) - y) < .01

