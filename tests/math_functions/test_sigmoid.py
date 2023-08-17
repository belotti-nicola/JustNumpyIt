import pytest
import numpy as np

from src.math_functions.sigmoid import Sigmoid


testdata = [
    {
        "x":[0,0,0,0],
        "y":[.5,.5,.5,.5],
        },
    {
        "x":[1,-2,3,-4],
        "y":[.731, .119, .952, .017],
        }
]

def test_sigmoid():

    for t in testdata:
        x = np.array(t["x"],np.double)
        y = np.array(t["y"],np.double)
        assert np.linalg.norm(Sigmoid.compute(x) - y) < .01
