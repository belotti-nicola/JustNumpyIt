import pytest
import numpy as np

from src.neuralnetworks.layers.denselayer import DenseL

A = np.array([[-1,  2,  1,  1],
              [ 2, -2, -1, -3],
              [ 3, -3,  2,  2]],
              np.double)
b = np.array([1,2,-3],np.double).reshape(3,1)

testdata = [
    {
        "x":[0,1,-3,2],
        "y":[2,-3,-8],
        "grad":[]
        },
    {
        "x":[2,2,3,1],
        "y":[7,-4,5],
        "grad":[]
        },
]



def test_forward():
    L = DenseL(4,3)
    L.setA(A)
    L.setb(b)

    for t in testdata:
        x = np.array(t["x"],np.double).reshape(4,1)
        y = np.array(t["y"],np.double).reshape(3,1)
        assert np.array_equal(L.forward(x),y )

def test_backward():
    L = DenseL(4,3)
    L.setA(A)
    L.setb(b)
    for t in testdata:
        x = np.array(t["x"],np.double).reshape(4,1)
        L.backward(np.ones((3,1)))
        assert np.array_equal(L.dA,
                              np.dot(A.T,np.ones((3,1))))
        assert np.array_equal(L.db,np.ones((3,1)))

