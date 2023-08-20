import pytest
import numpy as np

from src.neuralnetworks.layers.denselayer import DenseL

A = np.array([[-1,  2,  1,  1],
              [ 2, -2, -1, -3],
              [ 3, -3,  2,  2]],
              np.double).reshape(3,4)
b = np.array([1,2,-3],np.double).reshape(3,1)

testdata = [
    ([0,1,-3,2],[2,-3,-8],[]),
    ([2,2,3,1],[7,-4,5],[]),
    ([1,1,-3,2],[1,-1,-5],[]),
    ([3,-6,2,-1],[-13,21,26],[]),
    ([1,1,-3,-1],[-2,8,-11],[])
]


@pytest.mark.parametrize("inp,outp,_unused",testdata)
def test_forward(inp,outp,_unused):
    L = DenseL(4,3)
    L.setA(A)
    L.setb(b)
    
    x = np.array(inp,np.double).reshape(4,1)
    y = np.array(outp,np.double).reshape(3,1)
    
    assert y.shape == (3,1)
    assert np.array_equal(L.forward(x),y)


