import pytest
import numpy as np

from src.neuralnetworks.activations.relu import ReLUL


testdata = [
    ([0, 1,-3, 2],[0,1,0,2]),
    ([2, 2, 3, 1],[2,2,3,1]),
    ([1, 1,-3,-1],[1,1,0,0]),
    ([0,-6, 2, 0],[0,0,2,0])
]

activationLayer = ReLUL()

@pytest.mark.parametrize("inp,outp",testdata)
def test_forward(inp,outp):   
    x = np.array(inp,np.double).reshape(4,1)
    y = np.array(outp,np.double).reshape(4,1)
    computed = activationLayer.forward(x)
    assert np.linalg.norm(computed-y) < .001

@pytest.mark.parametrize("inp,outp",testdata)
def test_backward(inp,outp):
    x = np.array(inp,np.double).reshape(4,1)
    y = np.array(outp,np.double).reshape(4,1)
    computed = activationLayer.forward(x)
    assert np.linalg.norm(computed[0] - outp[0]) < .001
    assert np.linalg.norm(computed[1] - outp[1]) < .001
    assert np.linalg.norm(computed[2] - outp[2]) < .001
    assert np.linalg.norm(computed[3] - outp[3]) < .001
