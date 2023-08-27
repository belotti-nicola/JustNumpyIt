import pytest
import numpy as np

from src.neuralnetworks.activations.tanh import Tanh,TanhL


testdata = [
    ([0, 1,-3, 2],[0, 0.7615,   -0.9950547536867304513319,   0.9640275800758168839464],[1,0.42011,0.0098754975,0.070704]),
    ([2, 2, 3, 1],[0.9640275800758168839464 , 0.9640275800758168839464,   0.9950547536867304513319,     0.7615],[0.070704,0.070704,0.0098754975,0.42011]),
    ([1, 1,-3,-1],[0.7615,  0.7615,   -0.9950547536867304513319,   -0.7615],[0.42011, 0.42011,0.0098754975,0.42011]),
    ([0,-6, 2, 0],[0,  -0.9999877116507955705644, 0.9640275800758168839464, 0],[1,0.0000399996,0.070704,1])
]

activationLayer = TanhL()

@pytest.mark.parametrize("inp,outp,_unused",testdata)
def test_forward(inp,outp,_unused):   
    x = np.array(inp,np.double).reshape(4,1)
    y = np.array(outp,np.double).reshape(4,1)
    computed = activationLayer.forward(x)
    assert np.absolute(computed[0] - y[0]) < .001
    assert np.absolute(computed[1] - y[1]) < .001
    assert np.absolute(computed[2] - y[2]) < .001
    assert np.absolute(computed[3] - y[3]) < .001


@pytest.mark.parametrize("inp,_unused,input_gradient",testdata)
def test_backward(inp,_unused,input_gradient):
    x = np.array(inp,np.double).reshape(4,1)
    input_gradient = np.array(input_gradient,np.double).reshape(4,1)
    activationLayer.forward(x)
    computed = activationLayer.backward(np.ones((4,1)))


    assert np.absolute(computed[0] - input_gradient[0] ) < .001
    assert np.absolute(computed[1] - input_gradient[1] ) < .001
    assert np.absolute(computed[2] - input_gradient[2] ) < .001
    assert np.absolute(computed[3] - input_gradient[3] ) < .001



