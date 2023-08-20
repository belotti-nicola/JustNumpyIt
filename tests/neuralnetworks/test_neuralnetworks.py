import pytest

from src.neuralnetworks.genericnetwork import NeuralNetwork
from src.neuralnetworks.layers.denselayer import DenseL
from src.neuralnetworks.activations.relu import ReLUL
from src.neuralnetworks.activations.softmax import SoftMaxL
from src.neuralnetworks.costs.mse import MSEC

import numpy as np
from src.optimizer.descent_algorithms import gradient_descent

testdata = [
    ([0, 1,-3, 2],  [0.9932623568, 0.006692549117,     0.00004509404124],[0,1,0]),
    ([2, 2, 3, 1],  [0.88078,      0.000014,           0.11920         ],[0,0,1]),
    ([1, 1,-3,-1],  [0.00004,      0.9999,             0.000000005     ],[1,0,0]),
    ([0,-6, 2, 0],  [0,            0.0009,             0.9990889       ],[1,0,0])
]

D = DenseL(4,3)
D.setA(
    np.array([[-1, 2, 1, 1],
              [ 2,-2,-1,-3],
              [ 3,-3, 2, 2]],np.double)
)
D.setb(
    np.array( [ 1, 2, -3],np.double).reshape(3,1)
)

NN = NeuralNetwork([
        D,
        SoftMaxL()
    ],MSEC()
    )


@pytest.mark.parametrize("inp,outp,_unused",testdata)
def test_forward(inp,outp,_unused):   
    x = np.array(inp,np.double).reshape(4,1)
    y = NN.forward(x)
    expected = np.array(outp,np.double).reshape(3,1)
    assert y.shape == (3,1)
    assert y[0] - expected[0] < .01
    assert y[1] - expected[1] < .01
    assert y[2] - expected[2] < .01

@pytest.mark.parametrize("inp,_unused,label",testdata)
def test_backward(inp,_unused,label):
    x = np.array(inp,np.double).reshape(4,1)
    
    NN.forward(x)
    y = np.array(label,np.double).reshape(3,1)
    NN.backward(y)


    assert NN.layers[0].dA is not None
    assert NN.layers[0].db is not None
    assert NN.layers[1] is not None
    assert NN.layers[1] is not None

@pytest.mark.parametrize("inp,_unused,label",testdata)
def test_one_step_cost_reduction(inp,_unused,label):
    x = np.array(inp,np.double).reshape(4,1)
    y = np.array(label,np.double).reshape(3,1)

    out1 = NN.forward(x)
    C1 = NN.cost.forward(out1,y)

    NN.backward(y)

    newb = NN.layers[0].b + .1 * NN.layers[0].db
    NN.layers[0].setb(newb)
    out2 = NN.forward(x)
    C2 = NN.cost.forward(out2,y)

    assert C2 - C1 > 0

@pytest.mark.parametrize("inp,_unused,label",testdata)
def test_one_step_cost_reduction_dA(inp,_unused,label):
    x = np.array(inp,np.double).reshape(4,1)
    y = np.array(label,np.double).reshape(3,1)

    out1 = NN.forward(x)
    C1 = NN.cost.forward(out1,y)

    NN.backward(y)

    newA = NN.layers[0].A + .1 * NN.layers[0].dA
    NN.layers[0].setA(newA)
    out2 = NN.forward(x)
    C2 = NN.cost.forward(out2,y)

    assert C2 - C1 > 0