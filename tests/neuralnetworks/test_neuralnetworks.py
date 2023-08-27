import pytest

from src.neuralnetworks.concrete.genericnetwork import NeuralNetwork
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
    assert np.absolute(y[0] - expected[0]) < .001
    assert np.absolute(y[1] - expected[1]) < .001
    assert np.absolute(y[2] - expected[2]) < .001

@pytest.mark.parametrize("inp,_unused,label",testdata)
def test_backward(inp,_unused,label):
    x = np.array(inp,np.double).reshape(4,1)
    
    out = NN.forward(x)
    y = np.array(label,np.double).reshape(3,1)
    NN.backward(y)

    SMderivative = NN.layers[1].backward(out)
    expected_db = np.multiply(
        np.multiply(SMderivative,1-SMderivative),
        out-y
    )
    expected_dA = np.dot(
        expected_db,
        NN.layers[0].input.T
    )

    assert NN.layers[0].dA.shape == expected_dA.shape

    assert np.absolute(NN.layers[0].dA[0][0] - expected_dA[0][0]) < .01
    assert np.absolute(NN.layers[0].dA[0][1] - expected_dA[0][1]) < .01
    assert np.absolute(NN.layers[0].dA[0][2] - expected_dA[0][2]) < .01
    assert np.absolute(NN.layers[0].dA[0][3] - expected_dA[0][0]) < .01

    assert np.absolute(NN.layers[0].dA[1][0] - expected_dA[1][0]) < .01
    assert np.absolute(NN.layers[0].dA[1][1] - expected_dA[1][1]) < .01
    assert np.absolute(NN.layers[0].dA[1][2] - expected_dA[1][2]) < .01
    assert np.absolute(NN.layers[0].dA[1][3] - expected_dA[1][3]) < .01

    assert np.absolute(NN.layers[0].dA[2][0] - expected_dA[2][0]) < .01
    assert np.absolute(NN.layers[0].dA[2][1] - expected_dA[2][1]) < .01
    assert np.absolute(NN.layers[0].dA[2][2] - expected_dA[2][2]) < .01
    assert np.absolute(NN.layers[0].dA[2][3] - expected_dA[2][3]) < .01



    assert np.absolute(NN.layers[0].db[0] - expected_db[0]) < .01
    assert np.absolute(NN.layers[0].db[1] - expected_db[1]) < .01
    assert np.absolute(NN.layers[0].db[2] - expected_db[2]) < .01


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