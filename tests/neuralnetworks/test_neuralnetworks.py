import pytest

from src.neuralnetworks.genericnetwork import NeuralNetwork
from src.neuralnetworks.layers.denselayer import DenseL
from src.neuralnetworks.activations.relu import ReLUL
from src.neuralnetworks.activations.softmax import SoftMaxL
from src.neuralnetworks.costs.mse import MSE

import numpy as np
from src.optimizer.descent_algorithms import gradient_descent

testdata = [
    ([0,1,-3,2], [0.9932623568, 0.006692549117,     0.00004509404124]),
    ([2,2,3,1],  [0.88078,      0.000014,           0.11920]),
    ([1,1,-3,-1],[0.00004,      0.9999,             0.000000005]),
    ([0,6,-2,0], [0,            0.0009,             0.9990889])
]

D = DenseL(3,4)
D.setA(
    np.array([[-1,2,1,1],
              [2,2,-1,-3],
              [3,-3,2,2]],np.double)
)
D.setb(
    np.array([1,2,-3],np.double).reshape(3,1)
)

NN = NeuralNetwork([
        D,
        SoftMaxL
    ],MSE
    )


@pytest.mark.parametrize("inp,outp",testdata)
def test_forward(inp,outp):   
    x = np.array(inp,np.double).reshape(4,1)
    y = NN.forward(x)
    expected = np.array(outp,np.double).reshape(3,1)
    assert y.shape == (3,1)
    assert y[0] - expected[0] < .01
    assert y[1] - expected[1] < .01
    assert y[2] - expected[2] < .01


'''
def test_backward():
    assert True



def test_one_step_cost_reduction():
    NN = NeuralNetwork([
        DenseL(4,3),
        ReLUL,
        DenseL(3,3),
        SoftMaxL
    ],  MSE
    )
    x1 = np.array([1,-1,2,1],np.double).reshape(4,1)
    x2 = np.array([-1,0,2,-1],np.double).reshape(4,1)
    tests = [x1,x2]
    labels = [0,1]
    C1 = NN.evaluate(tests,labels)
    gradient_descent(NN,1,.01,tests,labels,MSE)
    C2 = NN.evaluate(tests,labels)

    assert True
'''
