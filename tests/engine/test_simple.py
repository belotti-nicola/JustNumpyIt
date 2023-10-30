import numpy as np

from src.engine.simplesolver import SimpleEngine
from src.neuralnetworks.general_network import NeuralNetwork

from src.neuralnetworks.costs.mse import MSEC
from src.neuralnetworks.activations.softmax import SoftMaxL
from src.neuralnetworks.layers.denselayer import DenseL

A = np.array([[1, -1, 2],
              [2, -3, 1]],
              np.double).reshape(2,3)
b = np.array([[1],[1]],
              np.double).reshape(2,1)

def test_very_simple():
    data = [(1,1,1)]
    D = DenseL(3,2)
    D.setA(A)
    D.setb(b)
    layers = [
        D,
        SoftMaxL()
    ]
    cost = MSEC()
    labels = [
        [0,1]
    ]
    NN = NeuralNetwork(layers,cost,data)
    se = SimpleEngine(NN,labels)

    se.optimize(1,labels)

    assert NN.getLayers()[0].dA[0][0] == 1
    

