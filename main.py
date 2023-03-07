from src.idx.idxhandler import IDX
from src.model.neuralnetwork import NeuralNetwork

from src.utils.math_functions.relu import ReLU
from src.utils.math_functions.softmax import SoftMax

import numpy as np

ITERATIONS = 1
alpha = .2

data        = IDX('data/MNIST/train-images.idx3-ubyte').numpy().reshape(60000,784).T
labels      = IDX('data/MNIST/train-labels.idx1-ubyte').numpy()
test_data   = IDX('data/MNIST/t10k-images.idx3-ubyte').numpy()
test_labels = IDX('data/MNIST/t10k-labels.idx1-ubyte').numpy()

#first layer
A1 = np.ndarray(shape=(10,784), buffer=np.random.randn(10,784))
b1 = np.dot( 
        np.ndarray(shape=(10,1)   , buffer=np.random.randn(10,1)) ,
        np.ndarray(shape=(1,60000), buffer=np.ones(shape=(1,60000))) 
    )

#second layer
A2 = np.ndarray(shape=(10,10),buffer=np.random.randn(10,10))
b2 = np.dot( 
        np.ndarray(shape=(10,1)   , buffer=np.random.randn(10,1)),
        np.ndarray(shape=(1,60000), buffer=np.ones(shape=(1,60000))) 
    )

def forward_propagation(data,A1,b1,A2,b2):
    y1 = ReLU.fun( np.dot(A1,data) + b1)
    y2 = SoftMax.fun(np.dot(A2,y1) + b2)
    return y1,y2

def backward_propagation(y1,y2):
    print(y2.shape)
    dA2 = np.dot(SoftMax.der(y2),y2)
    db2 = np.dot(SoftMax.der(y2),1)
    dA1 = np.dot(
            dA2,
            np.dot(ReLU.der(y1),y1)
    )
    db1 = np.dot(
            dA2,
            np.dot(ReLU.der(y1),1)
    )

    return dA1,db1,dA2,db2


def label_encoding(vector): 
    dim = max(vector.shape)
    substitution_for_vector = np.zeros((dim,10),dtype=int)
    for i in range(dim):
        index = vector[i]
        substitution_for_vector[i][index] = 1
    return substitution_for_vector


labels_matrix = label_encoding(labels).T
for i in range(ITERATIONS):
    y1,y2 = forward_propagation(data,A1,b1,A2,b2)
    LOSS = ((y2 - labels_matrix)**2).mean(axis=1)
    dA1,db1,dA2,db2 = backward_propagation(y1,y2)
    A1 = A1 + alpha * dA1 
    b1 = b1 + alpha * db1 
    A2 = A2 + alpha * dA2
    b2 = b2 + alpha * db2