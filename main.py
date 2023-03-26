from src.idx.idxhandler import IDX
from src.model.neuralnetwork import NeuralNetwork

from src.utils.math_functions.relu import ReLU
from src.utils.math_functions.softmax import SoftMax

import numpy as np

ITERATIONS = 1
alpha = .08

data        = IDX('data/MNIST/train-images.idx3-ubyte').numpy().reshape(60000,784).T
labels      = IDX('data/MNIST/train-labels.idx1-ubyte').numpy().reshape(60000,1).T
test_data   = IDX('data/MNIST/t10k-images.idx3-ubyte').numpy().reshape(10000,784).T
test_labels = IDX('data/MNIST/t10k-labels.idx1-ubyte').numpy().reshape(10000,1).T

#first layer
A1 = np.ndarray(shape=(10,784), buffer=np.random.randn(10,784))
b1 = np.ndarray(shape=(10,1)  , buffer=np.random.randn(10,1)) 

#second layer
A2 = np.ndarray(shape=(10,10),buffer=np.random.randn(10,10))
b2 = np.ndarray(shape=(10,1)   , buffer=np.random.randn(10,1))
     
def forward_propagation(data,A1,b1,A2,b2):
    Y1 = np.dot(A1,data) + b1
    Z1 = ReLU.fun(Y1)
    Y2 = np.dot(A2,Y1)   + b2
    Z2 = SoftMax.fun(Y2)
    return Y1,Z1,Y2,Z2

def backward_propagation(y_hat,labels,y1,z1,y2,z2):
    dz2 = ( y_hat - labels ) * SoftMax.fun(y2)
    dA2 = np.dot ( dz2, z2.T )
    db2 = dz2

    dz1 = np.dot( A1, dz2 ) * ReLU.der(y1)
    dA1 = np.dot( dz1, z1.T )
    db1 = dz1

    return dA1,db1,dA2,db2


def label_encoding(vector): 
    dim = max(vector.shape)
    substitution_for_vector = np.zeros((dim,10),dtype=int)
    for i in range(dim):
        index = vector[i]
        substitution_for_vector[i][index] = 1
    return substitution_for_vector


for i in range(ITERATIONS):
    dA1,db1,dA2,db2 = 0,0,0,0
    for d in data:
        Y1,z1,Y2,z2 = forward_propagation(data,A1,b1,A2,b2)
        ddA1,ddb1,ddA2,ddb2 = backward_propagation(z2,labels,Y1,z1,Y2,z2)
        dA2 = dA2+ddA2
        db2 = db2+ddb2
        dA1 = dA1+ddA1
        db1 = db1+ddb1

    A1 = A1 - alpha * dA1
    b1 = b1 - alpha * db1
    A2 = A2 - alpha * dA2
    b2 = b2 - alpha * db2