from src.idx.idxhandler import IDX
from src.model.neuralnetwork import NeuralNetwork

from src.utils.math_functions.relu import ReLU
from src.utils.math_functions.softmax import SoftMax

import numpy as np

ITERATIONS = 1
eta = .008

data        = IDX('data/MNIST/train-images.idx3-ubyte').numpy().reshape(60000,784).T
labels      = IDX('data/MNIST/train-labels.idx1-ubyte').numpy().reshape(60000,1).T
test_data   = IDX('data/MNIST/t10k-images.idx3-ubyte').numpy().reshape(10000,784).T
test_labels = IDX('data/MNIST/t10k-labels.idx1-ubyte').numpy().reshape(10000,1).T

#first layer
A1 = np.ndarray(shape=(10,784) ,  buffer=np.random.randn(10,784))
b1 = np.ndarray(shape=(10,1)   ,  buffer=np.random.randn(10,1)) 

#second layer
A2 = np.ndarray(shape=(10,10)  ,  buffer=np.random.randn(10,10))
b2 = np.ndarray(shape=(10,1)   ,  buffer=np.random.randn(10,1))
     
def forward_propagation(data,A1,b1,A2,b2):
    Y1 = np.dot(A1,data) + b1
    A1 = ReLU.fun(Y1)
    Y2 = np.dot(A2,Y1) + b2
    A2 = SoftMax.fun(Y2)
    return Y1,A1,Y2,A2

def backward_propagation(data,y_hat,labels,y1,z1,y2,z2):
    dz2 = ( y_hat - labels ) * SoftMax.fun(y2)
    dA2 = np.multiply( dz2, z2.T )
    db2 = dz2

    dz1 = np.multiply( dz2, ReLU.der(y1) ) 
    dA1 = np.dot( dz1, data.T )
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
    Y1,A1,Y2,A2 = forward_propagation(data,A1,b1,A2,b2)
    dA1,dA2,db1,db2 = backward_propagation(data,A2,labels,Y1,A1,Y2,A2)
    
    A1 = A1 - eta * dA1
    b1 = b1 - eta * db1
    A2 = A2 - eta * dA2
    b2 = b2 - eta * db2