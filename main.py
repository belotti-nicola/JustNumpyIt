from src.idx.idxhandler import IDX
from src.model.neuralnetwork import NeuralNetwork

from src.utils.math_functions.relu import ReLU
from src.utils.math_functions.softmax import SoftMax

import numpy as np

import datetime

ITERATIONS = 10
eta = .001

def accuracy_computation(Y,L):
    counter = 0
    for i in range(60000):
        counter += np.argmax(Y[:,i]) == np.argmax(L[:,i])
    return counter / 600

def fun(y):
    return np.sum(y,axis=1)

def forward_propagation(data,A1,b1,A2,b2):
    B1 = np.dot(b1.T,np.ones((10,60000)))
    B2 = np.dot(b2.T,np.ones((10,60000)))

    Y1 = np.dot(A1,data/255) + B1
    Z1 = ReLU.fun(Y1)
    Y2 = np.dot(A2,Y1) + B2

    Z2 = SoftMax.fun(Y2)

    return Y1,Z1,Y2,Z2

def backward_propagation(data,labels,Y1,Z1,Y2,Z2):
    y_hat = Z2
    delta =  np.multiply( y_hat - labels , Z2 )

    dA2 = np.zeros((10,10),np.double)
    for i in range(60000):
        dA2 += np.dot(
                delta[:,[i]],
                Z1[:,[i]].T
        )
    db2 = fun(delta)

    delta = np.multiply( delta, Z1 ) 
    dA1 = np.zeros((10,784),np.double)
    for i in range(60000):
        dA1 += np.dot(
                delta[:,[i]],
                data[:,[i]].T
        )    
    db1 = fun(delta)
    return dA1,db1,dA2,db2

def label_encoding(vector):
    dim = max(vector.shape)
    retVal = np.zeros((10,dim),dtype=int)
    for i in range(dim):
        index = vector[0][i]
        retVal[index][i] = 1
    return retVal

def cost_computation(y_hats,labels):
    err = 0
    for i in range(60000):
        err += np.dot(
                (y_hats[:,[i]] - labels[:,[i]]).T,
                (y_hats[:,[i]] - labels[:,[i]])
        ).astype(np.double)
    
    return err[0]/60000

data        = IDX('data/MNIST/train-images.idx3-ubyte').numpy().reshape(60000,784).T
labels      = IDX('data/MNIST/train-labels.idx1-ubyte').numpy().reshape(60000,1).T
test_data   = IDX('data/MNIST/t10k-images.idx3-ubyte').numpy().reshape(10000,784).T
test_labels = IDX('data/MNIST/t10k-labels.idx1-ubyte').numpy().reshape(10000,1).T

#first layer
A1 = np.ndarray(shape=(10,784) ,  buffer=np.random.randn(10,784))
b1 = np.ndarray(shape=(10,)   ,  buffer=np.random.randn(10,1)) 

#second layer
A2 = np.ndarray(shape=(10,10)  ,  buffer=np.random.randn(10,10))
b2 = np.ndarray(shape=(10,)   ,  buffer=np.random.randn(10,1))

labels = label_encoding(labels)
LOSSES = []
for i in range(ITERATIONS):
    Y1,Z1,Y2,Z2 = forward_propagation(data,A1,b1,A2,b2)
    dA1,db1,dA2,db2 = backward_propagation(data,labels,Y1,Z1,Y2,Z2)


    A2 = A2 + eta * dA2
    b2 = b2 + eta * db2
    A1 = A1 + eta * dA1
    b1 = b1 + eta * db1

    COST = cost_computation(Y2,labels)
    accuracy = accuracy_computation(Y2,labels)
    LOSSES.append(COST)

    if( i%1 == 0 ): print(COST,datetime.datetime.now(),accuracy)