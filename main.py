from src.idx.idxhandler import IDX
from src.model.neuralnetwork import NeuralNetwork

from src.utils.math_functions.relu import ReLU
from src.utils.math_functions.softmax import SoftMax

import numpy as np

ITERATIONS = 1

data        = IDX('data/MNIST/train-images.idx3-ubyte').numpy().reshape(60000,784).T
labels      = IDX('data/MNIST/train-labels.idx1-ubyte').numpy()
test_data   = IDX('data/MNIST/t10k-images.idx3-ubyte').numpy()
test_labels = IDX('data/MNIST/t10k-labels.idx1-ubyte').numpy()

#first layer
A1 = np.ndarray(shape=(10,784), buffer=np.random.randn(10,784))
b1 = np.dot( 
        np.ndarray(shape=(10,1), buffer=np.random.randn(10,1)) ,
        np.ndarray(shape=(1,60000), buffer=np.ones(shape=(1,60000))) 
)

#second layer
A2 = np.ndarray(shape=(10,10),buffer=np.random.randn(10,10))
b2 = np.dot( 
        np.ndarray(shape=(10,1), buffer=np.random.randn(10,1)),
        np.ndarray(shape=(1,60000), buffer=np.ones(shape=(1,60000))) 
    )

def forward_propagation(data,A1,b1,A2,b2):
    y1 = ReLU.fun( np.dot(A1,data) + b1)
    y2 = SoftMax.fun(np.dot(A2,y1) + b2)
    return y1,y2

def backward_propagation(a,b):
    pass

for i in range(ITERATIONS):
    y1,y2 = forward_propagation(data,A1,b1,A2,b2)
    LOSS = np.dot(
        np.dot( 
            np.ndarray(shape=(1,10), buffer=np.ones(shape=(1,10))),
            y2-labels,
        ),
        np.dot(
            np.ndarray(shape=(1,10), buffer=np.ones(shape=(1,10))),
            y2-labels,
        ).T 
    )
    #backward_propagation(y1,y2)
 
    print(LOSS)
