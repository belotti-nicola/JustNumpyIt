from src.neuralnetworks.genericnetwork import NeuralNetwork
from src.idx.idxhandler import IDX
from src.neuralnetworks.layers.denselayer import DenseL
from src.neuralnetworks.activations.softmax import SoftMaxL
from src.neuralnetworks.activations.relu import ReLUL
from src.neuralnetworks.costs.mse import MSEC
import numpy as np

ITERATIONS = 1000
alpha = .01

def print_image(image):
    i = 0
    for r in range(28):
        line = ""
        for c in range(28):
            line += "." if int(image[i]) > 80.0 else " "
            i += 1
        print(line)

def one_hot(integer):
    vect = np.zeros((10,1))
    vect[integer] = 1
    return vect

if __name__ == "__main__":
    train_images = IDX("data/MNIST/train-images.idx3-ubyte").numpy().reshape(60000,784,1)[1:20]
    train_label  = IDX("data/MNIST/train-labels.idx1-ubyte").numpy().reshape(60000,1)[1:20]


    NN = NeuralNetwork([
            DenseL(784,10),
            SoftMaxL()
    ],MSEC()
    )

    for i in range(ITERATIONS):
        for image,label in zip(train_images,train_label):
            labels_vector = one_hot(label)
            NN.forward(image)
            NN.backward(labels_vector)

            newb1 = NN.layers[0].b + alpha * NN.layers[0].db

            NN.layers[0].setb(newb1)

    counter = 0    

    for image,label in zip(train_images,train_label):
        prediction = NN.forward(image)
        if np.argmax(prediction) == label:
            counter = counter + 1
        print(np.argmax(prediction),label)

    print(counter)