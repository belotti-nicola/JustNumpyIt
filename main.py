from src.neuralnetworks.concrete.genericnetwork import NeuralNetwork
from src.idx.idxhandler import IDX
from src.neuralnetworks.layers.denselayer import DenseL
from src.neuralnetworks.activations.softmax import SoftMaxL
from src.neuralnetworks.activations.relu import ReLUL
from src.neuralnetworks.costs.mse import MSEC
import numpy as np

ITERATIONS = 1
alpha = .1

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
    train_images = IDX("data/MNIST/train-images.idx3-ubyte").numpy().reshape((60000,784,1))[0:1]
    train_label  = IDX("data/MNIST/train-labels.idx1-ubyte").numpy().reshape(60000,1)[0:1]


    NN = NeuralNetwork([
            DenseL(784,10),
            SoftMaxL()
    ],MSEC()
    )



    for i in range(ITERATIONS):
        for image,label in zip(train_images,train_label):
            labels_vector = one_hot(label[0])
            NN.forward(image/255)
            NN.backward(labels_vector)
            
        newb1 = NN.layers[0].b + alpha * NN.layers[0].db
        NN.layers[0].setb(newb1)
        newA = NN.layers[0].A + alpha * NN.layers[0].dA
        NN.layers[0].setA(newA)
        print("IT",np.max(NN.layers[0].dA),np.max(NN.layers[0].db))

    
    c = 0
    for i in range(1):
        output = NN.forward(train_images[i])
        prediction = np.argmax(output)
        print(output.T)
        if(prediction == train_label[i][0]):
            c+=1

    print(c)    
