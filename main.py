from src.neuralnetworks.layers.denselayer import Dense
from src.neuralnetworks.genericnetwork import NeuralNetwork
from src.idx.idxhandler import IDX



ITERATIONS = 1000

def print_image(image):
    i = 0
    for r in range(28):
        line = ""
        for c in range(28):
            line += "." if int(image[i]) > 80.0 else " "
            i += 1
        print(line)


if __name__ == "__main__":
    train_images = IDX("data/MNIST/train-images.idx3-ubyte").numpy().reshape(60000,784,1)
    train_label  = IDX("data/MNIST/train-labels.idx1-ubyte").numpy().reshape(60000,1)

    NN = NeuralNetwork([
        Dense(784,10),
        SoftMax()
    ],MSE())
