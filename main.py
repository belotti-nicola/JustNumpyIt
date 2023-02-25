from src.idx.idxhandler import IDX
from src.model.neuralnetwork import NeuralNetwork

data        = IDX('data/MNIST/train-images.idx3-ubyte').numpy()
labels      = IDX('data/MNIST/train-labels.idx1-ubyte').numpy()
test_data   = IDX('data/MNIST/t10k-images.idx3-ubyte').numpy()
test_labels = IDX('data/MNIST/t10k-labels.idx1-ubyte').numpy()

models = [
    NeuralNetwork([10,"Relu",10,"Softmax"])
]

for m in models:
    m.train(data,labels)
    m.evaluate(test_data,test_labels)

