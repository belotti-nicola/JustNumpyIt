from src.neuralnetworks.general_network import NeuralNetwork

class SimpleEngine:
    def __init__(self,NN:NeuralNetwork,labels) -> None:
        self._nn = NN
        self._labels = labels
    
    def forward(self,x):
        layers = self._nn.getLayers()
        x_i = x
        for l in layers:
            y_i = l.forward(x_i)
            x_i = y_i
    
    def backward(self,label,y_hat):
        self._nn._cost.forward(label,y_hat)
        self._nn._cost.backward(label)
        for l in self._nn.getLayers():
            outp_g = l.backward(outp_g,label)
            

    def optimize(self,ITERATION:int,labels):
        data = self._nn.getData()
        for i in range(ITERATION):
            for d,l in zip(data,labels):
                y_hat = self.forward(d)
                self.backward(y_hat,l)