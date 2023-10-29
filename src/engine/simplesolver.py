class SimpleEngine:
    def __init__(self,) -> None:
        self._data = None
        self._neural_network = None
        self._solving_algorithm = None
    
    def forward(self,x):
        x_i = x
        for l in self._neural_network.layers():
            y_i = l.forward(x_i)
            x_i = y_i
    
    def backward(self,y_hat,label):
        self._neural_network.backward(y_hat,label)

    def optimize(self,ITERATION:int,labels):
        if( self._data is not None and
            self._neural_network is not None and
            self._solving_algorithm is not None):
            
            for i in range(ITERATION):
                for d in self._data:
                    y_hat = self.forward(d)
                    self.backward(y_hat,labels[i])

                for l in self._neural_network.layers():
                    l.update()
                

                