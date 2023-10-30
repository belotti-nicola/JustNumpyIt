class NeuralNetwork:
    def __init__(self,layers,cost,data) -> None:
        self._layers = layers
        self._cost = cost
        self._data = data
    
    def getLayers(self):
        return self._layers
    def getData(self):
        return self._data