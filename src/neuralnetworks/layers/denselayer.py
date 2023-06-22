class Dense:
    def __init__(self,inputSize, outputSize,dataTypeSelector) -> None:
        self.weights = createWeightsMatrix(
            inputSize,
            outputSize
        )
        self.biases  = createBiasesMatrix(
            outputSize,
        )
        
    def forward(self,input):
        Y = dotProduct(
            self.weights,
            input
        )
        return Y + self.biases
    
    def backward(self,input,output,error):
        delta = hadamaartProduct(output,error)
        deltaW = dotProduct(
            delta,
            self.weights.T
        )
        deltaB = delta
        return 