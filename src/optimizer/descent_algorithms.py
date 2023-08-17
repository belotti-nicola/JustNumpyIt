def stocastic_gradient_descent():
    pass

def gradient_descent(neuralnetwork,
                     iterations,
                     alpha,
                     data,
                     labels,
                     costfunction):

    grad = costfunction.derivative(data,labels)

    activations = []
    for image,label in zip(data,labels):
        x = image
        for l in neuralnetwork.layers:
            y = l.forward(x)
            x = y
            activations.append(y)
        
