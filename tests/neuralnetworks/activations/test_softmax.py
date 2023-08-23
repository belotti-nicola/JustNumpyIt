import pytest
import numpy as np

from src.neuralnetworks.activations.softmax import SoftMaxL,SoftMax


testdata = [
    ([0, 1,-3, 2],[0.089628824664082, 0.24363640539052,   0.0044623564212819,   0.66227241352412]),
    ([2, 2, 3, 1],[0.19661193324148,  0.19661193324148,   0.53444664538852,     0.072329488128513]),
    ([1, 1,-3,-1],[0.46432780248952,  0.46432780248952,   0.0085044603563976,   0.062839934664554]),
    ([0,-6, 2, 0],[0.10647886802891,  2.6393472589564E-4, 0.78677832921628,     0.10647886802891])
]

activationLayer = SoftMaxL()

@pytest.mark.parametrize("inp,outp",testdata)
def test_forward(inp,outp):   
    x = np.array(inp,np.double).reshape(4,1)
    y = np.array(outp,np.double).reshape(4,1)
    computed = activationLayer.forward(x)
    assert np.linalg.norm(computed[0] - outp[0]) < .001
    assert np.linalg.norm(computed[1] - outp[1]) < .001
    assert np.linalg.norm(computed[2] - outp[2]) < .001
    assert np.linalg.norm(computed[3] - outp[3]) < .001



@pytest.mark.parametrize("inp,outp",testdata)
def test_backward(inp,outp):
    x = np.array(inp,np.double).reshape(4,1)
    computed = activationLayer.forward(x)
    input_gradient = activationLayer.backward(np.ones((4,1)))

    sm_gradient = np.multiply(SoftMax.compute(x),np.ones((4,1))-SoftMax.compute(x))
    expected_input_gradient = np.multiply(
        np.ones((4,1)),
        sm_gradient
    )

    assert np.linalg.norm(input_gradient[0][0] - expected_input_gradient[0][0]) < .001
