import pytest
from functools import reduce

from src.data.idx.idxhandler import IDX


testdata = [
    (IDX("data/MNIST/train-images.idx3-ubyte"),b'\x00\x00\x08\x03',None,3,[60000,28,28]),
    (IDX("data/MNIST/train-labels.idx1-ubyte"),b'\x00\x00\x08\x01',None,1,[60000]),
    (IDX("data/MNIST/t10k-images.idx3-ubyte" ),b'\x00\x00\x08\x03',None,3,[10000,28,28]),
    (IDX("data/MNIST/t10k-labels.idx1-ubyte" ),b'\x00\x00\x08\x01',None,1,[10000])
]

@pytest.mark.parametrize("instance,magicnumber,datatype,datadimensions,dimensions",testdata)
def test_MNIST_idx_objects(instance,magicnumber,datatype,datadimensions,dimensions):
    assert instance.magic_number == magicnumber
    assert instance.data_type == datatype
    assert instance.data_dimensions == datadimensions
    assert instance.dimensions == dimensions

    
    assert len(instance._data) == reduce((lambda x,y : x*y),instance.dimensions)

    assert instance.numpy().shape == tuple(instance.dimensions)

