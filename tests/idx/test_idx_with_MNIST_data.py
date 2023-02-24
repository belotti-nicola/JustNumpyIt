import pytest

from src.idx.idxhandler import IDX

testdata = [
    (IDX("data/MNIST/train-images.idx3-ubyte"),b'\x00\x00\x08\x03',None,3,[60000,28,28]),
    (IDX("data/MNIST/train-labels.idx1-ubyte"),b'\x00\x00\x08\x01',None,1,[60000]),
    (IDX("data/MNIST/t10k-images.idx3-ubyte" ),b'\x00\x00\x08\x03',None,3,[10000,28,28]),
    (IDX("data/MNIST/t10k-labels.idx1-ubyte" ),b'\x00\x00\x08\x01',None,1,[10000])
]

@pytest.mark.parametrize("instance,m,d,dd,ddd",testdata)
def test_MNIST_idx_objects(instance,m,d,dd,ddd):
    assert instance.magic_number == m
    assert instance.data_type == d
    assert instance.data_dimensions == dd
    assert instance.dimensions == ddd
'''def test_MNIST_train_labels():
    obj_to_test = IDX("data/MNIST/train-labels.idx1-ubyte")
    assert obj_to_test.magic_number == b'\x00\x00\x08\x01'
    assert obj_to_test.data_type == None
    assert obj_to_test.data_dimensions == 1
    assert obj_to_test.dimensions == [10000]

def test_MNIST_test_images():
    obj_to_test = IDX("data/MNIST/t10k-images.idx3-ubyte")
    assert obj_to_test.magic_number == b'\x00\x00\x08\x03'
    assert obj_to_test.data_type == None
    assert obj_to_test.data_dimensions == 3
    assert obj_to_test.dimensions == [10000,28,28]
def test_MNIST_test_labels():
    obj_to_test = IDX("data/MNIST/train-labels.idx1-ubyte")
    assert obj_to_test.magic_number == b'\x00\x00\x08\x01'
    assert obj_to_test.data_type == None
    assert obj_to_test.data_dimensions == 1
    assert obj_to_test.dimensions == [10000]



'''