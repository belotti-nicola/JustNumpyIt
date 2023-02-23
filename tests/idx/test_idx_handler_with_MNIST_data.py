import pytest

from src.idx.idxhandler import IDX


def test_MNIST_train_images():
    obj_to_test = IDX("data/MNIST/train-images.idx3-ubyte")
    assert obj_to_test.magic_number == b'\x00\x00\x08\x03'
    assert obj_to_test.data_type == None
    assert obj_to_test.data_dimension == 3
    assert obj_to_test.dimensions == [60000,28,28]
def test_MNIST_train_labels():
    obj_to_test = IDX("data/MNIST/train-labels.idx1-ubyte")
    assert obj_to_test.magic_number == b'\x00\x00\x08\x01'
    assert obj_to_test.data_type == None
    assert obj_to_test.data_dimension == 1
    assert obj_to_test.dimensions == [10000]

def test_MNIST_test_images():
    obj_to_test = IDX("data/MNIST/t10k-images.idx3-ubyte")
    assert obj_to_test.magic_number == b'\x00\x00\x08\x03'
    assert obj_to_test.data_type == None
    assert obj_to_test.data_dimension == 3
    assert obj_to_test.dimensions == [10000,28,28]
def test_MNIST_test_labels():
    obj_to_test = IDX("data/MNIST/train-labels.idx1-ubyte")
    assert obj_to_test.magic_number == b'\x00\x00\x08\x01'
    assert obj_to_test.data_type == None
    assert obj_to_test.data_dimension == 1
    assert obj_to_test.dimensions == [10000]



