from src.matrixAbstraction.matrix import ABCMatrix
import numpy as np


class NumpyWrapper(ABCMatrix):
    def __init__(self,rows,columns) -> None:
        self._data = np.empty((rows,columns))
    
    @staticmethod
    def createRandomWeightsMatrix(rows,columns):
        pass

    @staticmethod
    def createRandomBiasesMatrix(rows,columns):
        pass
    
    def __add__(self, obj):
        data = self._data + obj._data
        retWrapp = NumpyWrapper(data.shape[0],data.shape[1])
        retWrapp._data = data
        return retWrapp
    
    def __sub__(self, obj):
        if isinstance(obj, NumpyWrapper): #TODO check always
            return self._data - obj._data
        else:
            raise(Exception("value error:"))

    def __mul__(self, obj):
        data = np.dot(self._data,obj._data)
        retWrapp = NumpyWrapper(obj._data.shape[0],obj._data.shape[1])
        retWrapp._data = data
        return retWrapp
    
    def hadamardProduct(self, other):
        return np.array(self._data) * np.array(other._data)
    
    def __str__(self):
        return str(self._data)
    
    @staticmethod
    def createZerosMatrix(rows, columns):
        obj = NumpyWrapper(rows,columns)
        obj._data = np.zeros(shape=(rows,columns))
        return obj
    @staticmethod
    def createOnesMatrix(rows, columns):
        obj = NumpyWrapper(rows,columns)
        obj._data = np.ones(shape=(rows,columns))
        return obj