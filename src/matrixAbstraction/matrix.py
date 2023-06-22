from abc import ABC, abstractmethod

class ABCMatrix(ABC):

    '''
    Creational operations, here i use no self to point out that they 
    are class methods returning instances
    '''
    @abstractmethod
    def createRandomWeightsMatrix(r,c):
        pass

    @abstractmethod
    def createRandomBiasesMatrix(dim):
        pass

    @abstractmethod
    def createOnesMatrix(rows,columns):
        pass

    @abstractmethod
    def createZerosMatrix(rows,columns):
        pass
    

    '''
    Mathematical operations, self is not needed, but is used to 
    point out that these functions are between same class objects
    '''
    @abstractmethod    
    def __add__(self,other):
        pass
    
    @abstractmethod 
    def __sub__(self,other):
        pass

    @abstractmethod 
    def __mul__(self,other):
        pass

    @abstractmethod 
    def hadamardProduct(self,other):
        pass


