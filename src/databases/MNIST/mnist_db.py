from src.databases.abstract_database import absDatabase
from src.utils.csv import readCSV

class MNIST(absDatabase):
    def __init__(self) -> None:
        self.test  = readCSV('src/databases/MNIST/mnist_test.csv')
        self.train = readCSV('src/databases/MNIST/mnist_train.csv')


    def getBatchTest(self, dim):
        pass
    def getBatchTest(self, dim):
        pass
    def getDatabaseTest(self):
        pass
    def getDatabaseTrain(self):
        pass
    
