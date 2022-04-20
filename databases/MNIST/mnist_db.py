from  src.databases.abstract_database import absDatabase
from utils.csv import readCSV
from pathlib import Path

class MNIST(absDatabase):
    def __init__(self) -> None:
        self.test = readCSV(Path('databases/MNIST/mnist_test.csv'))
        self.train = readCSV(Path('databases/MNIST/mnist_train.csv'))


    def getBatchTest(self, dim):
        pass
    def getBatchTest(self, dim):
        pass
    def getDatabaseTest(self):
        pass
    def getDatabaseTrain(self):
        pass
    
db = MNIST()    
print(len(db.test))
print(len(db.train))