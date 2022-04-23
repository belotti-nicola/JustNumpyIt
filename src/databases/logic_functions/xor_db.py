from src.utils.csv import readCSV
from src.databases.abstract_database import absDatabase

from random import shuffle

class Xor(absDatabase):
    def __init__(self) -> None:
        self.test  = readCSV('src/databases/logic_functions/test.csv')
        self.train = readCSV('src/databases/logic_functions/train.csv')
        self.test = random.shuffle()

    def getBatchTest(self, dim):
        pass
    def getBatchTest(self, dim):
        pass
    def getDatabaseTest(self):
        pass
    def getDatabaseTrain(self):
        pass
    
