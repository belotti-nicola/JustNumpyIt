from src.utils.csv import readCSV
from src.databases.abstract_database import absDatabase

class Xor(absDatabase):
    def __init__(self) -> None:
        self.test  = readCSV('src/databases/logic_functions/test.csv')
        self.train = readCSV('src/databases/logic_functions/train.csv')


    def getBatchTest(self, dim):
        pass
    def getBatchTest(self, dim):
        pass
    def getDatabaseTest(self):
        pass
    def getDatabaseTrain(self):
        pass
    
