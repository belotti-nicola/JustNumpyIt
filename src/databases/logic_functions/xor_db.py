from src.utils.csv import readCSV
from src.databases.abstract_database import absDatabase

import random

class Xor(absDatabase):
    def __init__(self) -> None:
        self.test  = readCSV('src/databases/logic_functions/test.csv')
        self.test = random.shuffle()
        self.ntest = len(self.train)

        self.train = readCSV('src/databases/logic_functions/train.csv')
        self.train = random.shuffle()
        self.ntrain = len(self.train)

    def getBatchTest(self, dim):
        if dim < self.ntest:
            tmp = random.shuffle()
            return tmp[0:dim]
        
    def getDatabaseTest(self):
        return self.test
    def getDatabaseTrain(self):
        return self.train
    
