from __future__ import annotations
from abc import ABC, abstractmethod

class absDatabase(ABC):
    @abstractmethod
    def getDatabaseTrain(self):
        pass
    
    def getDatabaseTest(self):
        pass

    def getBatchTest(self,dim):
        pass

    def getBatchTrain(self,dim):
        pass

