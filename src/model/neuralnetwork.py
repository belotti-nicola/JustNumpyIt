import numpy as np

from src.algorithms.factory import AlgorithmsFactory

class UntrainedModel(Exception):
    pass


class NeuralNetwork:

    def __init__(self,l:list) -> None:
        self._layers = l
        self._parameters = None

    def train(self,data,labels,iterations,alg_selector=0):
        solver = AlgorithmsFactory.get(alg_selector)


    def evaluate(self,a,b):
        if self._parameters == None:
            raise UntrainedModel()