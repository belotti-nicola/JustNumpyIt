from .gradient_descent import GD
from .stocasthic_gradient_descent import SGD

class UnsupportedAlgorithm(Exception):
    pass


class AlgorithmsFactory():
    @staticmethod
    def get(selector:int):
        if selector == 0:
            return GD()
        elif selector == 1:
            return SGD()
        else: 
            raise UnsupportedAlgorithm(str(selector))

     