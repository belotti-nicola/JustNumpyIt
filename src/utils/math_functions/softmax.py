import math

class SoftMax():
    @staticmethod
    def fun(x):
        sum_of_e_powers = sum(math.pow(math.e,x))
        return math.pow(math.e,x)/sum_of_e_powers
    def der(x):
        pass