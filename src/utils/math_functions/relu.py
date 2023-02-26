class ReLU():
    @staticmethod
    def fun(x):
        return max(x,0)
    
    @staticmethod
    def der(x):
        return 0 if x<= 0 else 1