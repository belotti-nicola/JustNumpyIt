def readCSV(file):
    from numpy import genfromtxt
    my_data = genfromtxt(file, delimiter=',')
    return my_data