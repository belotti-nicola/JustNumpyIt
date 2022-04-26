import matplotlib.pyplot as plt
import numpy as np

'''
w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
'''
from math import floor

class PlotterObj:

    def __init__(self,nx,ny) -> None:
        self.dimX = nx
        self.dimY = ny
        self.count = 0
        self.fig, self.axs = plt.subplots(nx, ny)
    
    def addSubPlot(self,M:np):
        x = floor(self.count/self.dimX)
        y = self.count%self.dimX
        self.axs[x,y].imshow(M)
        self.count += 1

    def show(self):
        plt.show()