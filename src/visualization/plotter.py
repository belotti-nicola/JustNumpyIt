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

class PlotterObj:

    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y
        self.fig, self.axs = plt.subplots(2, 2)
    
    def addSubPlot(self,M:np):
        self.axs[0, 0].plot(x, y)
        
for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

    def show():