#from src.visualization.plotter import PlotterObj
from matplotlib.pyplot import imshow,plot
from src.utils.csv import readCSV

with open('src/visualization/example.csv') as f:
    imgs = readCSV(f)
    img = imgs[0][1:785]
    img = img.reshape(28,28)
    img = imshow(img[0],cmap='gray',vmin=0, vmax=255)
    imgplot = plot(img)