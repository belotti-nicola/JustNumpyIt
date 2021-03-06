#from src.visualization.plotter import PlotterObj
from matplotlib.pyplot import imshow,plot,show
from src.utils.csv import readCSV
from src.visualization.plotter import PlotterObj

with open('src/visualization/example.csv') as f:
    imgs = readCSV(f)
    img1 = imgs[1][1:785]
    img1 = img1.reshape(28,28)
    img2 = imgs[5][1:785]
    img2 = img2.reshape(28,28)
    img3 = imgs[9][1:785]
    img3 = img3.reshape(28,28)
    img4 = imgs[8][1:785]
    img4 = img4.reshape(28,28)
    po = PlotterObj(2,2)
    po.addSubPlot(img1)
    po.addSubPlot(img2)
    po.addSubPlot(img3)
    po.addSubPlot(img4)

    po.show()