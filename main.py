from src.databases.MNIST.mnist_db import MNIST

db = MNIST()    
print(len(db.test))
print(len(db.train))