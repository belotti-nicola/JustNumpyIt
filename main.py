import struct 

'''with open("data/MNIST/t10k-images.idx3-ubyte","rb") as f:
    bytes = f.read(8)

    magic, size = struct.unpack(">II", bytes)

print(magic) # 2049
print(size)  # 60000
'''

with open("data/MNIST/train-images.idx3-ubyte","rb") as f:
    somelines = f.read(16)
    print(somelines[0],somelines[1],somelines[2],somelines[3])
    print(somelines[4],somelines[5],somelines[6],somelines[7])
    print(somelines[8],somelines[9],somelines[10],somelines[11])
    print(somelines[12],somelines[13],somelines[14],somelines[15])
