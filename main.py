import struct 

'''with open("data/MNIST/t10k-images.idx3-ubyte","rb") as f:
    bytes = f.read(8)

    magic, size = struct.unpack(">II", bytes)

print(magic) # 2049
print(size)  # 60000
'''

with open("data/MNIST/train-images.idx3-ubyte","rb") as f:
    somelines = f.read(16)
    str_out = ""
    i = 1
    for line in somelines:
        str_out += str(line) + ("" if i%4 else "\n")
        i+=1

    print(str_out)

