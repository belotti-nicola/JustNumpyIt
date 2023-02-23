class MalformedIDX(Exception):
    pass

class IDX:
    magic_number = 0x0
    data_dimension = -1
    data_type = None
    dimensions = []

    def __init__(self,pathname: str) -> None:      
        with open(pathname,"rb") as fr:
            byteslist = fr.read(4)
            if len(byteslist) < 4: 
                raise MalformedIDX("File dimension is less than 4 Bytes")
            else:
                self.magic_number = byteslist[::]
                type_selector = byteslist[2]
                self.data_dimension = byteslist[3]
                fr.seek(4)
                rest_of_the_file = fr.read()


        
obj_to_test = IDX("data/MNIST/train-images.idx3-ubyte")
print(obj_to_test.magic_number)