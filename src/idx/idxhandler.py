from .typefactory import IDXTypeFactory

class MalformedIDX(Exception):
    pass

class IDX:
    magic_number = 0x0
    data_dimensions = -1
    data_type = None
    dimensions = []

    def __init__(self,pathname: str) -> None:
        self.dimensions = []      
        with open(pathname,"rb") as fr:
            byteslist = fr.read(4)
            if len(byteslist) < 4: 
                raise MalformedIDX("File dimension is less than 4 Bytes")
            else:
                self.magic_number = byteslist[::]
                self.type_selector = byteslist[2:3]
                self.data_dimensions = byteslist[3]
                self._fun = IDXTypeFactory.getType(self.type_selector)

                fr.seek(4)
                for i in range(self.data_dimensions):
                    dim = int.from_bytes(fr.read(4),'big')
                    self.dimensions.append(dim)
                    fr.seek(4)
                self._data = fr.read()


        
