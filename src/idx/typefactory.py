class IDXTypeFactory:
    @staticmethod 
    def getType(selector_byte: bytes):
        if selector_byte == b'\x08':
            return lambda x : int.from_bytes(x,'big')
        else :
            return None
