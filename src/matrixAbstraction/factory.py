from src.matrixAbstraction.concreteObjects.concreteNumpy import NumpyWrapper

class UnsupportedMatrixFormat(Exception):
    def __init__(self, format) -> None:
        super().__init__(format+" is not supported.")

class MatrixFactory():
    def __init__(self) -> None:
        self._supportedFormats = {
            "numpy":NumpyWrapper
            }

    def register_format(self, format, creator):
        self._supportedFormats[format] = creator

    def get_creator(self, format):
        creator = self._supportedFormats.get(format)
        if not creator:
            raise UnsupportedMatrixFormat(format)
        return creator