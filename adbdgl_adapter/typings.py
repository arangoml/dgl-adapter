__all__ = [
    "Json",
    "ADBMetagraph",
    "ADBMetagraphValues",
    "DGLMetagraph",
    "DGLMetagraphValues",
    "DGLCanonicalEType",
    "DGLDataDict",
]

from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Union

from pandas import DataFrame
from torch import Tensor

Json = Dict[str, Any]

DataFrameToTensor = Callable[[DataFrame], Tensor]
TensorToDataFrame = Callable[[Tensor], DataFrame]

ADBEncoders = Dict[str, DataFrameToTensor]
ADBMetagraphValues = Union[str, DataFrameToTensor, ADBEncoders]
ADBMetagraph = Dict[str, Dict[str, Dict[str, ADBMetagraphValues]]]


DGLCanonicalEType = Tuple[str, str, str]
DGLDataDict = Dict[DGLCanonicalEType, Tuple[Tensor, Tensor]]
DGLNData = DefaultDict[str, DefaultDict[str, Tensor]]
DGLEData = DefaultDict[str, DefaultDict[DGLCanonicalEType, Tensor]]

DGLDataTypes = Union[str, DGLCanonicalEType]
DGLMetagraphValues = Union[str, List[str], TensorToDataFrame]
DGLMetagraph = Dict[str, Dict[DGLDataTypes, Dict[Any, DGLMetagraphValues]]]

ADBMap = DefaultDict[DGLDataTypes, Dict[str, int]]
DGLMap = DefaultDict[DGLDataTypes, Dict[int, str]]
