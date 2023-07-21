__all__ = [
    "Json",
    "ADBMetagraph",
    "ADBMetagraphValues",
    "DGLMetagraph",
    "DGLMetagraphValues",
    "DGLCanonicalEType",
    "DGLDataDict",
    "ADBMap",
    "DGLMap",
]

from typing import Any, Callable, DefaultDict, Dict, List, Set, Tuple, Union

from pandas import DataFrame
from torch import Tensor

Json = Dict[str, Any]

DataFrameToTensor = Callable[[DataFrame], Tensor]
TensorToDataFrame = Callable[[Tensor, DataFrame], DataFrame]

ADBEncoders = Dict[str, DataFrameToTensor]
ADBMetagraphValues = Union[str, DataFrameToTensor, ADBEncoders]
ADBMetagraph = Dict[str, Dict[str, Union[Set[str], Dict[str, ADBMetagraphValues]]]]


DGLCanonicalEType = Tuple[str, str, str]
DGLData = DefaultDict[str, DefaultDict[Union[str, DGLCanonicalEType], Tensor]]
DGLDataDict = Dict[DGLCanonicalEType, Tuple[Tensor, Tensor]]

DGLDataTypes = Union[str, DGLCanonicalEType]
DGLMetagraphValues = Union[str, List[str], TensorToDataFrame]
DGLMetagraph = Dict[
    str, Dict[DGLDataTypes, Union[Set[str], Dict[Any, DGLMetagraphValues]]]
]

ADBMap = DefaultDict[DGLDataTypes, Dict[str, int]]
DGLMap = DefaultDict[DGLDataTypes, Dict[int, str]]
