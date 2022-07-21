__all__ = ["Json", "ArangoMetagraph", "DGLCanonicalEType"]

from typing import Any, Dict, Set, Tuple

from torch import Tensor

Json = Dict[str, Any]
ArangoMetagraph = Dict[str, Dict[str, Set[str]]]


DGLCanonicalEType = Tuple[str, str, str]
DGLDataDict = Dict[DGLCanonicalEType, Tuple[Tensor, Tensor]]
