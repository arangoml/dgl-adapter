#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Any, List, Set, Union

from arango.graph import Graph as ArangoDBGraph
from dgl import DGLGraph
from dgl.heterograph import DGLHeteroGraph
from torch.functional import Tensor

from .typings import ArangoMetagraph, DGLCanonicalEType, Json


class Abstract_ADBDGL_Adapter(ABC):
    def __init__(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def arangodb_to_dgl(
        self, name: str, metagraph: ArangoMetagraph, **query_options: Any
    ) -> DGLHeteroGraph:
        raise NotImplementedError  # pragma: no cover

    def arangodb_collections_to_dgl(
        self, name: str, v_cols: Set[str], e_cols: Set[str], **query_options: Any
    ) -> DGLHeteroGraph:
        raise NotImplementedError  # pragma: no cover

    def arangodb_graph_to_dgl(self, name: str, **query_options: Any) -> DGLHeteroGraph:
        raise NotImplementedError  # pragma: no cover

    def dgl_to_arangodb(
        self,
        name: str,
        dgl_g: Union[DGLGraph, DGLHeteroGraph],
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ArangoDBGraph:
        raise NotImplementedError  # pragma: no cover

    def etypes_to_edefinitions(
        self, canonical_etypes: List[DGLCanonicalEType]
    ) -> List[Json]:
        raise NotImplementedError  # pragma: no cover

    def __prepare_dgl_features(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def __insert_dgl_features(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def __prepare_adb_attributes(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def __fetch_adb_docs(self) -> None:
        raise NotImplementedError  # pragma: no cover

    @property
    def DEFAULT_CANONICAL_ETYPE(self) -> List[DGLCanonicalEType]:
        return [("_N", "_E", "_N")]


class Abstract_ADBDGL_Controller(ABC):
    def _adb_attribute_to_dgl_feature(self, key: str, col: str, val: Any) -> Any:
        raise NotImplementedError  # pragma: no cover

    def _dgl_feature_to_adb_attribute(self, key: str, col: str, val: Tensor) -> Any:
        raise NotImplementedError  # pragma: no cover
