#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Any, Set, Union

from arango.graph import Graph as ArangoDBGraph
from dgl import DGLGraph, DGLHeteroGraph

from .typings import ADBMetagraph, DGLCanonicalEType, DGLMetagraph, Json


class Abstract_ADBDGL_Adapter(ABC):
    def __init__(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def arangodb_to_dgl(
        self, name: str, metagraph: ADBMetagraph, **query_options: Any
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
        metagraph: DGLMetagraph = {},
        explicit_metagraph: bool = True,
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ArangoDBGraph:
        raise NotImplementedError  # pragma: no cover


class Abstract_ADBDGL_Controller(ABC):
    def _prepare_dgl_node(self, dgl_node: Json, node_type: str) -> Json:
        raise NotImplementedError  # pragma: no cover

    def _prepare_dgl_edge(self, dgl_edge: Json, edge_type: DGLCanonicalEType) -> Json:
        raise NotImplementedError  # pragma: no cover
