#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony Mahanna
"""

from abc import ABC
from os import pread


class ADBDGL_Adapter(ABC):
    def __init__(self):
        raise NotImplementedError()  # pragma: no cover

    def arangodb_to_dgl(self):
        raise NotImplementedError()  # pragma: no cover

    def arangodb_collections_to_dgl(self):
        raise NotImplementedError()  # pragma: no cover

    def arangodb_graph_to_dgl(self):
        raise NotImplementedError()  # pragma: no cover

    def dgl_to_arangodb(self):
        raise NotImplementedError()  # pragma: no cover

    def etypes_to_edefinitions(self):
        raise NotImplementedError()  # pragma: no cover

    def __validate_attributes(self):
        raise NotImplementedError()  # pragma: no cover

    def __fetch_adb_docs(self):
        raise NotImplementedError()  # pragma: no cover

    def __insert_adb_vertex(self):
        raise NotImplementedError()  # pragma: no cover

    def __insert_adb_edge(self):
        raise NotImplementedError()  # pragma: no cover

    def __prepare_dgl_features(self):
        raise NotImplementedError()  # pragma: no cover

    def __insert_dgl_features(self):
        raise NotImplementedError()  # pragma: no cover

    @property
    def DEFAULT_CANONICAL_ETYPE(self):
        return [("_N", "_E", "_N")]

    @property
    def CONNECTION_ATRIBS(self):
        return {"hostname", "username", "password", "dbName"}

    @property
    def METAGRAPH_ATRIBS(self):
        return {"vertexCollections", "edgeCollections"}

    @property
    def EDGE_DEFINITION_ATRIBS(self):
        return {"edge_collection", "from_vertex_collections", "to_vertex_collections"}


class ADBDGL_Controller(ABC):
    def __init__(self):
        raise NotImplementedError()  # pragma: no cover

    def _prepare_arangodb_vertex(self):
        raise NotImplementedError()  # pragma: no cover

    def _prepare_arangodb_edge(self):
        raise NotImplementedError()  # pragma: no cover

    def _identify_dgl_node(self):
        raise NotImplementedError()  # pragma: no cover

    def _identify_dgl_edge(self):
        raise NotImplementedError()  # pragma: no cover

    def _keyify_dgl_node(self):
        raise NotImplementedError()  # pragma: no cover

    def _keyify_dgl_edge(self):
        raise NotImplementedError()  # pragma: no cover

    @property
    def VALID_KEY_CHARS(self):
        return {
            "_",
            "-",
            ":",
            ".",
            "@",
            "(",
            ")",
            "+",
            ",",
            "=",
            ";",
            "$",
            "!",
            "*",
            "'",
            "%",
        }
