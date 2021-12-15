#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony Mahanna
"""

from abc import ABC


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

    def __prepare_dgl_features(self):
        raise NotImplementedError()  # pragma: no cover

    def __insert_dgl_features(self):
        raise NotImplementedError()  # pragma: no cover

    def __prepare_adb_attributes(self):
        raise NotImplementedError()  # pragma: no cover

    def __insert_adb_docs(self):
        raise NotImplemented()  # pragma: no cover

    def __fetch_adb_docs(self):
        raise NotImplementedError()  # pragma: no cover

    def __validate_attributes(self):
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
    def _adb_attribute_to_dgl_feature(self):
        raise NotImplementedError()  # pragma: no cover

    def _dgl_feature_to_adb_attribute(self):
        raise NotImplementedError()  # pragma: no cover
