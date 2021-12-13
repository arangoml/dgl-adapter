#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony Mahanna
"""

from torch._C import StringType, Value
from torch.functional import Tensor
from .abc import ADBDGL_Adapter
from .adbdgl_controller import Base_ADBDGL_Controller

from arango import ArangoClient
from collections import defaultdict

import dgl
import torch

from typing import Any, Tuple, Union
from dgl.heterograph import DGLHeteroGraph
from dgl.view import HeteroEdgeDataView, HeteroNodeDataView

from dgl import DGLGraph
from arango.graph import Graph as ArangoDBGraph


class ArangoDB_DGL_Adapter(ADBDGL_Adapter):
    def __init__(
        self,
        conn: dict,
        controller_class: Base_ADBDGL_Controller = Base_ADBDGL_Controller,
    ):
        self.__validate_attributes("connection", set(conn), self.CONNECTION_ATRIBS)
        if issubclass(controller_class, Base_ADBDGL_Controller) is False:
            msg = "controller_class must inherit from Base_ADBDGL_Controller"
            raise TypeError(msg)

        username = conn["username"]
        password = conn["password"]
        db_name = conn["dbName"]

        protocol = conn.get("protocol", "https")
        host = conn["hostname"]
        port = str(conn.get("port", 8529))

        url = protocol + "://" + host + ":" + port
        print(f"Connecting to {url}")
        self.__db = ArangoClient(hosts=url).db(db_name, username, password, verify=True)
        self.__cntrl: Base_ADBDGL_Controller = controller_class()

    def arangodb_to_dgl(
        self,
        name: str,
        graph_attributes: dict,
        **query_options,
    ):
        self.__validate_attributes("graph", set(graph_attributes), self.GRAPH_ATRIBS)

        data_dict = {}
        ndata = defaultdict(lambda: defaultdict(list))
        edata = defaultdict(lambda: defaultdict(list))

        for v_col, atribs in graph_attributes["vertexCollections"].items():
            node_id = 0
            for v in self.__fetch_adb_docs(v_col, atribs, query_options):
                self.__cntrl.adb_map[v["_id"]] = {
                    "id": node_id,
                    "collection": v_col,
                }
                node_id += 1

                self.__prepare_dgl_features(ndata, v_col, atribs, v)

        from_col = set()
        to_col = set()
        for e_col, atribs in graph_attributes["edgeCollections"].items():
            from_nodes = []
            to_nodes = []
            for e in self.__fetch_adb_docs(e_col, atribs, query_options):
                from_node = self.__cntrl.adb_map[e["_from"]]
                to_node = self.__cntrl.adb_map[e["_to"]]

                from_col.add(from_node["collection"])
                to_col.add(to_node["collection"])
                if len(from_col) > 1 or len(to_col) > 1:
                    raise ValueError(f"too many '_from' & '_to' collections in {e_col}")

                from_nodes.append(from_node["id"])
                to_nodes.append(to_node["id"])

                self.__prepare_dgl_features(edata, e_col, atribs, e)

            data_dict[(from_col.pop(), e_col, to_col.pop())] = (
                torch.tensor(from_nodes),
                torch.tensor(to_nodes),
            )

        dgl_graph: DGLHeteroGraph = dgl.heterograph(data_dict)
        self.__insert_dgl_features(ndata, dgl_graph.ndata)
        self.__insert_dgl_features(edata, dgl_graph.edata)

        print(f"DGL: {name} created")
        return dgl_graph

    def arangodb_collections_to_dgl(
        self,
        name: str,
        vertex_collections: set,
        edge_collections: set,
        **query_options,
    ):
        graph_attributes = {
            "vertexCollections": {col: {} for col in vertex_collections},
            "edgeCollections": {col: {} for col in edge_collections},
        }

        return self.arangodb_to_dgl(name, graph_attributes, **query_options)

    def arangodb_graph_to_dgl(self, name: str, **query_options):
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_dgl(name, v_cols, e_cols, **query_options)

    def dgl_to_arangodb(self, name: str, dgl_g: Union[DGLGraph, DGLHeteroGraph]):
        adb_v_cols = set()
        adb_e_cols = set()
        e_definitions = []

        is_dgl_data = dgl_g.canonical_etypes == self.DEFAULT_CANONICAL_ETYPE
        if is_dgl_data:
            e_col = name + "_" + self.DEFAULT_ETYPE
            adb_e_cols.add(e_col)
            from_col = to_col = name + "_" + self.DEFAULT_NTYPE
            adb_v_cols.add(from_col)
            e_definitions = self.etypes_to_edefinitions([(from_col, e_col, to_col)])
        else:
            adb_v_cols = set(dgl_g.ntypes)
            adb_e_cols = set(dgl_g.etypes)
            e_definitions = self.etypes_to_edefinitions(dgl_g.canonical_etypes)

        adb_documents = defaultdict(list)
        for v_col in adb_v_cols:
            if self.__db.has_collection(v_col) is False:
                self.__db.create_collection(v_col)

            node: Tensor
            for node in dgl_g.nodes(None if is_dgl_data else v_col):
                id: int = node.item()
                adb_documents[v_col].append({"_key": str(id)})

        for e_col in adb_e_cols:
            if self.__db.has_collection(e_col) is False:
                self.__db.create_collection(e_col, edge=True)

            from_nodes: Tensor
            to_nodes: Tensor
            from_nodes, to_nodes = dgl_g.edges(etype=None if is_dgl_data else e_col)

            if is_dgl_data is False:
                from_col, _, to_col = dgl_g.to_canonical_etype(e_col)

            for from_node, to_node in zip(from_nodes, to_nodes):
                adb_documents[e_col].append(
                    {
                        "_from": f"{from_col}/{str(from_node.item())}",
                        "_to": f"{to_col}/{str(to_node.item())}",
                    }
                )

        self.__db.delete_graph(name, ignore_missing=True)
        adb_graph: ArangoDBGraph = self.__db.create_graph(name, e_definitions)

        for col, doc_list in adb_documents.items():  # insert remaining documents
            self.__db.collection(col).import_bulk(doc_list, on_duplicate="replace")

        print(f"ArangoDB: {name} created")
        return adb_graph

    def etypes_to_edefinitions(self, canonical_etypes: list) -> list:
        edge_definitions = []
        for dgl_from, dgl_e, dgl_to in canonical_etypes:
            edge_definitions.append(
                {
                    "from_vertex_collections": [dgl_from],
                    "edge_collection": dgl_e,
                    "to_vertex_collections": [dgl_to],
                }
            )

        return edge_definitions

    def __insert_dgl_features(
        self,
        features: defaultdict,
        data: Union[HeteroNodeDataView, HeteroEdgeDataView],
    ):
        for key, col_dict in features.items():
            for col, array in col_dict.items():
                data[key] = {**data[key], col: torch.tensor(array)}

    def __prepare_dgl_features(
        self,
        features: defaultdict,
        col: str,
        attributes: set,
        doc: dict,
    ):
        for a in attributes:
            if a not in doc:
                raise KeyError(f"{a} not in {doc['_id']}")
            array: list = features[a][col]
            array.append(self.__cntrl.adb_attribute_to_dgl_feature(col, a, doc[a]))

    def __fetch_adb_docs(self, col: str, attributes: set, query_options: dict):
        """Fetches ArangoDB documents within a collection.

        :param col: The ArangoDB collection.
        :type col: str
        :param attributes: The set of document attributes.
        :type attributes: set
        :param query_options: Keyword arguments to specify AQL query options when fetching documents from the ArangoDB instance.
        :type query_options: **kwargs
        :return: Result cursor.
        :rtype: arango.cursor.Cursor
        """
        aql = f"""
            FOR doc IN {col}
                RETURN MERGE(
                    KEEP(doc, {list(attributes)}), 
                    {{"_id": doc._id, "_from": doc._from, "_to": doc._to}}
                )
        """

        return self.__db.aql.execute(aql, **query_options)

    def __validate_attributes(self, type: str, attributes: set, valid_attributes: set):
        """Validates that a set of attributes includes the required valid attributes.

        :param type: The context of the attribute validation (e.g connection attributes, graph attributes, etc).
        :type type: str
        :param attributes: The provided attributes, possibly invalid.
        :type attributes: set
        :param valid_attributes: The valid attributes.
        :type valid_attributes: set
        :raise ValueError: If **valid_attributes** is not a subset of **attributes**
        """
        if valid_attributes.issubset(attributes) is False:
            missing_attributes = valid_attributes - attributes
            raise ValueError(f"Missing {type} attributes: {missing_attributes}")
