#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony Mahanna
"""

from .abc import ADBDGL_Adapter
from .adbdgl_controller import Base_ADBDGL_Controller

from arango import ArangoClient
from collections import defaultdict

import dgl
import torch

from typing import Any, Union
from dgl.heterograph import DGLHeteroGraph
from dgl.view import HeteroEdgeDataView, HeteroNodeDataView


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
            dgl_node_count: int = 0
            for v in self.__fetch_adb_docs(v_col, atribs, query_options):
                self.__cntrl.adb_map[v["_id"]] = {
                    "id": dgl_node_count,
                    "collection": v_col,
                }
                dgl_node_count += 1

                self.__prepare_dgl_features(v_col, atribs, v, ndata)

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

                self.__prepare_dgl_features(e_col, atribs, e, edata)

            data_dict[(from_col.pop(), e_col, to_col.pop())] = (
                torch.tensor(from_nodes),
                torch.tensor(to_nodes),
            )

        dgl_graph: DGLHeteroGraph = dgl.heterograph(data_dict)
        self.__insert_dgl_features(ndata, dgl_graph.ndata)
        self.__insert_dgl_features(edata, dgl_graph.edata)

        print(f"DGL: {name} created")
        return dgl_graph

    def __insert_dgl_features(
        self,
        features: defaultdict[Any, defaultdict[Any, list]],
        data: Union[HeteroNodeDataView, HeteroEdgeDataView],
    ):
        for key, col_dict in features.items():
            for col, array in col_dict.items():
                data[key] = {**data[key], col: torch.tensor(array)}

    def __prepare_dgl_features(
        self,
        col: str,
        attributes: set,
        doc: dict,
        features: defaultdict[Any, defaultdict[Any, list]],
    ):
        for a in attributes:
            if a not in doc:
                raise KeyError(f"{a} not in {doc['_id']}")
            features[a][col].append(self.__cntrl.attribute_to_feature(col, a, doc[a]))

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
