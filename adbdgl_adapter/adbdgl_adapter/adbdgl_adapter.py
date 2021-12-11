#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony Mahanna
"""

from dgl.heterograph import DGLHeteroGraph
from .abc import ADBDGL_Adapter
from .adbdgl_controller import Base_ADBDGL_Controller

import dgl
import torch
from arango import ArangoClient


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
        self, name: str, graph_attributes: dict, is_keep=True, **query_options
    ):
        self.__validate_attributes("graph", set(graph_attributes), self.GRAPH_ATRIBS)
        data_dict = {}

        for v_col, atribs in graph_attributes["vertexCollections"].items():
            dgl_node_count: int = 0
            for v in self.__fetch_adb_docs(v_col, atribs, is_keep, query_options):
                self.__cntrl.adb_map[v["_id"]] = {
                    "id": dgl_node_count,
                    "collection": v_col,
                }
                dgl_node_count += 1

        for e_col, atribs in graph_attributes["edgeCollections"].items():
            src_nodes = []
            dst_nodes = []
            for e in self.__fetch_adb_docs(e_col, atribs, is_keep, query_options):
                src_nodes.append(self.__cntrl.adb_map[e["_from"]]["id"])
                dst_nodes.append(self.__cntrl.adb_map[e["_to"]]["id"])

            src_collection = self.__cntrl.adb_map[e["_from"]]["collection"]
            dst_collection = self.__cntrl.adb_map[e["_to"]]["collection"]
            data_dict[(src_collection, e_col, dst_collection)] = (
                torch.tensor(src_nodes),
                torch.tensor(dst_nodes),
            )

        dgl_graph: DGLHeteroGraph = dgl.heterograph(data_dict)
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

        return self.arangodb_to_dgl(
            name, graph_attributes, is_keep=False, **query_options
        )

    def arangodb_graph_to_dgl(self, name: str, **query_options):
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_dgl(name, v_cols, e_cols, **query_options)

    def __fetch_adb_docs(
        self, col: str, attributes: set, is_keep: bool, query_options: dict
    ):
        """Fetches ArangoDB documents within a collection.

        :param col: The ArangoDB collection.
        :type col: str
        :param attributes: The set of document attributes.
        :type attributes: set
        :param is_keep: Only keep the document attributes specified in **attributes** when returning the document. Otherwise, all document attributes are included.
        :type is_keep: bool
        :param query_options: Keyword arguments to specify AQL query options when fetching documents from the ArangoDB instance.
        :type query_options: **kwargs
        :return: Result cursor.
        :rtype: arango.cursor.Cursor
        """
        aql = f"""
            FOR doc IN {col}
                RETURN {is_keep} ? 
                    MERGE(KEEP(doc, {list(attributes)}), {{"_id": doc._id}}) : doc
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
