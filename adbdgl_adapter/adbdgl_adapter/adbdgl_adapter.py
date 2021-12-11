#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony Mahanna
"""

from .abc import ADBDGL_Adapter
from .adbdgl_controller import Base_ADBDGL_Controller

import dgl
from dgl.heterograph import DGLHeteroGraph

import torch
from collections import defaultdict
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
        self,
        name: str,
        graph_attributes: dict,
        extract_attributes=True,
        **query_options,
    ):
        self.__validate_attributes("graph", set(graph_attributes), self.GRAPH_ATRIBS)
        # if extract_attributes and type(self.__cntrl) == Base_ADBDGL_Controller:
        #     raise ValueError(
        #         f"Must implement custom ADBDGL_Controller if extract_attributes flag is enabled"
        #     )

        data_dict = {}

        ndata = defaultdict(lambda: defaultdict(list))  # wtf am i doing
        edata = defaultdict(lambda: defaultdict(list))  # wtf am i doing

        for v_col, atribs in graph_attributes["vertexCollections"].items():
            dgl_node_count: int = 0
            for v in self.__fetch_adb_docs(v_col, atribs, query_options):
                self.__cntrl.adb_map[v["_id"]] = {
                    "id": dgl_node_count,
                    "collection": v_col,
                }
                dgl_node_count += 1

                if extract_attributes:
                    for atrib in atribs:
                        if atrib not in v:
                            raise KeyError(f"{atrib} not in {v['_id']}")
                        ndata[atrib][v_col].append(
                            self.__cntrl.extract_dgl_atribute(v.get(atrib), atrib)
                        )

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

                if extract_attributes:
                    for atrib in atribs:
                        if atrib not in e:
                            raise KeyError(f"{atrib} not in {e['_id']}")
                        edata[atrib][e_col].append(
                            self.__cntrl.extract_dgl_atribute(e.get(atrib), atrib)
                        )

            data_dict[(from_col.pop(), e_col, to_col.pop())] = (
                torch.tensor(from_nodes),
                torch.tensor(to_nodes),
            )

        dgl_graph: DGLHeteroGraph = dgl.heterograph(data_dict)
        if extract_attributes:
            for key, col_dict in ndata.items():
                for col, val in col_dict.items():
                    dgl_graph.ndata[key] = {
                        **dgl_graph.ndata[key],
                        col: torch.tensor(val),
                    }
            for key, col_dict in edata.items():
                for col, val in col_dict.items():
                    dgl_graph.edata[key] = {
                        **dgl_graph.edata[key],
                        col: torch.tensor(val),
                    }

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
            name, graph_attributes, extract_attributes=False, **query_options
        )

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
