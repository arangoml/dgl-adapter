#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony Mahanna
"""

from .abc import ADBDGL_Adapter
from .adbdgl_controller import Base_ADBDGL_Controller

from arango import ArangoClient
from arango.graph import Graph as ArangoDBGraph

import dgl
from dgl import DGLGraph
from dgl.heterograph import DGLHeteroGraph
from dgl.view import HeteroEdgeDataView, HeteroNodeDataView

import torch
from torch.functional import Tensor

from typing import Union
from collections import defaultdict


class ArangoDB_DGL_Adapter(ADBDGL_Adapter):
    """ArangoDB-DGL adapter.

    :param conn: Connection details to an ArangoDB instance.
    :type conn: dict
    :param controller_class: The ArangoDB-DGL controller, for controlling how ArangoDB attributes are converted into DGL features, and vice-versa. Optionally re-defined by the user if needed (otherwise defaults to Base_ADBDGL_Controller).
    :type controller_class: Base_ADBDGL_Controller
    :raise ValueError: If missing required keys in conn
    """

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
        metagraph: dict,
        **query_options,
    ):
        """Create a DGL graph from user-defined metagraph.

        :param name: The DGL graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import to DGL, along with their associated attributes to keep.
        :type metagraph: dict
        :param query_options: Keyword arguments to specify AQL query options when fetching documents from the ArangoDB instance.
        :type query_options: **kwargs
        :return: A DGL Heterograph
        :rtype: dgl.heterograph.DGLHeteroGraph
        :raise ValueError: If missing required keys in metagraph

        Here is an example entry for parameter **metagraph**:

        .. code-block:: python
        {
            "vertexCollections": {
                "account": {"Balance", "account_type", "customer_id", "rank"},
                "bank": {"Country", "Id", "bank_id", "bank_name"},
                "customer": {"Name", "Sex", "Ssn", "rank"},
            },
            "edgeCollections": {
                "accountHolder": {},
                "transaction": {},
            },
        }
        """
        self.__validate_attributes("graph", set(metagraph), self.METAGRAPH_ATRIBS)

        adb_map = dict()  # Maps ArangoDB vertex IDs to DGL node IDs

        # Dictionaries for constructing a heterogeneous graph.
        data_dict = dict()
        ndata = defaultdict(lambda: defaultdict(list))
        edata = defaultdict(lambda: defaultdict(list))

        for v_col, atribs in metagraph["vertexCollections"].items():
            for i, v in enumerate(self.__fetch_adb_docs(v_col, atribs, query_options)):
                adb_map[v["_id"]] = {
                    "id": i,
                    "col": v_col,
                }

                self.__prepare_dgl_features(ndata, atribs, v, v_col)

        from_col = set()
        to_col = set()
        for e_col, atribs in metagraph["edgeCollections"].items():
            from_nodes = []
            to_nodes = []
            for e in self.__fetch_adb_docs(e_col, atribs, query_options):
                from_node = adb_map[e["_from"]]
                to_node = adb_map[e["_to"]]

                from_col.add(from_node["col"])
                to_col.add(to_node["col"])
                if len(from_col | to_col) > 2:
                    raise ValueError(
                        f"Can't convert to DGL: too many '_from' & '_to' collections in {e_col}"
                    )

                from_nodes.append(from_node["id"])
                to_nodes.append(to_node["id"])

                self.__prepare_dgl_features(edata, atribs, e, e_col)

            data_dict[(from_col.pop(), e_col, to_col.pop())] = (
                torch.tensor(from_nodes),
                torch.tensor(to_nodes),
            )

        dgl_g: DGLHeteroGraph = dgl.heterograph(data_dict)
        has_one_ntype = len(dgl_g.ntypes) == 1
        has_one_etype = len(dgl_g.etypes) == 1

        self.__insert_dgl_features(ndata, dgl_g.ndata, has_one_ntype)
        self.__insert_dgl_features(edata, dgl_g.edata, has_one_etype)

        print(f"DGL: {name} created")
        return dgl_g

    def arangodb_collections_to_dgl(
        self,
        name: str,
        vertex_collections: set,
        edge_collections: set,
        **query_options,
    ):
        """Create a DGL graph from ArangoDB collections.

        :param name: The DGL graph name.
        :type name: str
        :param vertex_collections: A set of ArangoDB vertex collections to import to DGL.
        :type vertex_collections: set
        :param edge_collections: A set of ArangoDB edge collections to import to DGL.
        :type edge_collections: set
        :param query_options: Keyword arguments to specify AQL query options when fetching documents from the ArangoDB instance.
        :type query_options: **kwargs
        :return: A DGL Heterograph
        :rtype: dgl.heterograph.DGLHeteroGraph
        """
        metagraph = {
            "vertexCollections": {col: {} for col in vertex_collections},
            "edgeCollections": {col: {} for col in edge_collections},
        }

        return self.arangodb_to_dgl(name, metagraph, **query_options)

    def arangodb_graph_to_dgl(self, name: str, **query_options):
        """Create a DGL graph from an ArangoDB graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param query_options: Keyword arguments to specify AQL query options when fetching documents from the ArangoDB instance.
        :type query_options: **kwargs
        :return: A DGL Heterograph
        :rtype: dgl.heterograph.DGLHeteroGraph
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_dgl(name, v_cols, e_cols, **query_options)

    def dgl_to_arangodb(
        self, name: str, dgl_g: Union[DGLGraph, DGLHeteroGraph], batch_size: int = 1000
    ):
        """Create an ArangoDB graph from a DGL graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param dgl_g: The existing DGL graph.
        :type dgl_g: Union[dgl.DGLGraph, dgl.heterograph.DGLHeteroGraph]
        :param batch_size: The maximum number of documents to insert at once
        :type batch_size: int
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        """
        is_dgl_data = dgl_g.canonical_etypes == self.DEFAULT_CANONICAL_ETYPE
        adb_v_cols = [name + dgl_g.ntypes[0]] if is_dgl_data else dgl_g.ntypes
        adb_e_cols = [name + dgl_g.etypes[0]] if is_dgl_data else dgl_g.etypes
        e_definitions = self.etypes_to_edefinitions(
            [
                (
                    adb_v_cols[0],
                    adb_e_cols[0],
                    adb_v_cols[0],
                )
            ]
            if is_dgl_data
            else dgl_g.canonical_etypes
        )

        has_one_ntype = len(dgl_g.ntypes) == 1
        has_one_etype = len(dgl_g.etypes) == 1

        adb_documents = defaultdict(list)
        for v_col in adb_v_cols:
            v_col_docs = adb_documents[v_col]

            if self.__db.has_collection(v_col) is False:
                self.__db.create_collection(v_col)

            node: Tensor
            for node in dgl_g.nodes(None if is_dgl_data else v_col):
                dgl_node_id: int = node.item()
                vertex = {"_key": str(dgl_node_id)}
                self.__prepare_adb_attributes(
                    dgl_g.ndata,
                    dgl_g.node_attr_schemes(None if is_dgl_data else v_col).keys(),
                    dgl_node_id,
                    vertex,
                    v_col,
                    has_one_ntype,
                )
                v_col_docs.append(vertex)

                if len(v_col_docs) >= batch_size:
                    self.__db.collection(v_col).import_bulk(
                        v_col_docs, on_duplicate="replace"
                    )
                    v_col_docs.clear()

        from_col: str
        to_col: str
        from_nodes: Tensor
        to_nodes: Tensor
        for e_col in adb_e_cols:
            e_col_docs = adb_documents[e_col]

            if self.__db.has_collection(e_col) is False:
                self.__db.create_collection(e_col, edge=True)

            if is_dgl_data:
                from_col = to_col = adb_v_cols[0]
            else:
                from_col, _, to_col = dgl_g.to_canonical_etype(e_col)

            from_nodes, to_nodes = dgl_g.edges(etype=None if is_dgl_data else e_col)
            for dgl_edge_id, (from_node, to_node) in enumerate(
                zip(from_nodes, to_nodes)
            ):
                edge = {
                    "_key": str(dgl_edge_id),
                    "_from": f"{from_col}/{str(from_node.item())}",
                    "_to": f"{to_col}/{str(to_node.item())}",
                }
                self.__prepare_adb_attributes(
                    dgl_g.edata,
                    dgl_g.edge_attr_schemes(None if is_dgl_data else e_col).keys(),
                    dgl_edge_id,
                    edge,
                    e_col,
                    has_one_etype,
                )
                e_col_docs.append(edge)

                if len(e_col_docs) >= batch_size:
                    self.__db.collection(e_col).import_bulk(
                        e_col_docs, on_duplicate="replace"
                    )
                    e_col_docs.clear()

        self.__db.delete_graph(name, ignore_missing=True)
        adb_graph: ArangoDBGraph = self.__db.create_graph(name, e_definitions)

        for col, doc_list in adb_documents.items():  # insert remaining documents
            self.__db.collection(col).import_bulk(doc_list, on_duplicate="replace")

        print(f"ArangoDB: {name} created")
        return adb_graph

    def etypes_to_edefinitions(self, canonical_etypes: list) -> list:
        """Converts a DGL graph's canonical_etypes property to ArangoDB graph edge definitions

        :param canonical_etypes: A list of string triplets (str, str, str) for source node type, edge type and destination node type.
        :type canonical_etypes: list[tuple]
        :return: ArangoDB Edge Definitions
        :rtype: list[dict[str, Union[str, list[str]]]]

        Here is an example of **edge_definitions**:

        .. code-block:: python
        [
            {
                "edge_collection": "teach",
                "from_vertex_collections": ["teachers"],
                "to_vertex_collections": ["lectures"]
            }
        ]
        """
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

    def __prepare_dgl_features(
        self,
        features_data: defaultdict,
        attributes: set,
        doc: dict,
        col: str,
    ):
        """Convert a set of ArangoDB attributes into valid DGL features

        :param features_data: A dictionary storing the DGL features formatted as lists.
        :type features_data: defaultdict[Any, defaultdict[Any, list]]
        :param col: The collection the current document belongs to
        :type col: str
        :param attributes: A set of ArangoDB attribute keys to convert into DGL features
        :type attributes: set
        :param doc: The current ArangoDB document
        :type doc: dict

        """
        key: str
        for key in attributes:
            arr: list = features_data[key][col]
            arr.append(self.__cntrl._adb_attribute_to_dgl_feature(key, col, doc[key]))

    def __insert_dgl_features(
        self,
        features_data: defaultdict,
        data: Union[HeteroNodeDataView, HeteroEdgeDataView],
        has_one_type: bool,
    ):
        """Insert valid DGL features into a DGL graph.

        :param features_data: A dictionary storing the DGL features formatted as lists.
        :type features_data: defaultdict[Any, defaultdict[Any, list]]
        :param data: The (empty) ndata or edata instance attribute of a dgl graph, which is about to receive the **features_data**.
        :type data: Union[HeteroNodeDataView, HeteroEdgeDataView]
        :param has_one_type: Set to True if the DGL graph only has one ntype, or one etype.
        :type has_one_type: bool
        """
        col_dict: dict
        for key, col_dict in features_data.items():
            for col, array in col_dict.items():
                data[key] = (
                    torch.tensor(array)
                    if has_one_type
                    else {**data[key], col: torch.tensor(array)}
                )

    def __prepare_adb_attributes(
        self,
        data: Union[HeteroNodeDataView, HeteroEdgeDataView],
        features: set,
        id: int,
        doc: dict,
        col: str,
        has_one_type: bool,
    ):
        """Convert DGL features into a set of ArangoDB attributes for a given document

        :param data: The ndata or edata instance attribute of a dgl graph, filled with node or edge feature data.
        :type data: Union[HeteroNodeDataView, HeteroEdgeDataView]
        :param features: A set of DGL feature keys to convert into ArangoDB attributes
        :type features: set
        :param id: The ID of the current DGL node / edge
        :type id: int
        :param doc: The current ArangoDB document
        :type doc: dict
        :param col: The collection the current document belongs to
        :type col: str
        :param has_one_type: Set to True if the DGL graph only has one ntype, or one etype.
        :type has_one_type: bool
        """
        for key in features:
            tensor = data[key] if has_one_type else data[key][col]
            doc[key] = self.__cntrl._dgl_feature_to_adb_attribute(key, col, tensor[id])

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
