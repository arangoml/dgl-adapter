#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Union

from arango import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase
from arango.graph import Graph as ArangoDBGraph
from arango.result import Result
from dgl import DGLGraph, heterograph
from dgl.heterograph import DGLHeteroGraph
from dgl.view import HeteroEdgeDataView, HeteroNodeDataView
from torch import tensor
from torch.functional import Tensor

from .abc import Abstract_ADBDGL_Adapter
from .controller import ADBDGL_Controller
from .typings import ArangoMetagraph, DGLCanonicalEType, DGLDataDict, Json


class ADBDGL_Adapter(Abstract_ADBDGL_Adapter):
    """ArangoDB-DGL adapter.

    :param conn: Connection details to an ArangoDB instance.
    :type conn: adbdgl_adapter.typings.Json
    :param controller: The ArangoDB-DGL controller, for controlling how
        ArangoDB attributes are converted into DGL features, and vice-versa.
        Optionally re-defined by the user if needed (otherwise defaults to
        ADBDGL_Controller).
    :type controller: adbdgl_adapter.controller.ADBDGL_Controller
    :raise ValueError: If missing required keys in conn
    """

    def __init__(
        self,
        conn: Json,
        controller: ADBDGL_Controller = ADBDGL_Controller(),
    ):
        self.__validate_attributes("connection", set(conn), self.CONNECTION_ATRIBS)
        if issubclass(type(controller), ADBDGL_Controller) is False:
            msg = "controller must inherit from ADBDGL_Controller"
            raise TypeError(msg)

        username: str = conn["username"]
        password: str = conn["password"]
        db_name: str = conn["dbName"]
        host: str = conn["hostname"]
        protocol: str = conn.get("protocol", "https")
        port = str(conn.get("port", 8529))

        url = protocol + "://" + host + ":" + port

        print(f"Connecting to {url}")
        self.__db = ArangoClient(hosts=url).db(db_name, username, password, verify=True)
        self.__cntrl: ADBDGL_Controller = controller

    def db(self) -> StandardDatabase:
        return self.__db

    def arangodb_to_dgl(
        self, name: str, metagraph: ArangoMetagraph, **query_options: Any
    ) -> DGLHeteroGraph:
        """Create a DGLHeteroGraph from the user-defined metagraph.

        :param name: The DGL graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to DGL, along with their associated attributes to keep.
        :type metagraph: adbdgl_adapter.typings.ArangoMetagraph
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance.
        :type query_options: Any
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
                "transaction": {
                    "transaction_amt", "receiver_bank_id", "sender_bank_id"
                },
            },
        }
        """
        self.__validate_attributes("graph", set(metagraph), self.METAGRAPH_ATRIBS)

        # Maps ArangoDB vertex IDs to DGL node IDs
        adb_map: Dict[str, Dict[str, Any]] = dict()

        # Dictionaries for constructing a heterogeneous graph.
        data_dict: DGLDataDict = dict()
        ndata: DefaultDict[Any, Any] = defaultdict(lambda: defaultdict(list))
        edata: DefaultDict[Any, Any] = defaultdict(lambda: defaultdict(list))

        adb_v: Json
        for v_col, atribs in metagraph["vertexCollections"].items():
            for i, adb_v in enumerate(
                self.__fetch_adb_docs(v_col, atribs, query_options)
            ):
                adb_map[adb_v["_id"]] = {
                    "id": i,
                    "col": v_col,
                }

                self.__prepare_dgl_features(ndata, atribs, adb_v, v_col)

        adb_e: Json
        from_col: Set[str] = set()
        to_col: Set[str] = set()
        for e_col, atribs in metagraph["edgeCollections"].items():
            from_nodes: List[int] = []
            to_nodes: List[int] = []
            for adb_e in self.__fetch_adb_docs(e_col, atribs, query_options):
                from_node = adb_map[adb_e["_from"]]
                to_node = adb_map[adb_e["_to"]]

                from_col.add(from_node["col"])
                to_col.add(to_node["col"])
                if len(from_col | to_col) > 2:
                    raise ValueError(
                        f"""Can't convert to DGL:
                            too many '_from' & '_to' collections in {e_col}
                        """
                    )

                from_nodes.append(from_node["id"])
                to_nodes.append(to_node["id"])

                self.__prepare_dgl_features(edata, atribs, adb_e, e_col)

            data_dict[(from_col.pop(), e_col, to_col.pop())] = (
                tensor(from_nodes),
                tensor(to_nodes),
            )

        dgl_g: DGLHeteroGraph = heterograph(data_dict)
        has_one_ntype = len(dgl_g.ntypes) == 1
        has_one_etype = len(dgl_g.etypes) == 1

        self.__insert_dgl_features(ndata, dgl_g.ndata, has_one_ntype)
        self.__insert_dgl_features(edata, dgl_g.edata, has_one_etype)

        print(f"DGL: {name} created")
        return dgl_g

    def arangodb_collections_to_dgl(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        **query_options: Any,
    ) -> DGLHeteroGraph:
        """Create a DGL graph from ArangoDB collections.

        :param name: The DGL graph name.
        :type name: str
        :param v_cols: A set of ArangoDB vertex collections to
            import to DGL.
        :type v_cols: Set[str]
        :param e_cols: A set of ArangoDB edge collections to import to DGL.
        :type e_cols: Set[str]
        :param query_options: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type query_options: Any
        :return: A DGL Heterograph
        :rtype: dgl.heterograph.DGLHeteroGraph
        """
        metagraph: ArangoMetagraph = {
            "vertexCollections": {col: set() for col in v_cols},
            "edgeCollections": {col: set() for col in e_cols},
        }

        return self.arangodb_to_dgl(name, metagraph, **query_options)

    def arangodb_graph_to_dgl(self, name: str, **query_options: Any) -> DGLHeteroGraph:
        """Create a DGL graph from an ArangoDB graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param query_options: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type query_options: Any
        :return: A DGL Heterograph
        :rtype: dgl.heterograph.DGLHeteroGraph
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_dgl(name, v_cols, e_cols, **query_options)

    def dgl_to_arangodb(
        self, name: str, dgl_g: Union[DGLGraph, DGLHeteroGraph], batch_size: int = 1000
    ) -> ArangoDBGraph:
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
        is_default = dgl_g.canonical_etypes == self.DEFAULT_CANONICAL_ETYPE
        adb_v_cols: List[str] = [name + dgl_g.ntypes[0]] if is_default else dgl_g.ntypes
        adb_e_cols: List[str] = [name + dgl_g.etypes[0]] if is_default else dgl_g.etypes
        e_definitions = self.etypes_to_edefinitions(
            [
                (
                    adb_v_cols[0],
                    adb_e_cols[0],
                    adb_v_cols[0],
                )
            ]
            if is_default
            else dgl_g.canonical_etypes
        )

        has_one_ntype = len(dgl_g.ntypes) == 1
        has_one_etype = len(dgl_g.etypes) == 1

        adb_documents: DefaultDict[str, List[Json]] = defaultdict(list)
        for v_col in adb_v_cols:
            ntype = None if is_default else v_col
            v_col_docs = adb_documents[v_col]

            if self.__db.has_collection(v_col) is False:
                self.__db.create_collection(v_col)

            node: Tensor
            for node in dgl_g.nodes(ntype):
                dgl_node_id = node.item()
                adb_vertex = {"_key": str(dgl_node_id)}
                self.__prepare_adb_attributes(
                    dgl_g.ndata,
                    dgl_g.node_attr_schemes(ntype).keys(),
                    dgl_node_id,
                    adb_vertex,
                    v_col,
                    has_one_ntype,
                )

                self.__insert_adb_docs(v_col, v_col_docs, adb_vertex, batch_size)

        from_col: str
        to_col: str
        from_nodes: Tensor
        to_nodes: Tensor
        for e_col in adb_e_cols:
            etype = None if is_default else e_col
            e_col_docs = adb_documents[e_col]

            if self.__db.has_collection(e_col) is False:
                self.__db.create_collection(e_col, edge=True)

            if is_default:
                from_col = to_col = adb_v_cols[0]
            else:
                from_col, _, to_col = dgl_g.to_canonical_etype(e_col)

            from_nodes, to_nodes = dgl_g.edges(etype=etype)
            for dgl_edge_id, (from_node, to_node) in enumerate(
                zip(from_nodes, to_nodes)
            ):
                adb_edge = {
                    "_key": str(dgl_edge_id),
                    "_from": f"{from_col}/{str(from_node.item())}",
                    "_to": f"{to_col}/{str(to_node.item())}",
                }
                self.__prepare_adb_attributes(
                    dgl_g.edata,
                    dgl_g.edge_attr_schemes(etype).keys(),
                    dgl_edge_id,
                    adb_edge,
                    e_col,
                    has_one_etype,
                )

                self.__insert_adb_docs(e_col, e_col_docs, adb_edge, batch_size)

        self.__db.delete_graph(name, ignore_missing=True)
        adb_graph: ArangoDBGraph = self.__db.create_graph(name, e_definitions)

        for col, doc_list in adb_documents.items():  # insert remaining documents
            self.__db.collection(col).import_bulk(doc_list, on_duplicate="replace")

        print(f"ArangoDB: {name} created")
        return adb_graph

    def etypes_to_edefinitions(
        self, canonical_etypes: List[DGLCanonicalEType]
    ) -> List[Json]:
        """Converts a DGL graph's canonical_etypes property to ArangoDB graph edge definitions

        :param canonical_etypes: A list of string triplets (str, str, str) for
            source node type, edge type and destination node type.
        :type canonical_etypes: List[adbdgl_adapter.typings.DGLCanonicalEType]
        :return: ArangoDB Edge Definitions
        :rtype: List[adbdgl_adapter.typings.Json]

        Here is an example of **edge_definitions**:

        .. code-block:: python
        [
            {
                "edge_collection": "teaches",
                "from_vertex_collections": ["Teacher"],
                "to_vertex_collections": ["Lecture"]
            }
        ]
        """
        edge_definitions: List[Json] = []
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
        features_data: DefaultDict[Any, Any],
        attributes: Set[str],
        doc: Json,
        col: str,
    ) -> None:
        """Convert a set of ArangoDB attributes into valid DGL features

        :param features_data: A dictionary storing the DGL features formatted as lists.
        :type features_data: Defaultdict[Any, Any]
        :param attributes: A set of ArangoDB attribute keys to convert into DGL features
        :type attributes: Set[str]
        :param doc: The current ArangoDB document
        :type doc: adbdgl_adapter.typings.Json
        :param col: The collection the current document belongs to
        :type col: str
        """
        key: str
        for key in attributes:
            arr: List[Any] = features_data[key][col]
            arr.append(
                self.__cntrl._adb_attribute_to_dgl_feature(key, col, doc.get(key, None))
            )

    def __insert_dgl_features(
        self,
        features_data: DefaultDict[Any, Any],
        data: Union[HeteroNodeDataView, HeteroEdgeDataView],
        has_one_type: bool,
    ) -> None:
        """Insert valid DGL features into a DGL graph.

        :param features_data: A dictionary storing the DGL features formatted as lists.
        :type features_data: Defaultdict[Any, Any]
        :param data: The (empty) ndata or edata instance attribute of a dgl graph,
            which is about to receive **features_data**.
        :type data: Union[dgl.view.HeteroNodeDataView, dgl.view.HeteroEdgeDataView]
        :param has_one_type: Set to True if the DGL graph only has one ntype,
            or one etype.
        :type has_one_type: bool
        """
        col_dict: Dict[str, List[Any]]
        for key, col_dict in features_data.items():
            for col, array in col_dict.items():
                data[key] = (
                    tensor(array) if has_one_type else {**data[key], col: tensor(array)}
                )

    def __prepare_adb_attributes(
        self,
        data: Union[HeteroNodeDataView, HeteroEdgeDataView],
        features: Set[Any],
        id: Union[int, float, bool],
        doc: Json,
        col: str,
        has_one_type: bool,
    ) -> None:
        """Convert DGL features into a set of ArangoDB attributes for a given document

        :param data: The ndata or edata instance attribute of a dgl graph, filled with
            node or edge feature data.
        :type data: Union[dgl.view.HeteroNodeDataView, dgl.view.HeteroEdgeDataView]
        :param features: A set of DGL feature keys to convert into ArangoDB attributes
        :type features: Set[Any]
        :param id: The ID of the current DGL node / edge
        :type id: Union[int, float, bool]
        :param doc: The current ArangoDB document
        :type doc: adbdgl_adapter.typings.Json
        :param col: The collection the current document belongs to
        :type col: str
        :param has_one_type: Set to True if the DGL graph only has one ntype,
            or one etype.
        :type has_one_type: bool
        """
        for key in features:
            tensor = data[key] if has_one_type else data[key][col]
            doc[key] = self.__cntrl._dgl_feature_to_adb_attribute(key, col, tensor[id])

    def __insert_adb_docs(
        self,
        col: str,
        col_docs: List[Json],
        doc: Json,
        batch_size: int,
    ) -> None:
        """Insert an ArangoDB document into a list. If the list exceeds
        batch_size documents, insert into the ArangoDB collection.

        :param col: The collection name
        :type col: str
        :param col_docs: The existing documents data belonging to the collection.
        :type col_docs: List[adbdgl_adapter.typings.Json]
        :param doc: The current document to insert.
        :type doc: adbdgl_adapter.typings.Json
        :param batch_size: The maximum number of documents to insert at once
        :type batch_size: int
        """
        col_docs.append(doc)

        if len(col_docs) >= batch_size:
            self.__db.collection(col).import_bulk(col_docs, on_duplicate="replace")
            col_docs.clear()

    def __fetch_adb_docs(
        self, col: str, attributes: Set[str], query_options: Any
    ) -> Result[Cursor]:
        """Fetches ArangoDB documents within a collection.

        :param col: The ArangoDB collection.
        :type col: str
        :param attributes: The set of document attributes.
        :type attributes: Set[str]
        :param query_options: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type query_options: Any
        :return: Result cursor.
        :rtype: arango.cursor.Cursor
        """
        aql = f"""
            FOR doc IN {col}
                RETURN MERGE(
                    KEEP(doc, {list(attributes)}),
                    {{"_id": doc._id}},
                    doc._from ? {{"_from": doc._from, "_to": doc._to}}: {{}}
                )
        """

        return self.__db.aql.execute(aql, **query_options)

    def __validate_attributes(
        self, type: str, attributes: Set[str], valid_attributes: Set[str]
    ) -> None:
        """Validates that a set of attributes includes the required valid
        attributes.

        :param type: The context of the attribute validation
            (e.g connection attributes, graph attributes, etc).
        :type type: str
        :param attributes: The provided attributes, possibly invalid.
        :type attributes: Set[str]
        :param valid_attributes: The valid attributes.
        :type valid_attributes: Set[str]
        :raise ValueError: If **valid_attributes** is not a subset of **attributes**
        """
        if valid_attributes.issubset(attributes) is False:
            missing_attributes = valid_attributes - attributes
            raise ValueError(f"Missing {type} attributes: {missing_attributes}")
