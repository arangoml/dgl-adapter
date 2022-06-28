#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

from arango.cursor import Cursor
from arango.database import Database
from arango.graph import Graph as ADBGraph
from arango.result import Result
from dgl import DGLGraph, DGLHeteroGraph, heterograph
from dgl.view import HeteroEdgeDataView, HeteroNodeDataView
from torch import tensor
from torch.functional import Tensor

from .abc import Abstract_ADBDGL_Adapter
from .controller import ADBDGL_Controller
from .typings import ArangoMetagraph, DGLCanonicalEType, DGLDataDict, Json
from .utils import logger


class ADBDGL_Adapter(Abstract_ADBDGL_Adapter):
    """ArangoDB-DGL adapter.

    :param db: A python-arango database instance
    :type db: arango.database.Database
    :param controller: The ArangoDB-DGL controller, for controlling how
        ArangoDB attributes are converted into DGL features, and vice-versa.
        Optionally re-defined by the user if needed (otherwise defaults to
        ADBDGL_Controller).
    :type controller: adbdgl_adapter.controller.ADBDGL_Controller
    :param logging_lvl: Defaults to logging.INFO. Other useful options are
        logging.DEBUG (more verbose), and logging.WARNING (less verbose).
    :type logging_lvl: str | int
    :raise ValueError: If invalid parameters
    """

    def __init__(
        self,
        db: Database,
        controller: ADBDGL_Controller = ADBDGL_Controller(),
        logging_lvl: Union[str, int] = logging.INFO,
    ):
        self.set_logging(logging_lvl)

        if issubclass(type(db), Database) is False:
            msg = "**db** parameter must inherit from arango.database.Database"
            raise TypeError(msg)

        if issubclass(type(controller), ADBDGL_Controller) is False:
            msg = "**controller** parameter must inherit from ADBDGL_Controller"
            raise TypeError(msg)

        self.__db = db
        self.__cntrl: ADBDGL_Controller = controller

        logger.info(f"Instantiated ADBDGL_Adapter with database '{db.name}'")

    @property
    def db(self) -> Database:
        return self.__db  # pragma: no cover

    @property
    def cntrl(self) -> ADBDGL_Controller:
        return self.__cntrl  # pragma: no cover

    def set_logging(self, level: Union[int, str]) -> None:
        logger.setLevel(level)

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
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
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
        logger.debug(f"Starting arangodb_to_dgl({name}, ...):")

        # Maps ArangoDB vertex IDs to DGL node IDs
        adb_map: Dict[str, Dict[str, Any]] = dict()

        # Dictionaries for constructing a heterogeneous graph.
        data_dict: DGLDataDict = dict()
        ndata: DefaultDict[Any, Any] = defaultdict(lambda: defaultdict(list))
        edata: DefaultDict[Any, Any] = defaultdict(lambda: defaultdict(list))

        adb_v: Json
        for v_col, atribs in metagraph["vertexCollections"].items():
            logger.debug(f"Preparing '{v_col}' vertices")
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
            logger.debug(f"Preparing '{e_col}' edges")
            from_nodes: List[int] = []
            to_nodes: List[int] = []
            for adb_e in self.__fetch_adb_docs(e_col, atribs, query_options):
                from_node = adb_map[adb_e["_from"]]
                to_node = adb_map[adb_e["_to"]]

                from_col.add(from_node["col"])
                to_col.add(to_node["col"])
                if len(from_col | to_col) > 2:
                    raise ValueError(  # pragma: no cover
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
        logger.debug(f"Is graph '{name}' homogenous? {has_one_ntype and has_one_etype}")

        self.__insert_dgl_features(ndata, dgl_g.ndata, has_one_ntype)
        self.__insert_dgl_features(edata, dgl_g.edata, has_one_etype)

        logger.info(f"Created DGL '{name}' Graph")
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
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
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
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A DGL Heterograph
        :rtype: dgl.heterograph.DGLHeteroGraph
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_dgl(name, v_cols, e_cols, **query_options)

    def dgl_to_arangodb(
        self,
        name: str,
        dgl_g: Union[DGLGraph, DGLHeteroGraph],
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ADBGraph:
        """Create an ArangoDB graph from a DGL graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param dgl_g: The existing DGL graph.
        :type dgl_g: Union[dgl.DGLGraph, dgl.heterograph.DGLHeteroGraph]
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections.
        :type overwrite_graph: bool
        :param import_options: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type import_options: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        """
        logger.debug(f"Starting dgl_to_arangodb({name}, ...):")

        is_default = dgl_g.canonical_etypes == self.DEFAULT_CANONICAL_ETYPE
        logger.debug(f"Is graph '{name}' using default canonical_etypes? {is_default}")

        edge_definitions = self.etypes_to_edefinitions(
            [(name + "_N", name + "_E", name + "_N")]
            if is_default
            else dgl_g.canonical_etypes
        )

        if overwrite_graph:
            logger.debug("Overwrite graph flag is True. Deleting old graph.")
            self.__db.delete_graph(name, ignore_missing=True)

        if self.__db.has_graph(name):
            adb_graph = self.__db.graph(name)
        else:
            adb_graph = self.__db.create_graph(name, edge_definitions)

        adb_v_cols = adb_graph.vertex_collections()
        adb_e_cols = [e_d["edge_collection"] for e_d in adb_graph.edge_definitions()]

        has_one_vcol = len(adb_v_cols) == 1
        has_one_ecol = len(adb_e_cols) == 1
        logger.debug(f"Is graph '{name}' homogenous? {has_one_vcol and has_one_ecol}")

        adb_documents: DefaultDict[str, List[Json]] = defaultdict(list)

        for v_col in adb_v_cols:
            v_col_docs = adb_documents[v_col]
            ntype = None if is_default else v_col
            features = dgl_g.node_attr_schemes(ntype).keys()

            node: Tensor
            logger.debug(f"Preparing {dgl_g.number_of_nodes(ntype)} '{v_col}' nodes")
            for node in dgl_g.nodes(ntype):
                dgl_node_id = node.item()
                adb_vertex = {"_key": str(dgl_node_id)}
                self.__prepare_adb_attributes(
                    dgl_g.ndata,
                    features,
                    dgl_node_id,
                    adb_vertex,
                    v_col,
                    has_one_vcol,
                )

                v_col_docs.append(adb_vertex)

        from_col: str
        to_col: str
        from_n: Tensor
        to_n: Tensor
        for e_col in adb_e_cols:
            e_col_docs = adb_documents[e_col]
            etype = None if is_default else e_col
            features = dgl_g.edge_attr_schemes(etype).keys()

            canonical_etype = None
            if is_default:
                from_col = to_col = adb_v_cols[0]
            else:
                canonical_etype = dgl_g.to_canonical_etype(e_col)
                from_col, _, to_col = canonical_etype

            logger.debug(f"Preparing {dgl_g.number_of_edges(etype)} '{e_col}' edges")
            for index, (from_n, to_n) in enumerate(zip(*dgl_g.edges(etype=etype))):
                adb_edge = {
                    "_key": str(index),
                    "_from": f"{from_col}/{str(from_n.item())}",
                    "_to": f"{to_col}/{str(to_n.item())}",
                }
                self.__prepare_adb_attributes(
                    dgl_g.edata,
                    features,
                    index,
                    adb_edge,
                    e_col,
                    has_one_ecol,
                    canonical_etype,
                )

                e_col_docs.append(adb_edge)

        for col, doc_list in adb_documents.items():  # import documents into ArangoDB
            logger.debug(f"Inserting {len(doc_list)} documents into '{col}'")
            result = self.__db.collection(col).import_bulk(doc_list, **import_options)
            logger.debug(result)

        logger.info(f"Created ArangoDB '{name}' Graph")
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
                logger.debug(f"Inserting {len(array)} '{key}' features into '{col}'")
                data[key] = tensor(array) if has_one_type else {col: tensor(array)}

    def __prepare_adb_attributes(
        self,
        data: Union[HeteroNodeDataView, HeteroEdgeDataView],
        features: Set[Any],
        id: Union[int, float, bool],
        doc: Json,
        col: str,
        has_one_col: bool,
        canonical_etype: Optional[DGLCanonicalEType] = None,
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
        :param has_one_col: Set to True if the ArangoDB graph has one
            vertex collection or one edge collection only.
        :type has_one_col: bool
        :param canonical_etype: The DGL canonical edge type belonging to the current
            **col**, provided that **col** is an edge collection (ignored otherwise).
        :type canonical_etype: adbdgl_adapter.typings.DGLCanonicalEType
        """
        for key in features:
            tensor = data[key] if has_one_col else data[key][canonical_etype or col]
            doc[key] = self.__cntrl._dgl_feature_to_adb_attribute(key, col, tensor[id])

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
