#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Union

from arango.database import Database
from arango.graph import Graph as ADBGraph
from dgl import DGLGraph, DGLHeteroGraph, graph, heterograph
from dgl.view import HeteroEdgeDataView, HeteroNodeDataView
from pandas import DataFrame
from torch import Tensor, cat, tensor

from .abc import Abstract_ADBDGL_Adapter
from .controller import ADBDGL_Controller
from .exceptions import ADBMetagraphError, DGLMetagraphError
from .typings import (
    ADBMetagraph,
    ADBMetagraphValues,
    DGLCanonicalEType,
    DGLDataDict,
    DGLMetagraph,
    DGLMetagraphValues,
    Json,
)
from .utils import logger, progress, validate_adb_metagraph, validate_dgl_metagraph


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
        self.__cntrl = controller

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
        self, name: str, metagraph: ADBMetagraph, **query_options: Any
    ) -> Union[DGLGraph, DGLHeteroGraph]:
        """Create a DGL graph from ArangoDB data. DOES carry
            over node/edge features/labels, via the **metagraph**.

        :param name: The DGL graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to DGL, along with collection-level specifications to indicate
            which ArangoDB attributes will become DGL features/labels.
            See below for examples of **metagraph**.
        :type metagraph: adbdgl_adapter.typings.ADBMetagraph
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A DGL Homogeneous or Heterogeneous graph object
        :rtype: dgl.DGLGraph | dgl.DGLHeteroGraph
        :raise adbdgl_adapter.exceptions.ADBMetagraphError: If invalid metagraph.

        #TODO: Metagraph examples
        """
        logger.debug(f"--arangodb_to_dgl('{name}')--")

        validate_adb_metagraph(metagraph)

        is_homogeneous = (
            len(metagraph["vertexCollections"]) == 1
            and len(metagraph["edgeCollections"]) == 1
        )

        # Maps ArangoDB vertex IDs to DGL node IDs
        adb_map: Dict[str, Json] = dict()

        # The data for constructing a graph,
        # which takes the form of (U, V).
        # (U[i], V[i]) forms the edge with ID i in the graph.
        data_dict: DGLDataDict = dict()

        # The node data view for storing node features
        ndata: DefaultDict[str, DefaultDict[str, Tensor]]
        ndata = defaultdict(lambda: defaultdict())

        # The edge data view for storing edge features
        edata: DefaultDict[str, DefaultDict[DGLCanonicalEType, Tensor]]
        edata = defaultdict(lambda: defaultdict())

        for v_col, meta in metagraph["vertexCollections"].items():
            logger.debug(f"Preparing '{v_col}' vertices")

            df = self.__fetch_adb_docs(v_col, query_options)
            adb_map.update({adb_id: dgl_id for dgl_id, adb_id in enumerate(df["_id"])})

            for k, v in meta.items():
                ndata[k][v_col] = self.__build_tensor_from_dataframe(df, k, v)

        v_cols: List[str] = list(metagraph["vertexCollections"].keys())
        for e_col, meta in metagraph["edgeCollections"].items():
            logger.debug(f"Preparing '{e_col}' edges")

            df = self.__fetch_adb_docs(e_col, query_options)
            df["from_col"] = df["_from"].str.split("/").str[0]
            df["to_col"] = df["_to"].str.split("/").str[0]

            for (from_col, to_col), count in (
                df[["from_col", "to_col"]].value_counts().items()
            ):
                edge_type: DGLCanonicalEType = (from_col, e_col, to_col)
                if from_col not in v_cols or to_col not in v_cols:
                    msg = f"""
                        Skipping {edge_type}, as its vertex collections
                        were not specified in the metagraph.
                    """
                    logger.debug(msg)
                    continue

                logger.debug(f"Preparing {count} '{edge_type}' edges")

                # Get the edge data corresponding to the current edge type
                et_df = df[(df["from_col"] == from_col) & (df["to_col"] == to_col)]
                from_nodes = [adb_map[_id] for _id in et_df["_from"]]
                to_nodes = [adb_map[_id] for _id in et_df["_to"]]

                data_dict[edge_type] = (tensor(from_nodes), tensor(to_nodes))
                for k, v in meta.items():
                    edata[k][edge_type] = self.__build_tensor_from_dataframe(
                        et_df, k, v
                    )

        dgl_g: Union[DGLGraph, DGLHeteroGraph]
        if is_homogeneous:
            data = list(data_dict.values())[0]
            dgl_g = graph(data)
        else:
            dgl_g = heterograph(data_dict)

        has_one_ntype = len(dgl_g.ntypes) == 1
        has_one_etype = len(dgl_g.canonical_etypes) == 1

        self.__set_dgl_data(dgl_g.ndata, ndata, has_one_ntype)
        self.__set_dgl_data(dgl_g.edata, edata, has_one_etype)

        logger.info(f"Created DGL '{name}' Graph")
        return dgl_g

    def arangodb_collections_to_dgl(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        **query_options: Any,
    ) -> Union[DGLGraph, DGLHeteroGraph]:
        """Create a DGL graph from ArangoDB collections. Due to risk of
            ambiguity, this method DOES NOT transfer ArangoDB attributes to DGL.

        :param name: The DGL graph name.
        :type name: str
        :param v_cols: The set of ArangoDB vertex collections to import to DGL.
        :type v_cols: Set[str]
        :param e_cols: The set of ArangoDB edge collections to import to DGL.
        :type e_cols: Set[str]
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A DGL Homogeneous or Heterogeneous graph object
        :rtype: dgl.DGLGraph | dgl.DGLHeteroGraph
        :raise adbdgl_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        metagraph: ADBMetagraph = {
            "vertexCollections": {col: dict() for col in v_cols},
            "edgeCollections": {col: dict() for col in e_cols},
        }

        return self.arangodb_to_dgl(name, metagraph, **query_options)

    def arangodb_graph_to_dgl(
        self, name: str, **query_options: Any
    ) -> Union[DGLGraph, DGLHeteroGraph]:
        """Create a DGL graph from an ArangoDB graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A DGL Homogeneous or Heterogeneous graph object
        :rtype: dgl.DGLGraph | dgl.DGLHeteroGraph
        :raise adbdgl_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_dgl(name, v_cols, e_cols, **query_options)

    def dgl_to_arangodb(
        self,
        name: str,
        dgl_g: Union[DGLGraph, DGLHeteroGraph],
        metagraph: DGLMetagraph = {},
        explicit_metagraph: bool = True,
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ADBGraph:
        """Create an ArangoDB graph from a DGL graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param dgl_g: The existing DGL graph.
        :type dgl_g: Union[dgl.DGLGraph, dgl.heterograph.DGLHeteroGraph]
        :param metagraph: An optional object mapping the DGL keys of
            the node & edge data to strings, list of strings, or user-defined
            functions. NOTE: Unlike the metagraph for ArangoDB to DGL, this
            one is optional. See below for an example of **metagraph**.
        :type metagraph: adbdgl_adapter.typings.DGLMetagraph
        :param explicit_metagraph: Whether to take the metagraph at face value or not.
            If False, node & edge types OMITTED from the metagraph will be
            brought over into ArangoDB. Defaults to True.
        :type explicit_metagraph: bool
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections. Defaults to False.
        :type overwrite_graph: bool
        :param import_options: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type import_options: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        :raise adbdgl_adapter.exceptions.DGLMetagraphError: If invalid metagraph.

        #TODO: Metagraph examples
        """
        logger.debug(f"--dgl_to_arangodb('{name}')--")

        validate_dgl_metagraph(metagraph)

        has_one_ntype = len(dgl_g.ntypes) == 1
        has_one_etype = len(dgl_g.canonical_etypes) == 1
        has_default_canonical_etypes = dgl_g.canonical_etypes == ["_N", "_E", "_N"]

        node_types: List[str]
        edge_types: List[DGLCanonicalEType]
        if metagraph and explicit_metagraph:
            node_types = metagraph.get("nodeTypes", {}).keys()  # type: ignore
            edge_types = metagraph.get("edgeTypes", {}).keys()  # type: ignore

        elif has_default_canonical_etypes:
            n_type = name + "_N"
            node_types = [n_type]
            edge_types = [(n_type, name + "_E", n_type)]

        else:
            node_types = dgl_g.ntypes
            edge_types = dgl_g.canonical_etypes

        if overwrite_graph:
            logger.debug("Overwrite graph flag is True. Deleting old graph.")
            self.__db.delete_graph(name, ignore_missing=True)

        if self.__db.has_graph(name):
            adb_graph = self.__db.graph(name)
        else:
            edge_definitions = self.etypes_to_edefinitions(edge_types)
            orphan_collections = self.ntypes_to_ocollections(node_types, edge_types)
            adb_graph = self.__db.create_graph(
                name, edge_definitions, orphan_collections
            )

        n_meta = metagraph.get("nodeTypes", {})
        for n_type in node_types:
            n_key = None if has_one_ntype else n_type

            num_nodes = dgl_g.num_nodes(n_key)
            df = DataFrame([{"_key": str(i)} for i in range(num_nodes)])

            meta = n_meta.get(n_type, {})
            for k, t in dgl_g.nodes[n_key].data.items():
                if type(t) is Tensor and len(t) == num_nodes:
                    v = meta.get(k, k)
                    df = df.join(self.__build_dataframe_from_tensor(t, k, v))

            if type(self.__cntrl) is not ADBDGL_Controller:
                f = lambda n: self.__cntrl._prepare_dgl_node(n, n_type)
                df = df.apply(f, axis=1)

            self.__insert_adb_docs(n_type, df.to_dict("records"), import_options)

        e_meta = metagraph.get("edgeTypes", {})
        for e_type in edge_types:
            e_key = None if has_one_etype else e_type
            from_col, _, to_col = e_type

            from_nodes, to_nodes = dgl_g.edges(etype=e_key)
            df = DataFrame(
                zip(*(from_nodes.tolist(), to_nodes.tolist())), columns=["_from", "_to"]
            )
            df["_from"] = from_col + "/" + df["_from"].astype(str)
            df["_to"] = to_col + "/" + df["_to"].astype(str)

            meta = e_meta.get(e_type, {})            
            for k, t in dgl_g.edges[e_key].data.items():
                if type(t) is Tensor and len(t) == dgl_g.num_edges(e_key):
                    v = meta.get(k, k)
                    df = df.join(self.__build_dataframe_from_tensor(t, k, v))

            if type(self.__cntrl) is not ADBDGL_Controller:
                f = lambda e: self.__cntrl._prepare_dgl_edge(e, e_type)
                df = df.apply(f, axis=1)

            self.__insert_adb_docs(e_type, df.to_dict("records"), import_options)

        logger.info(f"Created ArangoDB '{name}' Graph")
        return adb_graph

    def etypes_to_edefinitions(self, edge_types: List[DGLCanonicalEType]) -> List[Json]:
        """Converts a DGL graph's canonical_etypes property to ArangoDB graph edge definitions

        :param edge_types: A list of string triplets (str, str, str) for
            source node type, edge type and destination node type.
        :type edge_types: List[adbdgl_adapter.typings.DGLCanonicalEType]
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

        if not edge_types:
            return []

        edge_type_map: DefaultDict[str, DefaultDict[str, Set[str]]]
        edge_type_map = defaultdict(lambda: defaultdict(set))

        for edge_type in edge_types:
            from_col, e_col, to_col = edge_type
            edge_type_map[e_col]["from"].add(from_col)
            edge_type_map[e_col]["to"].add(to_col)

        edge_definitions: List[Json] = []
        for e_col, v_cols in edge_type_map.items():
            edge_definitions.append(
                {
                    "from_vertex_collections": list(v_cols["from"]),
                    "edge_collection": e_col,
                    "to_vertex_collections": list(v_cols["to"]),
                }
            )

        return edge_definitions

    def ntypes_to_ocollections(
        self, node_types: List[str], edge_types: List[DGLCanonicalEType]
    ) -> List[str]:
        """Converts DGL node_types to ArangoDB orphan collections, if any.

        :param node_types: A list of strings representing the DGL node types.
        :type node_types: List[str]
        :param edge_types: A list of string triplets (str, str, str) for
            source node type, edge type and destination node type.
        :type edge_types: List[adbdgl_adapter.typings.DGLCanonicalEType]
        :return: ArangoDB Orphan Collections
        :rtype: List[str]
        """

        non_orphan_collections = set()
        for from_col, _, to_col in edge_types:
            non_orphan_collections.add(from_col)
            non_orphan_collections.add(to_col)

        orphan_collections = set(node_types) ^ non_orphan_collections
        return list(orphan_collections)

    def __set_dgl_data(
        self,
        dgl_data: Union[HeteroNodeDataView, HeteroEdgeDataView],
        dgl_data_temp: Union[
            DefaultDict[str, DefaultDict[str, Tensor]],
            DefaultDict[str, DefaultDict[DGLCanonicalEType, Tensor]],
        ],
        has_one_type: bool,
    ) -> None:
        """Copies **dgl_data_temp** into **dgl_data**. This method is (unfortunately)
            required, since a dgl graph's `ndata` and `edata` properties can't be
            manually set (i.e `g.ndata = ndata` is not possible).

        :param dgl_data: The (empty) ndata or edata instance attribute of a dgl graph,
            which is about to receive **dgl_data_temp**.
        :type dgl_data: Union[dgl.view.HeteroNodeDataView, dgl.view.HeteroEdgeDataView]
        :param dgl_data_temp: A temporary place to store the ndata or edata features.
        :type dgl_data_temp: DefaultDict[str, DefaultDict[str, Tensor]]
        :param has_one_type: Set to True if the DGL graph only has one
            node type or edge type.
        :type has_one_type: bool
        """
        for feature_name, feature_map in dgl_data_temp.items():
            for data_type, dgl_tensor in feature_map.items():
                dgl_data[feature_name] = (
                    dgl_tensor if has_one_type else {data_type: dgl_tensor}
                )

    def __fetch_adb_docs(self, col: str, query_options: Any) -> DataFrame:
        """Fetches ArangoDB documents within a collection. Returns the
            documents in a Pandas DataFrame.

        :param col: The ArangoDB collection.
        :type col: str
        :param query_options: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type query_options: Any
        :return: A Pandas DataFrame representing the ArangoDB documents.
        :rtype: pandas.DataFrame
        """
        aql = """
            FOR doc IN @@col
                RETURN doc
        """

        with progress(
            f"Export: {col}",
            text_style="#97C423",
            spinner_style="#7D3B04",
        ) as p:
            p.add_task("__fetch_adb_docs")

            return DataFrame(
                self.__db.aql.execute(
                    aql, count=True, bind_vars={"@col": col}, **query_options
                )
            )

    def __insert_adb_docs(
        self, doc_type: Union[str, DGLCanonicalEType], docs: List[Json], kwargs: Any
    ) -> None:
        """Insert ArangoDB documents into their ArangoDB collection.

        :param doc_type: The node or edge type of the soon-to-be ArangoDB documents
        :type doc_type: str | tuple[str, str, str]
        :param docs: To-be-inserted ArangoDB documents
        :type docs: List[Json]
        :param kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        """
        col = doc_type if type(doc_type) is str else doc_type[1]

        with progress(
            f"Import: {doc_type} ({len(docs)})",
            text_style="#349AEF",
            spinner_style="#FBFEFE",
        ) as p:
            p.add_task("__insert_adb_docs")

            result = self.__db.collection(col).import_bulk(docs, **kwargs)
            logger.debug(result)

    def __build_tensor_from_dataframe(
        self,
        adb_df: DataFrame,
        meta_key: str,
        meta_val: ADBMetagraphValues,
    ) -> Tensor:
        """Constructs a DGL-ready Tensor from a Pandas Dataframe, based on
        the nature of the user-defined metagraph.

        :param adb_df: The Pandas Dataframe representing ArangoDB data.
        :type adb_df: pandas.DataFrame
        :param meta_key: The current ArangoDB-DGL metagraph key
        :type meta_key: str
        :param meta_val: The value mapped to **meta_key** to
            help convert **df** into a DGL-ready Tensor.
            e.g the value of `metagraph['vertexCollections']['users']['x']`.
        :type meta_val: adbdgl_adapter.typings.ADBMetagraphValues
        :return: A DGL-ready tensor equivalent to the dataframe
        :rtype: torch.Tensor
        :raise adbdgl_adapter.exceptions.ADBMetagraphError: If invalid **meta_val**.
        """
        logger.debug(f"__build_tensor_from_dataframe(df, '{meta_key}', {meta_val})")

        if type(meta_val) is str:
            return tensor(adb_df[meta_val].to_list())

        if type(meta_val) is dict:
            data = []
            for attr, encoder in meta_val.items():
                if encoder is None:
                    data.append(tensor(adb_df[attr].to_list()))
                elif callable(encoder):
                    data.append(encoder(adb_df[attr]))
                else:  # pragma: no cover
                    msg = f"Invalid encoder for ArangoDB attribute '{attr}': {encoder}"
                    raise ADBMetagraphError(msg)

            return cat(data, dim=-1)

        if callable(meta_val):
            # **meta_val** is a user-defined that returns a tensor
            user_defined_result = meta_val(adb_df)

            if type(user_defined_result) is not Tensor:  # pragma: no cover
                msg = f"Invalid return type for function {meta_val} ('{meta_key}')"
                raise ADBMetagraphError(msg)

            return user_defined_result

        raise ADBMetagraphError(f"Invalid {meta_val} type")  # pragma: no cover

    def __build_dataframe_from_tensor(
        self,
        dgl_tensor: Tensor,
        meta_key: str,
        meta_val: DGLMetagraphValues,
    ) -> DataFrame:
        """Builds a Pandas DataFrame from DGL Tensor, based on
        the nature of the user-defined metagraph.

        :param dgl_tensor: The Tensor representing DGL data.
        :type dgl_tensor: torch.Tensor
        :param meta_key: The current DGL-ArangoDB metagraph key
        :type meta_key
        :param meta_val: The value mapped to the DGL-ArangoDB metagraph key to
            help convert **tensor** into a Pandas Dataframe.
            e.g the value of `metagraph['nodeTypes']['users']['x']`.
        :type meta_val: adbdgl_adapter.typings.DGLMetagraphValues
        :return: A Pandas DataFrame equivalent to the Tensor
        :rtype: pandas.DataFrame
        :raise adbdgl_adapter.exceptions.DGLMetagraphError: If invalid **meta_val**.
        """
        logger.debug(f"__build_dataframe_from_tensor(df, '{meta_key}', {meta_val})")

        if type(meta_val) in [str, list]:
            columns = [meta_val] if type(meta_val) is str else meta_val

            df = DataFrame(columns=columns)
            df[meta_val] = dgl_tensor.tolist()
            return df

        if callable(meta_val):
            # **meta_val** is a user-defined function that returns a dataframe
            user_defined_result = meta_val(dgl_tensor)

            if type(user_defined_result) is not DataFrame:  # pragma: no cover
                msg = f"Invalid return type for function {meta_val} ('{meta_key}')"
                raise DGLMetagraphError(msg)

            return user_defined_result

        raise DGLMetagraphError(f"Invalid {meta_val} type")  # pragma: no cover
