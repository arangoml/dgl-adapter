#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from math import ceil
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union

from arango.cursor import Cursor
from arango.database import StandardDatabase
from arango.graph import Graph as ADBGraph
from dgl import DGLGraph, DGLHeteroGraph, graph, heterograph
from dgl.view import EdgeSpace, HeteroEdgeDataView, HeteroNodeDataView, NodeSpace
from pandas import DataFrame, Series
from rich.console import Group
from rich.live import Live
from rich.progress import Progress
from torch import Tensor, cat, tensor

from .abc import Abstract_ADBDGL_Adapter
from .controller import ADBDGL_Controller
from .exceptions import ADBMetagraphError, DGLMetagraphError
from .typings import (
    ADBMap,
    ADBMetagraph,
    ADBMetagraphValues,
    DGLCanonicalEType,
    DGLData,
    DGLDataDict,
    DGLDataTypes,
    DGLMetagraph,
    DGLMetagraphValues,
    Json,
)
from .utils import (
    get_bar_progress,
    get_export_spinner_progress,
    get_import_spinner_progress,
    logger,
    validate_adb_metagraph,
    validate_dgl_metagraph,
)


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
        db: StandardDatabase,
        controller: ADBDGL_Controller = ADBDGL_Controller(),
        logging_lvl: Union[str, int] = logging.INFO,
    ):
        self.set_logging(logging_lvl)

        if not isinstance(db, StandardDatabase):
            msg = "**db** parameter must inherit from arango.database.StandardDatabase"
            raise TypeError(msg)

        if not isinstance(controller, ADBDGL_Controller):
            msg = "**controller** parameter must inherit from ADBDGL_Controller"
            raise TypeError(msg)

        self.__db = db
        self.__async_db = db.begin_async_execution(return_result=False)
        self.__cntrl = controller

        logger.info(f"Instantiated ADBDGL_Adapter with database '{db.name}'")

    @property
    def db(self) -> StandardDatabase:
        return self.__db  # pragma: no cover

    @property
    def cntrl(self) -> ADBDGL_Controller:
        return self.__cntrl  # pragma: no cover

    def set_logging(self, level: Union[int, str]) -> None:
        logger.setLevel(level)

    ###########################
    # Public: ArangoDB -> DGL #
    ###########################

    def arangodb_to_dgl(
        self, name: str, metagraph: ADBMetagraph, **adb_export_kwargs: Any
    ) -> Union[DGLGraph, DGLHeteroGraph]:
        """Create a DGL graph from an ArangoDB Metagraph. Carries
        over node/edge data via the **metagraph**.

        :param name: The DGL graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to DGL, along with collection-level specifications to indicate
            which ArangoDB attributes will become DGL features/labels.

            The current supported **metagraph** values are:
                1) Set[str]: The set of DGL-ready ArangoDB attributes to store
                    in your DGL graph.

                2) Dict[str, str]: The DGL property name mapped to the ArangoDB
                    attribute name that stores your DGL ready data.

                3) Dict[str, Dict[str, None | Callable]]:
                    The DGL property name mapped to a dictionary, which maps your
                    ArangoDB attribute names to a callable Python Class
                    (i.e has a `__call__` function defined), or to None
                    (if the ArangoDB attribute is already a list of numerics).
                    NOTE: The `__call__` function must take as input a Pandas DataFrame,
                    and must return a PyTorch Tensor.

                4) Dict[str, Callable[[pandas.DataFrame], torch.Tensor]]:
                    The DGL property name mapped to a user-defined function
                    for custom behaviour. NOTE: The function must take as input
                    a Pandas DataFrame, and must return a PyTorch Tensor.

            See below for examples of **metagraph**.
        :type metagraph: adbdgl_adapter.typings.ADBMetagraph
        :param adb_export_kwargs: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type adb_export_kwargs: Any
        :return: A DGL Homogeneous or Heterogeneous graph object
        :rtype: dgl.DGLGraph | dgl.DGLHeteroGraph
        :raise adbdgl_adapter.exceptions.ADBMetagraphError: If invalid metagraph.

        **metagraph** examples

        1)
        .. code-block:: python
        {
            "vertexCollections": {
                "v0": {'x', 'y'}, # equivalent to {'x': 'x', 'y': 'y'}
                "v1": {'x'},
                "v2": {'x'},
            },
            "edgeCollections": {
                "e0": {'edge_attr'},
                "e1": {'edge_weight'},
            },
        }

        The metagraph above specifies that each document
        within the "v0" ArangoDB collection has a "pre-built" feature matrix
        named "x", and also has a node label named "y".
        We map these keys to the "x" and "y" properties of the DGL graph.

        2)
        .. code-block:: python
        {
            "vertexCollections": {
                "v0": {'x': 'v0_features', 'y': 'label'},
                "v1": {'x': 'v1_features'},
                "v2": {'x': 'v2_features'},
            },
            "edgeCollections": {
                "e0": {'edge_attr': 'e0_features'},
                "e1": {'edge_weight': 'edge_weight'},
            },
        }

        The metagraph above specifies that each document
        within the "v0" ArangoDB collection has a "pre-built" feature matrix
        named "v0_features", and also has a node label named "label".
        We map these keys to the "x" and "y" properties of the DGL graph.

        3)
        .. code-block:: python
        from adbdgl_adapter.encoders import IdentityEncoder, CategoricalEncoder

        {
            "vertexCollections": {
                "Movies": {
                    "x": {
                        "Action": IdentityEncoder(dtype=torch.long),
                        "Drama": IdentityEncoder(dtype=torch.long),
                        'Misc': None
                    },
                    "y": "Comedy",
                },
                "Users": {
                    "x": {
                        "Gender": CategoricalEncoder(),
                        "Age": IdentityEncoder(dtype=torch.long),
                    }
                },
            },
            "edgeCollections": {
                "Ratings": { "edge_weight": "Rating" }
            },
        }

        The metagraph above will build the "Movies" feature matrix 'x'
        using the ArangoDB 'Action', 'Drama' & 'misc' attributes, by relying on
        the user-specified Encoders (see adbdgl_adapter.encoders for examples).
        NOTE: If the mapped value is `None`, then it assumes that the ArangoDB attribute
        value is a list containing numerical values only.

        4)
        .. code-block:: python
        def udf_v0_x(v0_df):
            # process v0_df here to return v0 "x" feature matrix
            # ...
            return torch.tensor(v0_df["x"].to_list())

        def udf_v1_x(v1_df):
            # process v1_df here to return v1 "x" feature matrix
            # ...
            return torch.tensor(v1_df["x"].to_list())

        {
            "vertexCollections": {
                "v0": {
                    "x": udf_v0_x, # named functions
                    "y": (lambda df: tensor(df["y"].to_list())), # lambda functions
                },
                "v1": {"x": udf_v1_x},
                "v2": {"x": (lambda df: tensor(df["x"].to_list()))},
            },
            "edgeCollections": {
                "e0": {"edge_attr": (lambda df: tensor(df["edge_attr"].to_list()))},
            },
        }

        The metagraph above provides an interface for a user-defined function to
        build a DGL-ready Tensor from a DataFrame equivalent to the
        associated ArangoDB collection.
        """
        logger.debug(f"--arangodb_to_dgl('{name}')--")

        validate_adb_metagraph(metagraph)

        # Maps ArangoDB Vertex _keys to DGL Node ids
        adb_map: ADBMap = defaultdict(dict)

        # The data for constructing a graph,
        # which takes the form of (U, V).
        # (U[i], V[i]) forms the edge with ID i in the graph.
        data_dict: DGLDataDict = dict()

        # The node data view for storing node features
        ndata: DGLData = defaultdict(lambda: defaultdict(Tensor))

        # The edge data view for storing edge features
        edata: DGLData = defaultdict(lambda: defaultdict(Tensor))

        v_cols: List[str] = list(metagraph["vertexCollections"].keys())

        ######################
        # Vertex Collections #
        ######################

        for v_col, meta in metagraph["vertexCollections"].items():
            logger.debug(f"Preparing '{v_col}' vertices")

            # 1. Fetch ArangoDB vertices
            v_col_cursor, v_col_size = self.__fetch_adb_docs(
                v_col, False, meta, **adb_export_kwargs
            )

            # 2. Process ArangoDB vertices
            self.__process_adb_cursor(
                "#319BF5",
                v_col_cursor,
                v_col_size,
                self.__process_adb_vertex_df,
                v_col,
                adb_map,
                meta,
                ndata=ndata,
            )

        ####################
        # Edge Collections #
        ####################

        # The set of skipped edge types
        edge_type_blacklist: Set[DGLCanonicalEType] = set()

        for e_col, meta in metagraph["edgeCollections"].items():
            logger.debug(f"Preparing '{e_col}' edges")

            # 1. Fetch ArangoDB edges
            e_col_cursor, e_col_size = self.__fetch_adb_docs(
                e_col, True, meta, **adb_export_kwargs
            )

            # 2. Process ArangoDB edges
            self.__process_adb_cursor(
                "#FCFDFC",
                e_col_cursor,
                e_col_size,
                self.__process_adb_edge_df,
                e_col,
                adb_map,
                meta,
                edata=edata,
                data_dict=data_dict,
                v_cols=v_cols,
                edge_type_blacklist=edge_type_blacklist,
            )

        if not data_dict:  # pragma: no cover
            msg = f"""
                Can't create the DGL graph: no complete edge types found.
                The following edge types were skipped due to missing
                vertex collection specifications: {edge_type_blacklist}
            """
            raise ValueError(msg)

        dgl_g = self.__create_dgl_graph(data_dict, adb_map, metagraph)
        self.__link_dgl_data(dgl_g.ndata, ndata, len(dgl_g.ntypes) == 1)
        self.__link_dgl_data(dgl_g.edata, edata, len(dgl_g.canonical_etypes) == 1)

        logger.info(f"Created DGL '{name}' Graph")
        return dgl_g

    def arangodb_collections_to_dgl(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        **adb_export_kwargs: Any,
    ) -> Union[DGLGraph, DGLHeteroGraph]:
        """Create a DGL graph from ArangoDB collections. Due to risk of
        ambiguity, this method DOES NOT transfer ArangoDB attributes to DGL.

        :param name: The DGL graph name.
        :type name: str
        :param v_cols: The set of ArangoDB vertex collections to import to DGL.
        :type v_cols: Set[str]
        :param e_cols: The set of ArangoDB edge collections to import to DGL.
        :type e_cols: Set[str]
        :param adb_export_kwargs: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type adb_export_kwargs: Any
        :return: A DGL Homogeneous or Heterogeneous graph object
        :rtype: dgl.DGLGraph | dgl.DGLHeteroGraph
        :raise adbdgl_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        metagraph: ADBMetagraph = {
            "vertexCollections": {col: dict() for col in v_cols},
            "edgeCollections": {col: dict() for col in e_cols},
        }

        return self.arangodb_to_dgl(name, metagraph, **adb_export_kwargs)

    def arangodb_graph_to_dgl(
        self, name: str, **adb_export_kwargs: Any
    ) -> Union[DGLGraph, DGLHeteroGraph]:
        """Create a DGL graph from an ArangoDB graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param adb_export_kwargs: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type adb_export_kwargs: Any
        :return: A DGL Homogeneous or Heterogeneous graph object
        :rtype: dgl.DGLGraph | dgl.DGLHeteroGraph
        :raise adbdgl_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        graph = self.__db.graph(name)
        v_cols: Set[str] = graph.vertex_collections()
        edge_definitions: List[Json] = graph.edge_definitions()
        e_cols: Set[str] = {c["edge_collection"] for c in edge_definitions}

        return self.arangodb_collections_to_dgl(
            name, v_cols, e_cols, **adb_export_kwargs
        )

    ###########################
    # Public: DGL -> ArangoDB #
    ###########################

    def dgl_to_arangodb(
        self,
        name: str,
        dgl_g: Union[DGLGraph, DGLHeteroGraph],
        metagraph: DGLMetagraph = {},
        explicit_metagraph: bool = True,
        overwrite_graph: bool = False,
        batch_size: Optional[int] = None,
        use_async: bool = False,
        **adb_import_kwargs: Any,
    ) -> ADBGraph:
        """Create an ArangoDB graph from a DGL graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param dgl_g: The existing DGL graph.
        :type dgl_g: Union[dgl.DGLGraph, dgl.heterograph.DGLHeteroGraph]
        :param metagraph: An optional object mapping the DGL keys of
            the node & edge data to strings, list of strings, or user-defined
            functions. NOTE: Unlike the metagraph for ArangoDB to DGL, this
            one is optional.

            The current supported **metagraph** values are:
                1) Set[str]: The set of DGL data properties to store
                    in your ArangoDB database.

                2) Dict[str, str]: The DGL property name mapped to the ArangoDB
                    attribute name that will be used to store your DGL data in ArangoDB.

                3) List[str]: A list of ArangoDB attribute names that will break down
                    your tensor data, resulting in one ArangoDB attribute per feature.
                    Must know the number of node/edge features in advance to take
                    advantage of this metagraph value type.

                4) Dict[str, Callable[[pandas.DataFrame], torch.Tensor]]:
                    The DGL property name mapped to a user-defined function
                    for custom behaviour. NOTE: The function must take as input
                    a PyTorch Tensor, and must return a Pandas DataFrame.

            See below for an example of **metagraph**.
        :type metagraph: adbdgl_adapter.typings.DGLMetagraph
        :param explicit_metagraph: Whether to take the metagraph at face value or not.
            If False, node & edge types OMITTED from the metagraph will still be
            brought over into ArangoDB. Also applies to node & edge attributes.
            Defaults to True.
        :type explicit_metagraph: bool
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections. Defaults to False.
        :type overwrite_graph: bool
        :param batch_size: Process the DGL Nodes & Edges in batches of size
            **batch_size**. Defaults to `None`, which processes each
            NodeStorage & EdgeStorage in one batch.
        :type batch_size: int
        :param use_async: Performs asynchronous ArangoDB ingestion if enabled.
            Defaults to False.
        :type use_async: bool
        :param adb_import_kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type adb_import_kwargs: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        :raise adbdgl_adapter.exceptions.DGLMetagraphError: If invalid metagraph.

        **metagraph** example

        .. code-block:: python
        def y_tensor_to_2_column_dataframe(dgl_tensor):
            # A user-defined function to create two ArangoDB attributes
            # out of the 'y' label tensor
            label_map = {0: "Kiwi", 1: "Blueberry", 2: "Avocado"}

            df = pandas.DataFrame(columns=["label_num", "label_str"])
            df["label_num"] = dgl_tensor.tolist()
            df["label_str"] = df["label_num"].map(label_map)

            return df

        metagraph = {
            "nodeTypes": {
                "v0": {
                    "x": "features",  # 1)
                    "y": y_tensor_to_2_column_dataframe,  # 2)
                },
                "v1": {"x"} # 3)
            },
            "edgeTypes": {
                ("v0", "e0", "v0"): {"edge_attr": [ "a", "b"]}, # 4)
            },
        }

        The metagraph above accomplishes the following:
        1) Renames the DGL 'v0' 'x' feature matrix to 'features'
            when stored in ArangoDB.
        2) Builds a 2-column Pandas DataFrame from the 'v0' 'y' labels
            through a user-defined function for custom behaviour handling.
        3) Transfers the DGL 'v1' 'x' feature matrix under the same name.
        4) Dissasembles the 2-feature Tensor into two ArangoDB attributes,
            where each attribute holds one feature value.
        """
        logger.debug(f"--dgl_to_arangodb('{name}')--")

        validate_dgl_metagraph(metagraph)

        is_custom_controller = type(self.__cntrl) is not ADBDGL_Controller
        is_explicit_metagraph = metagraph != {} and explicit_metagraph

        has_one_ntype = len(dgl_g.ntypes) == 1
        has_one_etype = len(dgl_g.canonical_etypes) == 1

        # Get the Node & Edge types
        node_types, edge_types = self.__get_node_and_edge_types(
            name, dgl_g, metagraph, is_explicit_metagraph
        )

        # Create the ArangoDB Graph
        adb_graph = self.__create_adb_graph(
            name, overwrite_graph, node_types, edge_types
        )

        spinner_progress = get_import_spinner_progress("    ")

        #############
        # DGL Nodes #
        #############

        n_meta = metagraph.get("nodeTypes", {})
        for n_type in node_types:
            meta = n_meta.get(n_type, {})

            n_key = None if has_one_ntype else n_type

            ndata = dgl_g.nodes[n_key].data
            ndata_size = dgl_g.num_nodes(n_key)
            ndata_batch_size = batch_size or ndata_size

            start_index = 0
            end_index = min(ndata_batch_size, ndata_size)
            batches = ceil(ndata_size / ndata_batch_size)

            bar_progress = get_bar_progress(f"(DGL → ADB): '{n_type}'", "#97C423")
            bar_progress_task = bar_progress.add_task(n_type, total=ndata_size)

            with Live(Group(bar_progress, spinner_progress)):
                for _ in range(batches):
                    # 1. Process the Node batch
                    df = self.__process_dgl_node_batch(
                        n_type,
                        ndata,
                        ndata_size,
                        meta,
                        is_explicit_metagraph,
                        is_custom_controller,
                        start_index,
                        end_index,
                    )

                    bar_progress.advance(bar_progress_task, advance=len(df))

                    # 2. Insert the ArangoDB Node Documents
                    self.__insert_adb_docs(
                        spinner_progress, df, n_type, use_async, **adb_import_kwargs
                    )

                    # 3. Update the batch indices
                    start_index = end_index
                    end_index = min(end_index + ndata_batch_size, ndata_size)

        #############
        # DGL Edges #
        #############

        e_meta = metagraph.get("edgeTypes", {})
        for e_type in edge_types:
            meta = e_meta.get(e_type, {})

            e_key = None if has_one_etype else e_type

            edata = dgl_g.edges[e_key].data
            edata_size = dgl_g.num_edges(e_key)
            edata_batch_size = batch_size or edata_size

            start_index = 0
            end_index = min(edata_batch_size, edata_size)
            batches = ceil(edata_size / edata_batch_size)

            bar_progress = get_bar_progress(f"(DGL → ADB): {e_type}", "#994602")
            bar_progress_task = bar_progress.add_task(str(e_type), total=edata_size)

            from_nodes, to_nodes = dgl_g.edges(etype=e_key)

            with Live(Group(bar_progress, spinner_progress)):
                for _ in range(batches):
                    # 1. Process the Edge batch
                    df = self.__process_dgl_edge_batch(
                        e_type,
                        edata,
                        edata_size,
                        meta,
                        from_nodes,
                        to_nodes,
                        is_explicit_metagraph,
                        is_custom_controller,
                        start_index,
                        end_index,
                    )

                    bar_progress.advance(bar_progress_task, advance=len(df))

                    # 2. Insert the ArangoDB Edge Documents
                    self.__insert_adb_docs(
                        spinner_progress, df, e_type[1], use_async, **adb_import_kwargs
                    )

                    # 3. Update the batch indices
                    start_index = end_index
                    end_index = min(end_index + edata_batch_size, edata_size)

        logger.info(f"Created ArangoDB '{name}' Graph")
        return adb_graph

    ############################
    # Private: ArangoDB -> DGL #
    ############################

    def __fetch_adb_docs(
        self,
        col: str,
        is_edge: bool,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        **adb_export_kwargs: Any,
    ) -> Tuple[Cursor, int]:
        """ArangoDB -> DGL: Fetches ArangoDB documents within a collection.
        Returns the documents in a DataFrame.

        :param col: The ArangoDB collection.
        :type col: str
        :param is_edge: True if **col** is an edge collection.
        :type is_edge: bool
        :param meta: The MetaGraph associated to **col**
        :type meta: Set[str] | Dict[str, adbdgl_adapter.typings.ADBMetagraphValues]
        :param adb_export_kwargs: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type adb_export_kwargs: Any
        :return: A DataFrame representing the ArangoDB documents.
        :rtype: pandas.DataFrame
        """

        def get_aql_return_value() -> str:
            """Helper method to formulate the AQL `RETURN` value based on
            the document attributes specified in **meta**
            """
            attributes = ["_key"]
            attributes += ["_from", "_to"] if is_edge else []

            if type(meta) is set:
                attributes += list(meta)

            elif type(meta) is dict:
                for value in meta.values():
                    if type(value) is str:
                        attributes.append(value)
                    elif type(value) is dict:
                        attributes += list(value.keys())
                    elif callable(value):
                        # Cannot determine which attributes to extract if UDFs are used
                        # Therefore we just return the entire document
                        return "doc"

            return f"KEEP(doc, {attributes})"

        col_size: int = self.__db.collection(col).count()

        with get_export_spinner_progress(f"ADB Export: '{col}' ({col_size})") as p:
            p.add_task(col)

            cursor: Cursor = self.__db.aql.execute(
                f"FOR doc IN @@col RETURN {get_aql_return_value()}",
                bind_vars={"@col": col},
                **{**adb_export_kwargs, **{"stream": True}},
            )

            return cursor, col_size

    def __process_adb_cursor(
        self,
        progress_color: str,
        cursor: Cursor,
        col_size: int,
        process_adb_df: Callable[..., int],
        col: str,
        adb_map: ADBMap,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        **kwargs: Any,
    ) -> None:
        """ArangoDB -> DGL: Processes the ArangoDB Cursors for vertices and edges.

        :param progress_color: The progress bar color.
        :type progress_color: str
        :param cursor: The ArangoDB cursor for the current **col**.
        :type cursor: arango.cursor.Cursor
        :param col_size: The size of **col**.
        :type col_size: int
        :param process_adb_df: The function to process the cursor data
            (in the form of a Dataframe).
        :type process_adb_df: Callable
        :param col: The ArangoDB collection for the current **cursor**.
        :type col: str
        :param adb_map: The ArangoDB -> DGL map.
        :type adb_map: adbdgl_adapter.typings.ADBMap
        :param meta: The metagraph for the current **col**.
        :type meta: Set[str] | Dict[str, ADBMetagraphValues]
        :param kwargs: Additional keyword arguments to pass to **process_adb_df**.
        :type args: Any
        """

        progress = get_bar_progress(f"(ADB → DGL): '{col}'", progress_color)
        progress_task_id = progress.add_task(col, total=col_size)

        with Live(Group(progress)):
            i = 0
            while not cursor.empty():
                cursor_batch = len(cursor.batch())
                df = DataFrame([cursor.pop() for _ in range(cursor_batch)])

                i = process_adb_df(i, df, col, adb_map, meta, **kwargs)
                progress.advance(progress_task_id, advance=len(df))

                df.drop(df.index, inplace=True)

                if cursor.has_more():
                    cursor.fetch()

    def __process_adb_vertex_df(
        self,
        i: int,
        df: DataFrame,
        v_col: str,
        adb_map: ADBMap,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        ndata: DGLData,
    ) -> int:
        """ArangoDB -> DGL: Process the ArangoDB Vertex DataFrame
        into the DGL NData object.

        :param i: The last DGL Node id value.
        :type i: int
        :param df: The ArangoDB Vertex DataFrame.
        :type df: pandas.DataFrame
        :param v_col: The ArangoDB Vertex Collection.
        :type v_col: str
        :param adb_map: The ArangoDB -> DGL map.
        :type adb_map: adbdgl_adapter.typings.ADBMap
        :param meta: The metagraph for the current **v_col**.
        :type meta: Set[str] | Dict[str, ADBMetagraphValues]
        :param node_data: The node data view for storing node features
        :type node_data: adbdgl_adapter.typings.DGLData
        :return: The last DGL Node id value.
        :rtype: int
        """
        # 1. Map each ArangoDB _key to a DGL node id
        for adb_id in df["_key"]:
            adb_map[v_col][adb_id] = i
            i += 1

        # 2. Set the DGL Node Data
        self.__set_dgl_data(v_col, meta, ndata, df)

        return i

    def __process_adb_edge_df(
        self,
        _: int,
        df: DataFrame,
        e_col: str,
        adb_map: ADBMap,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        edata: DGLData,
        data_dict: DGLDataDict,
        v_cols: List[str],
        edge_type_blacklist: Set[DGLCanonicalEType],
    ) -> int:
        """ArangoDB -> DGL: Process the ArangoDB Edge DataFrame
        into the DGL EdgeData object.

        :param _: Not used.
        :type _: int
        :param df: The ArangoDB Edge DataFrame.
        :type df: pandas.DataFrame
        :param e_col: The ArangoDB Edge Collection.
        :type e_col: str
        :param adb_map: The ArangoDB -> DGL map.
        :type adb_map: adbdgl_adapter.typings.ADBMap
        :param meta: The metagraph for the current **e_col**.
        :type meta: Set[str] | Dict[str, ADBMetagraphValues]
        :param edata: The edge data view for storing edge features
        :type edata: adbdgl_adapter.typings.DGLData
        :param data_dict: The data for constructing a graph,
            which takes the form of (U, V).
            (U[i], V[i]) forms the edge with ID i in the graph.
        :type data_dict: adbdgl_adapter.typings.DGLDataDict
        :param v_cols: The list of ArangoDB Vertex Collections.
        :type v_cols: List[str]
        :param edge_type_blacklist: The set of skipped edge types
        :type edge_type_blacklist: Set[DGLCanonicalEType]
        :return: The last DGL Edge id value. This is a useless return value,
            but is needed for type hinting.
        :rtype: int
        """
        # 1. Split the ArangoDB _from & _to IDs into two columns
        df[["from_col", "from_key"]] = self.__split_adb_ids(df["_from"])
        df[["to_col", "to_key"]] = self.__split_adb_ids(df["_to"])

        # 2. Iterate over each edge type
        for (from_col, to_col), count in (
            df[["from_col", "to_col"]].value_counts().items()
        ):
            edge_type: DGLCanonicalEType = (from_col, e_col, to_col)

            # 3. Check for partial Edge Collection import
            if from_col not in v_cols or to_col not in v_cols:
                logger.debug(f"Skipping {edge_type}")
                edge_type_blacklist.add(edge_type)
                continue

            logger.debug(f"Preparing {count} {edge_type} edges")

            # 4. Get the edge data corresponding to the current edge type
            et_df = df[(df["from_col"] == from_col) & (df["to_col"] == to_col)]

            # 5. Map each ArangoDB from/to _key to the corresponding DGL node id
            from_nodes = et_df["from_key"].map(adb_map[from_col]).tolist()
            to_nodes = et_df["to_key"].map(adb_map[to_col]).tolist()

            # 6. Set/Update the DGL Edge Index
            if edge_type not in data_dict:
                data_dict[edge_type] = (tensor(from_nodes), tensor(to_nodes))
            else:
                previous_from_nodes, previous_to_nodes = data_dict[edge_type]
                data_dict[edge_type] = (
                    cat((previous_from_nodes, tensor(from_nodes))),
                    cat((previous_to_nodes, tensor(to_nodes))),
                )

            # 7. Set the DGL Edge Data
            self.__set_dgl_data(edge_type, meta, edata, df)

        return 1  # Useless return value, but needed for type hinting

    def __split_adb_ids(self, s: Series) -> Series:
        """AranogDB -> DGL: Helper method to split the ArangoDB IDs
        within a Series into two columns

        :param s: The Series containing the ArangoDB IDs.
        :type s: pandas.Series
        :return: A DataFrame with two columns: the ArangoDB Collection,
            and the ArangoDB _key.
        :rtype: pandas.Series
        """
        return s.str.split(pat="/", n=1, expand=True)

    def __set_dgl_data(
        self,
        data_type: DGLDataTypes,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        dgl_data: DGLData,
        df: DataFrame,
    ) -> None:
        """AranogDB -> DGL: A helper method to build the DGL NodeSpace or
        EdgeSpace object for the DGL graph. Is responsible for preparing the
        input **meta** such that it becomes a dictionary, and building DGL-ready
        tensors from the ArangoDB DataFrame **df**.

        :param data_type: The current node or edge type of the soon-to-be DGL graph.
        :type data_type: str | tuple[str, str, str]
        :param meta: The metagraph associated to the current ArangoDB vertex or
            edge collection. e.g metagraph['vertexCollections']['Users']
        :type meta: Set[str] |  Dict[str, adbdgl_adapter.typings.ADBMetagraphValues]
        :param dgl_data: The (currently empty) DefaultDict object storing the node or
            edge features of the soon-to-be DGL graph.
        :type dgl_data: adbdgl_adapter.typings.DGLData
        :param df: The DataFrame representing the ArangoDB collection data
        :type df: pandas.DataFrame
        """
        valid_meta: Dict[str, ADBMetagraphValues]
        valid_meta = meta if type(meta) is dict else {m: m for m in meta}

        for k, v in valid_meta.items():
            t = self.__build_tensor_from_dataframe(df, k, v)
            dgl_data[k][data_type] = cat((dgl_data[k][data_type], t))

    def __build_tensor_from_dataframe(
        self,
        adb_df: DataFrame,
        meta_key: str,
        meta_val: ADBMetagraphValues,
    ) -> Tensor:
        """AranogDB -> DGL: Constructs a DGL-ready Tensor from a Pandas
        Dataframe, based on the nature of the user-defined metagraph.

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
        logger.debug(
            f"__build_tensor_from_dataframe(df, '{meta_key}', {type(meta_val)})"
        )

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

    def __create_dgl_graph(
        self, data_dict: DGLDataDict, adb_map: ADBMap, metagraph: ADBMetagraph
    ) -> Union[DGLGraph, DGLHeteroGraph]:
        """AranogDB -> DGL: Creates a DGL graph from the given DGL data.

        :param data_dict: The data for constructing a graph,
            which takes the form of (U, V).
            (U[i], V[i]) forms the edge with ID i in the graph.
        :type data_dict: adbdgl_adapter.typings.DGLDataDict
        :param adb_map: A mapping of ArangoDB IDs to DGL IDs.
        :type adb_map: adbdgl_adapter.typings.ADBMap
        :param metagraph: The ArangoDB metagraph.
        :type metagraph: adbdgl_adapter.typings.ADBMetagraph
        :return: A DGL Homogeneous or Heterogeneous graph object
        :rtype: dgl.DGLGraph | dgl.DGLHeteroGraph
        """
        is_homogeneous = (
            len(metagraph["vertexCollections"]) == 1
            and len(metagraph["edgeCollections"]) == 1
        )

        if is_homogeneous:
            v_col = next(iter(metagraph["vertexCollections"]))
            data = next(iter(data_dict.values()))

            return graph(data, num_nodes=len(adb_map[v_col]))

        num_nodes_dict = {v_col: len(adb_map[v_col]) for v_col in adb_map}
        return heterograph(data_dict, num_nodes_dict)

    def __link_dgl_data(
        self,
        dgl_data: Union[HeteroNodeDataView, HeteroEdgeDataView],
        dgl_data_temp: DGLData,
        has_one_type: bool,
    ) -> None:
        """Links **dgl_data_temp** to **dgl_data**. This method is (unfortunately)
            required, since a dgl graph's `ndata` and `edata` properties can't be
            manually set (i.e `g.ndata = ndata` is not possible).

        :param dgl_data: The (empty) ndata or edata instance attribute of a dgl graph,
            which is about to receive **dgl_data_temp**.
        :type dgl_data: Union[dgl.view.HeteroNodeDataView, dgl.view.HeteroEdgeDataView]
        :param dgl_data_temp: A temporary place to store the ndata or edata features.
        :type dgl_data_temp: adbdgl_adapter.typings.DGLData
        :param has_one_type: Set to True if the DGL graph only has one
            node type or edge type.
        :type has_one_type: bool
        """
        for feature_name, feature_map in dgl_data_temp.items():
            for data_type, dgl_tensor in feature_map.items():
                dgl_data[feature_name] = (
                    dgl_tensor if has_one_type else {data_type: dgl_tensor}
                )

    ############################
    # Private: DGL -> ArangoDB #
    ############################

    def __get_node_and_edge_types(
        self,
        name: str,
        dgl_g: DGLGraph,
        metagraph: DGLMetagraph,
        is_explicit_metagraph: bool,
    ) -> Tuple[List[str], List[DGLCanonicalEType]]:
        """DGL -> ArangoDB: Returns the node & edge types of the DGL graph,
        based on the metagraph and whether the graph has default canonical etypes.

        :param name: The DGL graph name.
        :type name: str
        :param dgl_g: The existing DGL graph.
        :type dgl_g: dgl.DGLGraph
        :param metagraph: The DGL Metagraph.
        :type metagraph: adbdgl_adapter.typings.DGLMetagraph
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :return: The node & edge types of the DGL graph.
        :rtype: Tuple[List[str], List[adbdgl_adapter.typings.DGLCanonicalEType]]
        """
        node_types: List[str]
        edge_types: List[DGLCanonicalEType]

        has_default_canonical_etypes = dgl_g.canonical_etypes == [("_N", "_E", "_N")]

        if is_explicit_metagraph:
            node_types = metagraph.get("nodeTypes", {}).keys()  # type: ignore
            edge_types = metagraph.get("edgeTypes", {}).keys()  # type: ignore

        elif has_default_canonical_etypes:
            n_type = name + "_N"
            node_types = [n_type]
            edge_types = [(n_type, name + "_E", n_type)]

        else:
            node_types = dgl_g.ntypes
            edge_types = dgl_g.canonical_etypes

        return node_types, edge_types

    def __etypes_to_edefinitions(
        self, edge_types: List[DGLCanonicalEType]
    ) -> List[Json]:
        """Converts DGL canonical_etypes to ArangoDB edge_definitions

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

    def __ntypes_to_ocollections(
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

    def __create_adb_graph(
        self,
        name: str,
        overwrite_graph: bool,
        node_types: List[str],
        edge_types: List[DGLCanonicalEType],
    ) -> ADBGraph:
        """Creates an ArangoDB graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections. Defaults to False.
        :type overwrite_graph: bool
        :param node_types: A list of strings representing the DGL node types.
        :type node_types: List[str]
        :param edge_types: A list of string triplets (str, str, str) for
            source node type, edge type and destination node type.
        :type edge_types: List[adbdgl_adapter.typings.DGLCanonicalEType]
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        """
        if overwrite_graph:
            logger.debug("Overwrite graph flag is True. Deleting old graph.")
            self.__db.delete_graph(name, ignore_missing=True)

        if self.__db.has_graph(name):
            return self.__db.graph(name)

        edge_definitions = self.__etypes_to_edefinitions(edge_types)
        orphan_collections = self.__ntypes_to_ocollections(node_types, edge_types)

        return self.__db.create_graph(
            name,
            edge_definitions,
            orphan_collections,
        )

    def __process_dgl_node_batch(
        self,
        n_type: str,
        ndata: NodeSpace,
        ndata_size: int,
        meta: Union[Set[str], Dict[Any, DGLMetagraphValues]],
        is_explicit_metagraph: bool,
        is_custom_controller: bool,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """DGL -> ArangoDB: Processes the DGL Node batch
        into an ArangoDB DataFrame.

        :param n_type: The DGL node type.
        :type n_type: str
        :param ndata: The DGL Node Space for the current **n_type**.
        :type ndata: dgl.view.NodeSpace
        :param ndata_size: The size of **ndata**.
        :param ndata_size: int
        :param meta: The metagraph for the current **n_type**.
        :type meta: Set[str] | Dict[Any, adbdgl_adapter.typings.DGLMetagraphValues]
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :param is_custom_controller: Whether a custom controller is used.
        :type is_custom_controller: bool
        :param start_index: The start index of the current batch.
        :type start_index: int
        :param end_index: The end index of the current batch.
        :type end_index: int
        :return: The ArangoDB DataFrame representing the DGL Node batch.
        :rtype: pandas.DataFrame
        """
        # 1. Map each DGL node id to an ArangoDB _key
        adb_keys = [{"_key": str(i)} for i in range(start_index, end_index)]

        # 2. Set the ArangoDB Node Data
        df = self.__set_adb_data(
            DataFrame(adb_keys, index=range(start_index, end_index)),
            meta,
            ndata,
            ndata_size,
            is_explicit_metagraph,
            start_index,
            end_index,
        )

        # 3. Apply the ArangoDB Node Controller (if provided)
        if is_custom_controller:
            df = df.apply(lambda n: self.__cntrl._prepare_dgl_node(n, n_type), axis=1)

        return df

    def __process_dgl_edge_batch(
        self,
        e_type: DGLCanonicalEType,
        edata: EdgeSpace,
        edata_size: int,
        meta: Union[Set[str], Dict[Any, DGLMetagraphValues]],
        from_nodes: Tensor,
        to_nodes: Tensor,
        is_explicit_metagraph: bool,
        is_custom_controller: bool,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """DGL -> ArangoDB: Processes the DGL Edge batch
        into an ArangoDB DataFrame.

        :param e_type: The DGL edge type.
        :type e_type: adbdgl_adapter.typings.DGLCanonicalEType
        :param edata: The DGL EdgeSpace for the current **e_type**.
        :type edata: dgl.view.EdgeSpace
        :param edata_size: The size of **edata**.
        :param edata_size: int
        :param meta: The metagraph for the current **e_type**.
        :type meta: Set[str] | Dict[Any, adbdgl_adapter.typings.DGLMetagraphValues]
        :param from_nodes: Tensor representing the Source Nodes of the **e_type**.
        :type from_nodes: torch.Tensor
        :param to_nodes: Tensor representing the Destination Nodes of the **e_type**.
        :type to_nodes: torch.Tensor
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :param is_custom_controller: Whether a custom controller is used.
        :type is_custom_controller: bool
        :param start_index: The start index of the current batch.
        :type start_index: int
        :param end_index: The end index of the current batch.
        :type end_index: int
        :return: The ArangoDB DataFrame representing the DGL Edge batch.
        :rtype: pandas.DataFrame
        """
        from_col, _, to_col = e_type

        # 1. Map the DGL edges to ArangoDB _from & _to IDs
        data = zip(
            *(
                from_nodes[start_index:end_index].tolist(),
                to_nodes[start_index:end_index].tolist(),
            )
        )

        # 2. Set the ArangoDB Edge Data
        df = self.__set_adb_data(
            DataFrame(
                data,
                index=range(start_index, end_index),
                columns=["_from", "_to"],
            ),
            meta,
            edata,
            edata_size,
            is_explicit_metagraph,
            start_index,
            end_index,
        )

        df["_from"] = from_col + "/" + df["_from"].astype(str)
        df["_to"] = to_col + "/" + df["_to"].astype(str)

        # 3. Apply the ArangoDB Edge Controller (if provided)
        if is_custom_controller:
            df = df.apply(lambda e: self.__cntrl._prepare_dgl_edge(e, e_type), axis=1)

        return df

    def __set_adb_data(
        self,
        df: DataFrame,
        meta: Union[Set[str], Dict[Any, DGLMetagraphValues]],
        dgl_data: Union[NodeSpace, EdgeSpace],
        dgl_data_size: int,
        is_explicit_metagraph: bool,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """A helper method to build the ArangoDB Dataframe for the given
        collection. Is responsible for creating "sub-DataFrames" from DGL tensors,
        and appending them to the main dataframe **df**. If the data
        does not adhere to the supported types, or is not of specific length,
        then it is silently skipped.

        :param df: The main ArangoDB DataFrame containing (at minimum)
            the vertex/edge _id or _key attribute.
        :type df: pandas.DataFrame
        :param meta: The metagraph associated to the
            current DGL node or edge type. e.g metagraph['nodeTypes']['v0']
        :type meta: Set[str] | Dict[Any, adbdgl_adapter.typings.DGLMetagraphValues]
        :param dgl_data: The NodeSpace or EdgeSpace of the current
            DGL node or edge type.
        :type dgl_data: dgl.view.(NodeSpace | EdgeSpace)
        :param dgl_data_size: The size of the NodeStorage or EdgeStorage of the
            current DGL node or edge type.
        :type dgl_data_size: int
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :param start_index: The starting index of the current batch to process.
        :type start_index: int
        :param end_index: The ending index of the current batch to process.
        :type end_index: int
        :return: The completed DataFrame for the (soon-to-be) ArangoDB collection.
        :rtype: pandas.DataFrame
        :raise ValueError: If an unsupported DGL data value is found.
        """
        logger.debug(
            f"__set_adb_data(df, {meta}, {type(dgl_data)}, {is_explicit_metagraph}"
        )

        valid_meta: Dict[Any, DGLMetagraphValues]
        valid_meta = meta if type(meta) is dict else {m: m for m in meta}

        dgl_keys = set(valid_meta.keys()) if is_explicit_metagraph else dgl_data.keys()
        for meta_key in dgl_keys:
            data = dgl_data[meta_key]
            meta_val = valid_meta.get(meta_key, str(meta_key))

            if type(data) is Tensor and len(data) == dgl_data_size:
                df = df.join(
                    self.__build_dataframe_from_tensor(
                        data[start_index:end_index],
                        start_index,
                        end_index,
                        meta_key,
                        meta_val,
                    )
                )

        return df

    def __build_dataframe_from_tensor(
        self,
        dgl_tensor: Tensor,
        start_index: int,
        end_index: int,
        meta_key: Any,
        meta_val: DGLMetagraphValues,
    ) -> DataFrame:
        """Builds a Pandas DataFrame from DGL Tensor, based on
        the nature of the user-defined metagraph.

        :param dgl_tensor: The Tensor representing DGL data.
        :type dgl_tensor: torch.Tensor
        :param meta_key: The current DGL-ArangoDB metagraph key
        :type meta_key: Any
        :param meta_val: The value mapped to the DGL-ArangoDB metagraph key to
            help convert **tensor** into a Pandas Dataframe.
            e.g the value of `metagraph['nodeTypes']['users']['x']`.
        :type meta_val: adbdgl_adapter.typings.DGLMetagraphValues
        :return: A Pandas DataFrame equivalent to the Tensor
        :rtype: pandas.DataFrame
        :raise adbdgl_adapter.exceptions.DGLMetagraphError: If invalid **meta_val**.
        """
        logger.debug(
            f"__build_dataframe_from_tensor(df, '{meta_key}', {type(meta_val)})"
        )

        if type(meta_val) is str:
            df = DataFrame(index=range(start_index, end_index), columns=[meta_val])
            df[meta_val] = dgl_tensor.tolist()
            return df

        if type(meta_val) is list:
            num_features = dgl_tensor.size()[-1]
            if len(meta_val) != num_features:  # pragma: no cover
                msg = f"""
                    Invalid list length for **meta_val** ('{meta_key}'):
                    List length must match the number of
                    features found in the tensor ({num_features}).
                """
                raise DGLMetagraphError(msg)

            df = DataFrame(index=range(start_index, end_index), columns=meta_val)
            df[meta_val] = dgl_tensor.tolist()
            return df

        if callable(meta_val):
            # **meta_val** is a user-defined function that populates
            # and returns the empty dataframe
            empty_df = DataFrame(index=range(start_index, end_index))
            user_defined_result = meta_val(dgl_tensor, empty_df)

            if not isinstance(user_defined_result, DataFrame):  # pragma: no cover
                msg = f"""
                    Invalid return type for function {meta_val} ('{meta_key}').
                    Function must return Pandas DataFrame.
                """
                raise DGLMetagraphError(msg)

            if (
                user_defined_result.index.start != start_index
                or user_defined_result.index.stop != end_index
            ):  # pragma: no cover
                msg = f"""
                    User Defined Function {meta_val} ('{meta_key}') must return
                    DataFrame with start index {start_index} & stop index {end_index}
                """
                raise DGLMetagraphError(msg)

            return user_defined_result

        raise DGLMetagraphError(f"Invalid {meta_val} type")  # pragma: no cover

    def __insert_adb_docs(
        self,
        spinner_progress: Progress,
        df: DataFrame,
        col: str,
        use_async: bool,
        **adb_import_kwargs: Any,
    ) -> None:
        """DGL -> ArangoDB: Insert ArangoDB documents into their ArangoDB collection.

        :param spinner_progress: The spinner progress bar.
        :type spinner_progress: rich.progress.Progress
        :param df: To-be-inserted ArangoDB documents, formatted as a DataFrame
        :type df: pandas.DataFrame
        :param col: The ArangoDB collection name.
        :type col: str
        :param use_async: Performs asynchronous ArangoDB ingestion if enabled.
        :type use_async: bool
        :param adb_import_kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :param adb_import_kwargs: Any
        """
        action = f"ADB Import: '{col}' ({len(df)})"
        spinner_progress_task = spinner_progress.add_task("", action=action)

        docs = df.to_dict("records")
        db = self.__async_db if use_async else self.__db
        result = db.collection(col).import_bulk(docs, **adb_import_kwargs)
        logger.debug(result)

        df.drop(df.index, inplace=True)

        spinner_progress.stop_task(spinner_progress_task)
        spinner_progress.update(spinner_progress_task, visible=False)
