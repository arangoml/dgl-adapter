import logging
import os
from typing import Any, Dict, Set, Union

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .exceptions import ADBMetagraphError, DGLMetagraphError

logger = logging.getLogger(__package__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    f"[%(asctime)s] [{os.getpid()}] [%(levelname)s] - %(name)s: %(message)s",
    "%Y/%m/%d %H:%M:%S %z",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_export_spinner_progress(
    text: str,
) -> Progress:
    return Progress(
        TextColumn(text),
        SpinnerColumn("aesthetic", "#5BC0DE"),
        TimeElapsedColumn(),
        transient=True,
    )


def get_import_spinner_progress(text: str) -> Progress:
    return Progress(
        TextColumn(text),
        TextColumn("{task.fields[action]}"),
        SpinnerColumn("aesthetic", "#5BC0DE"),
        TimeElapsedColumn(),
        transient=True,
    )


def get_bar_progress(text: str, color: str) -> Progress:
    return Progress(
        TextColumn(text),
        BarColumn(complete_style=color, finished_style=color),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    )


def validate_adb_metagraph(metagraph: Dict[Any, Dict[Any, Any]]) -> None:
    meta: Union[Set[Any], Dict[Any, Any]]

    if "vertexCollections" not in metagraph:
        raise ADBMetagraphError("Missing 'vertexCollections' key in metagraph")

    if "edgeCollections" not in metagraph:
        raise ADBMetagraphError("Missing 'edgeCollections' key in metagraph")

    for parent_key in ["vertexCollections", "edgeCollections"]:
        sub_metagraph = metagraph[parent_key]
        if not sub_metagraph or type(sub_metagraph) != dict:
            raise ADBMetagraphError(f"{parent_key} must map to non-empty dictionary")

        for col, meta in sub_metagraph.items():
            if type(col) != str:
                msg = f"""
                    Invalid {parent_key} sub-key type:
                    {col} must be str
                """
                raise ADBMetagraphError(msg)

            if type(meta) == set:
                for m in meta:
                    if type(m) != str:
                        msg = f"""
                            Invalid set value type for {meta}:
                            {m} must be str
                        """
                        raise ADBMetagraphError(msg)

            elif type(meta) == dict:
                for meta_key, meta_val in meta.items():
                    if type(meta_key) != str:
                        msg = f"""
                            Invalid key type in {meta}:
                            {meta_key} must be str
                        """
                        raise ADBMetagraphError(msg)

                    if type(meta_val) not in [str, dict] and not callable(meta_val):
                        msg = f"""
                            Invalid mapped value type in {meta}:
                            {meta_val} must be
                                str | Dict[str, None | Callable] | Callable
                        """
                        raise ADBMetagraphError(msg)

                    if type(meta_val) == dict:
                        for k, v in meta_val.items():
                            if type(k) != str:
                                msg = f"""
                                    Invalid ArangoDB attribute key type:
                                    {v} must be str
                                """
                                raise ADBMetagraphError(msg)

                            if v is not None and not callable(v):
                                msg = f"""
                                    Invalid DGL Encoder type:
                                    {v} must be None | Callable
                                """
                                raise ADBMetagraphError(msg)
            else:
                msg = f"""
                    Invalid mapped value type for {col}:
                    {meta} must be dict | set
                """
                raise ADBMetagraphError(msg)


def validate_dgl_metagraph(metagraph: Dict[Any, Dict[Any, Any]]) -> None:
    meta: Union[Set[Any], Dict[Any, Any]]

    for node_type in metagraph.get("nodeTypes", {}).keys():
        if type(node_type) != str:
            msg = f"Invalid nodeTypes sub-key: {node_type} is not str"
            raise DGLMetagraphError(msg)

    for edge_type in metagraph.get("edgeTypes", {}).keys():
        if type(edge_type) != tuple:
            msg = f"Invalid edgeTypes sub-key: {edge_type} must be Tuple[str, str, str]"
            raise DGLMetagraphError(msg)
        else:
            for elem in edge_type:
                if type(elem) != str:
                    msg = f"{elem} in {edge_type} must be str"
                    raise DGLMetagraphError(msg)

    for parent_key in ["nodeTypes", "edgeTypes"]:
        for k, meta in metagraph.get(parent_key, {}).items():
            if type(meta) == set:
                for m in meta:
                    if type(m) != str:
                        msg = f"""
                            Invalid set value type for {meta}:
                            {m} must be str
                        """
                        raise DGLMetagraphError(msg)

            elif type(meta) == dict:
                for meta_val in meta.values():
                    if type(meta_val) not in [str, list] and not callable(meta_val):
                        msg = f"""
                            Invalid mapped value type in {meta}:
                            {meta_val} must be str | List[str] | Callable
                        """
                        raise DGLMetagraphError(msg)

                    if type(meta_val) == list:
                        for v in meta_val:
                            if type(v) != str:
                                msg = f"""
                                    Invalid ArangoDB attribute key type:
                                    {v} must be str
                                """
                                raise DGLMetagraphError(msg)
            else:
                msg = f"""
                    Invalid mapped value type for {k}:
                    {meta} must be dict | set
                """
                raise DGLMetagraphError(msg)
