import logging
import os
from typing import Any, Dict

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .exceptions import ADBMetagraphError, DGLMetagraphError

logger = logging.getLogger(__package__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    f"[%(asctime)s] [{os.getpid()}] [%(levelname)s] - %(name)s: %(message)s",
    "%Y/%m/%d %H:%M:%S %z",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def progress(
    text: str,
    text_style: str = "none",
    spinner_name: str = "aesthetic",
    spinner_style: str = "#5BC0DE",
    transient: bool = False,
) -> Progress:
    return Progress(
        TextColumn(text, style=text_style),
        SpinnerColumn(spinner_name, spinner_style),
        TimeElapsedColumn(),
        transient=transient,
    )


def validate_adb_metagraph(metagraph: Dict[Any, Dict[Any, Any]]) -> None:
    meta: Dict[Any, Any]

    if not metagraph.get("vertexCollections"):
        raise ADBMetagraphError("Missing 'vertexCollections' key in metagraph")

    if not metagraph.get("edgeCollections"):
        raise ADBMetagraphError("Missing 'edgeCollections' key in metagraph")

    for parent_key in ["vertexCollections", "edgeCollections"]:
        for col, meta in metagraph[parent_key].items():
            if type(col) != str:
                msg = f"Invalid {parent_key} sub-key type: {col} must be str"
                raise ADBMetagraphError(msg)

            for meta_val in meta.values():
                if type(meta_val) not in [str, dict] and not callable(meta_val):
                    msg = f"""
                        Invalid mapped value type in {meta}:
                        {meta_val} must be str | Dict[str, None | Callable] | Callable
                    """
                    raise ADBMetagraphError(msg)

                if type(meta_val) == dict:
                    for k, v in meta_val.items():
                        if type(k) != str:
                            msg = f"""
                                Invalid ArangoDB attribute key type: {v} must be str
                            """
                            raise ADBMetagraphError(msg)

                        if v is not None and not callable(v):
                            msg = f"""
                                Invalid DGL Encoder type: {v} must be None | Callable
                            """
                            raise ADBMetagraphError(msg)


def validate_dgl_metagraph(metagraph: Dict[Any, Dict[Any, Any]]) -> None:
    meta: Dict[Any, Any]

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
        for meta in metagraph.get(parent_key, {}).values():
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
