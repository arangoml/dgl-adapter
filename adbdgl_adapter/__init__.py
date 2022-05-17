import logging
import os

from adbdgl_adapter.adapter import ADBDGL_Adapter  # noqa: F401
from adbdgl_adapter.controller import ADBDGL_Controller  # noqa: F401

logger = logging.getLogger(__package__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    f"[%(asctime)s] [{os.getpid()}] [%(levelname)s] - %(name)s: %(message)s",
    "%Y/%m/%d %H:%M:%S %z",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
