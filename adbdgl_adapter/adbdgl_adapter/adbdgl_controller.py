#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .abc import ADBDGL_Controller

"""

@author: Anthony Mahanna
"""


class Base_ADBDGL_Controller(ADBDGL_Controller):
    """ArangoDB-DGL controller.

    Responsible for controlling how nodes & edges are handled when
    transitioning from ArangoDB to DGL, and vice-versa.
    """

    def __init__(self):
        self.dgl_map = dict()  # Maps DGL node IDs to ArangoDB vertex IDs
        self.adb_map = dict()  # Maps ArangoDB vertex IDs to DGL node IDs

    def extract_dgl_atribute(self, val, atrib: str):
        if atrib == "Sex":
            return 0 if val == "M" else 1

        if atrib == "Ssn":
            return int(str(val).replace("-", ""))

        try:
            res = float(val)
            return res
        except:
            return 0
