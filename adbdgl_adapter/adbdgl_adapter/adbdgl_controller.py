#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union
from .abc import ADBDGL_Controller
from collections import defaultdict

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
        self.lambda_map = defaultdict(lambda: defaultdict(int))

    def attribute_to_feature(self, col: str, atrib: str, val):
        """
        Given an ArangoDB attribute, its value (for that document),
        and the collection it belongs to, convert it to a valid
        DGL feature: https://docs.dgl.ai/en/0.6.x/guide/graph-feature.html.
        TLDR: DGL only accepts 'attributes' (a.k.a features) of numerical types.

        """
        if type(val) in [int, float]:
            return val

        if atrib == "Sex":
            return 0 if val == "M" else 1

        if atrib == "Ssn":
            return int(str(val).replace("-", ""))

        try:
            res = float(val)
            return res
        except:
            return 0

    def _arangodb_key_to_integer(self, key: str) -> int:
        """Given an ArangoDB _key, derive its representing integer.

        If the string only contains numbers, convert to int.
        Else, convert string to ascii code.

        :param string: The ArangoDB _key
        :type string: str
        :return: An int
        :rtype: int
        """
        try:
            return int(key)
        except ValueError:
            return int("".join([str(ord(c)).zfill(3) for c in key]))
