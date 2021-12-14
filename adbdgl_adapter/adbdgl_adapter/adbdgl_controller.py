#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

    def adb_attribute_to_dgl_feature(self, col: str, atrib: str, val):
        """
        Given an ArangoDB attribute * its value (for that document),
        and the collection it belongs to, convert it to a valid
        DGL feature: https://docs.dgl.ai/en/0.6.x/guide/graph-feature.html.
        TLDR: DGL only accepts 'attributes' (a.k.a features) of numerical types.
        """
        try:
            return float(val)
        except:
            return 0

        # if atrib == "Sex":
        #     return 0 if val == "M" else 1

        # if atrib == "Ssn":
        #     return int(str(val).replace("-", ""))

    def dgl_feature_to_adb_attribute(self, col: str, feature: str, val):
        """
        Given a DGL feature & its value (for that node or edge), convert it to a valid
        ArangoDB attribute. Not much is needed here if you want to keep the numerical-based
        values of DGL features.
        """
        if not val:
            return 0
        return val

    # def _adb_key_to_int(self, key: str) -> int:
    #     """Given an ArangoDB _key, derive its representing integer.

    #     If the string only contains numbers, convert to int.
    #     Else, convert string to ascii code.

    #     :param string: The ArangoDB _key
    #     :type string: str
    #     :return: An int
    #     :rtype: int
    #     """
    #     try:
    #         return int(key)
    #     except ValueError:
    #         return int("".join([str(ord(c)).zfill(3) for c in key]))
