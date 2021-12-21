#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .abc import ADBDGL_Controller
from collections import defaultdict
from torch.functional import Tensor

"""

@author: Anthony Mahanna
"""


class Base_ADBDGL_Controller(ADBDGL_Controller):
    """ArangoDB-DGL controller.

    Responsible for controlling how ArangoDB attributes
    are converted into DGL features, and vice-versa.
    """

    def _adb_attribute_to_dgl_feature(self, key: str, col: str, val):
        """
        Given an ArangoDB attribute key, its assigned value (for an arbitrary document),
        and the collection it belongs to, convert it to a valid
        DGL feature: https://docs.dgl.ai/en/0.6.x/guide/graph-feature.html.

        NOTE: You must override this function if you want to transfer non-numerical ArangoDB
        attributes to DGL (DGL only accepts 'attributes' (a.k.a features) of numerical types).
        Read more about DGL features here: https://docs.dgl.ai/en/0.6.x/new-tutorial/2_dglgraph.html#assigning-node-and-edge-features-to-graph.
        """
        try:
            return float(val)
        except:
            return 0

    def _dgl_feature_to_adb_attribute(self, key: str, col: str, val: Tensor):
        """
        Given a DGL feature key, its assigned value (for an arbitrary node or edge),
        and the collection it belongs to, convert it to a valid ArangoDB attribute (e.g string, list, number, ...).

        NOTE: No action is needed here if you want to keep the numerical-based values of your DGL features.
        """
        try:
            return val.item()
        except ValueError:
            print("HERERERERE")
            return val.tolist()
