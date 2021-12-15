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

        NOTE: DGL only accepts 'attributes' (a.k.a features) of numerical types.
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
        if not val or type(val) is not Tensor:
            return 0

        try:
            return val.item()
        except ValueError:
            return val.tolist()
