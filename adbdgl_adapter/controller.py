#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

from torch.functional import Tensor

from .abc import Abstract_ADBDGL_Controller

"""

@author: Anthony Mahanna
"""


class ADBDGL_Controller(Abstract_ADBDGL_Controller):
    """ArangoDB-DGL controller.

    Responsible for controlling how ArangoDB attributes
    are converted into DGL features, and vice-versa.

    You can derive your own custom ADBDGL_Controller if you want to maintain
    consistency between your ArangoDB attributes & your DGL features.
    """

    def _adb_attribute_to_dgl_feature(self, key: str, col: str, val: Any) -> Any:
        """
        Given an ArangoDB attribute key, its assigned value (for an arbitrary document),
        and the collection it belongs to, convert it to a valid
        DGL feature: https://docs.dgl.ai/en/0.6.x/guide/graph-feature.html.

        NOTE: You must override this function if you want to transfer non-numerical
        ArangoDB attributes to DGL (DGL only accepts 'attributes' (a.k.a features)
        of numerical types). Read more about DGL features here:
        https://docs.dgl.ai/en/0.6.x/new-tutorial/2_dglgraph.html#assigning-node-and-edge-features-to-graph.
        """
        try:
            return float(val)
        except (ValueError, TypeError, SyntaxError):
            return 0

    def _dgl_feature_to_adb_attribute(self, key: str, col: str, val: Tensor) -> Any:
        """
        Given a DGL feature key, its assigned value (for an arbitrary node or edge),
        and the collection it belongs to, convert it to a valid ArangoDB attribute
        (e.g string, list, number, ...).

        NOTE: No action is needed here if you want to keep the numerical-based values
        of your DGL features.
        """
        try:
            return val.item()
        except ValueError:
            return val.tolist()
