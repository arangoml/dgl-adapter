#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any

from torch.functional import Tensor

from .abc import Abstract_ADBDGL_Controller


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

        :param key: The ArangoDB attribute key name
        :type key: str
        :param col: The ArangoDB collection of the ArangoDB document.
        :type col: str
        :param val: The assigned attribute value of the ArangoDB document.
        :type val: Any
        :return: The attribute's representation as a DGL Feature
        :rtype: Any
        """
        if type(val) in [int, float, bool]:
            return val

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

        :param key: The DGL attribute key name
        :type key: str
        :param col: The ArangoDB collection of the (soon-to-be) ArangoDB document.
        :type col: str
        :param val: The assigned attribute value of the DGL node.
        :type val: Tensor
        :return: The feature's representation as an ArangoDB Attribute
        :rtype: Any
        """
        try:
            return val.item()
        except ValueError:
            return val.tolist()
