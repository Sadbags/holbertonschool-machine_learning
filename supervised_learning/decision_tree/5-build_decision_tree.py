#!/usr/bin/env python3
"""Method for max depth below."""

import numpy as np


class Node:
    """Class representing a node in a decision tree."""

    def __init__(
        self, feature=None, threshold=None, left_child=None,
        right_child=None, is_root=False, depth=0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth below this node."""
        if self.is_leaf:
            return self.depth
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below this node."""
        if only_leaves:
            count = self.left_child.count_nodes_below(only_leaves=True)
            count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        count = 1 + self.left_child.count_nodes_below()
        count += self.right_child.count_nodes_below()
        return count

    def __str__(self):
        """Method that returns the string
        representation of the current node """
        node_str = (
            f"root [feature={self.feature}, threshold={self.threshold}]\n"
            if self.is_root else
            f"-> node [feature={self.feature}, "
            f"threshold={self.threshold}]\n"
        )

        # If the node is a leaf, simply return the string representation
        if self.is_leaf:
            return node_str

        # Formatting for the left and right children
        left_str = self.left_child_add_prefix(
            self.left_child.__str__()) if self.left_child else ""
        right_str = self.right_child_add_prefix(
            self.right_child.__str__()) if self.right_child else ""

        return node_str + left_str + right_str

    def left_child_add_prefix(self, text):
        """ Add prefix to the left child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def right_child_add_prefix(self, text):
        """ Add prefix to the right child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["     " + "  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def get_leaves_below(self):
        """ returns the list of all leaves of the tree. """
        if self.is_leaf:
            return [self]
        return (
            self.left_child.get_leaves_below() +
            self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """ updates the bounds of the tree"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = max(
                        child.lower.get(self.feature, -np.inf), self.threshold)
                else:  # right child
                    child.upper[self.feature] = min(
                        child.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Update the indicator function for the leaves below
        the current node."""

        def is_large_enough(x):
            """Check if the input is large enough."""
            lower_bounds = np.array([self.lower.get(i, -np.inf)
                                     for i in range(x.shape[1])])
            return np.all(x >= lower_bounds, axis=1)

        def is_small_enough(x):
            """Check if the input is small enough."""
            upper_bounds = np.array([self.upper.get(i, np.inf)
                                     for i in range(x.shape[1])])
            return np.all(x <= upper_bounds, axis=1)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """ Method that predicts the value of a sample """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Class representing a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 as a leaf counts as one node."""
        return 1

    def __str__(self):
        """ returns the string """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """ returns leaves """
        return [self]

    def update_bounds_below(self):
        """ updates the bounds of the tree """
        pass

    def pred(self, x):
        """ predicts the value """
        return self.value


class Decision_Tree():
    """Class representing a decision tree."""

    def __init__(
        self, max_depth=10, min_pop=1, seed=0,
        split_criterion="random", root=None
    ):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the nodes in the tree, optionally only counting leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ returns the string """
        return self.root.__str__()

    def get_leaves(self):
        """ returns the leaves """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ updates the bounds of the tree """
        self.root.update_bounds_below()

    def pred(self, x):
        """ predicts value of a sample """
        return self.root.pred(x)
