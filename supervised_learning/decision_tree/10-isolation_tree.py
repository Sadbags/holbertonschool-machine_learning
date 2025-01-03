#!/usr/bin/env python3
""" Module for the Isolation_Random_Tree class. """
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """ This class defines an Isolation Random Tree. """

    def __init__(self, max_depth=10, seed=0, root=None):
        """ Initializes the isolation tree. """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """ Returns a string representation of the tree. """
        return self.root.__str__()

    def depth(self):
        """ Returns the depth of the tree. """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Returns the number of nodes in the tree. """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """ Returns all the leaves in the tree. """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Updates the bounds of the tree. """
        self.root.update_bounds_below()

    def update_predict(self):
        """ Updates the prediction function of the tree. """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([self.root.pred(x) for x in A])

    def np_extrema(self, arr):
        """ Returns the min and max of an array. """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ Generates a random split criterion for a node. """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """ Creates and returns a leaf child. """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ Creates and returns a node child. """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """ Recursively fits a node and its children. """
        node.feature, node.threshold = self.random_split_criterion(node)

        above_threshold = self.explanatory[:, node.feature] > node.threshold
        left_population = node.sub_population & above_threshold
        right_population = node.sub_population & ~above_threshold

        # Check if the left or right node should be a leaf
        is_left_leaf = np.any([
            node.depth >= self.max_depth - 1,
            np.sum(left_population) <= self.min_pop
        ])

        # Create left child
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Check if the right node should be a leaf
        is_right_leaf = np.any([
            node.depth >= self.max_depth - 1,
            np.sum(right_population) <= self.min_pop
        ])

        # Create right child
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """ Model to the training data. """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(explanatory.shape[0],
                                                dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
