
# The ‘sum-tree’ data structure used here is very similar in spirit to the array representation
# of a binary heap. However, instead of the usual heap property, the value of a parent node is
# the sum of its children. Leaf nodes store the transition priorities and the internal nodes are
# intermediate sums, with the parent node containing the sum over all priorities, p_total. This
# provides a efficient way of calculating the cumulative sum of priorities, allowing O(log N) updates
# and sampling. (Appendix B.2.1, Proportional prioritization)

# Additional useful links
# Good tutorial about SumTree data structure:  https://adventuresinmachinelearning.com/sumtree-introduction-python/
# How to represent full binary tree as array: https://stackoverflow.com/questions/8256222/binary-tree-represented-using-array

import math
import numpy as np

class SumTree(object):
    def __init__(self, size, epsilon):
        # epsilon here is the minimum prob to avoid not being chosen for some rare actions
        self.nodes = [epsilon] * (2 * size - 1)
        self.data = [None] * size
        
        self.size = size
        self.count = 0
        self.real_size = 0
        self.n_rows = math.ceil(self.size / 2) + 1

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        """
        This method replace the current data_idx node, and replace it
        with a new node with value assigned in value
        """

        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value
    
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total, print(cumsum, ">", self.total)

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()}, n_rows={self.n_rows.__repr__()})"
    
    def get_nodes(self):
        return self.nodes
    
    def get_leaf_nodes_properties(self):
        # get the lowest and highest node indicies
        leaf_nodes = self.nodes[-len(self.data):]
        print(leaf_nodes)
        quit()
        return np.argmin(leaf_nodes), np.argmax(leaf_nodes)
    
    def replace_data(self, index, data):
        n_none_leaf_nodes = self.size - self.real_size
        self.data[n_none_leaf_nodes + index] = data

