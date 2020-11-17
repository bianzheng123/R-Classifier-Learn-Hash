from model.learn_on_graph.graph.hnsw_graph.select_neighbor import base_algorithm
import numpy as np
import heapq


class SimpleAlgorithm(base_algorithm.BaseAlgorithm):
    def __init__(self, config):
        super(SimpleAlgorithm, self).__init__(config)
        # self.k_graph

    '''
    选出距离insert_point最近的k_graph个节点
    '''

    def select_neighbors(self, graph, base, candidate_elements, insert_point_idx):
        while len(candidate_elements) > self.k_graph:
            heapq.heappop(candidate_elements)
        return candidate_elements
