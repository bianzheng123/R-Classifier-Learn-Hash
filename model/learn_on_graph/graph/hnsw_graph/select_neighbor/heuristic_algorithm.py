from model.learn_on_graph.graph.hnsw_graph.select_neighbor import base_algorithm
from model.learn_on_graph.graph.hnsw_graph import graph_element
import copy
import numpy as np
import heapq


class HeuristicAlgorithm(base_algorithm.BaseAlgorithm):
    def __init__(self, config):
        super(HeuristicAlgorithm, self).__init__(config)
        self.extend_candidates = config['extend_candidates']
        self.keep_pruned_connections = config['keep_pruned_connections']
        # self.k_graph

    def select_neighbors(self, graph, base, candidate_elements, insert_point_idx):
        # 返回的是插入后的节点
        # candidate_elements是大根堆, 堆顶存放的是离query最远的点的idx和距离
        # res_insert是小根堆
        res_insert = []
        # working_queue是小根堆
        working_queue = []
        for ele in candidate_elements:
            working_ele = graph_element.GraphElement(ele.idx, -ele.distance)
            heapq.heappush(working_queue, working_ele)
        if self.extend_candidates:
            for ele in candidate_elements:
                for ele_adj_idx in graph[ele.idx - 1]:
                    ele_adj_distance = np.linalg.norm(base[ele_adj_idx - 1] - base[insert_point_idx - 1])
                    ele_adj = graph_element.GraphElement(ele_adj_idx, ele_adj_distance)
                    if ele_adj not in working_queue:
                        heapq.heappush(working_queue, ele_adj)
        # 这个是小根堆
        working_discard_queue = []
        while len(working_queue) > 0 and len(res_insert) < self.k_graph:
            ele = heapq.heappop(working_queue)
            result_cmp_distance = float('inf')
            for iter_res in res_insert:
                # 计算出ele和iter_res的距离
                dis = np.linalg.norm(base[iter_res.idx - 1] - base[ele.idx - 1])
                result_cmp_distance = min(result_cmp_distance, dis)
            if ele.distance < result_cmp_distance:
                heapq.heappush(res_insert, ele)
            else:
                heapq.heappush(working_discard_queue, ele)

        if self.keep_pruned_connections:
            while len(working_discard_queue) > 0 and len(res_insert) < self.k_graph:
                ele = heapq.heappop(working_discard_queue)
                heapq.heappush(res_insert, ele)
        return res_insert
