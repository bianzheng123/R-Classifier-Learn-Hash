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

    def insert(self, graph, base, insert_info):
        insert_point_idx, start_walk_idx = insert_info
        # 维护一个长度为k_graph的大根堆, 对节点进行bfs
        max_heap = []
        visited_set = set()
        # if i % 100 == 0:
        #     print("iter", i)
        visited_set.add(start_walk_idx)
        # 计算start_walk_idx节点和query之间的距离
        distance = np.linalg.norm(base[start_walk_idx - 1] - base[insert_point_idx - 1])
        max_heap.append((start_walk_idx, -distance))
        # 大根堆的全部邻居的距离都比自身大就行了
        while True:
            tmp_point_idx, tmp_point_distance = max_heap[0]
            tmp_point_distance = -tmp_point_distance
            neighbor_is_larger_self = True
            for neighbor in graph[tmp_point_idx - 1]:
                # 计算距离
                distance = np.linalg.norm(base[tmp_point_idx - 1] - base[insert_point_idx - 1])
                if neighbor in visited_set:
                    continue
                visited_set.add(neighbor)
                if len(max_heap) < self.k_graph:
                    # heap没有满, 就插入, 否则就舍弃
                    max_heap.append((neighbor, -distance))
                    heapq.heapify(max_heap)
                elif len(max_heap) == self.k_graph:
                    if distance < tmp_point_distance:
                        heapq.heapreplace(max_heap, (tmp_point_idx, -tmp_point_distance))
                        tmp_point_idx, tmp_point_distance = max_heap[0]
                        tmp_point_distance = -tmp_point_distance
                        neighbor_is_larger_self = False
            if neighbor_is_larger_self:
                break
        return max_heap
