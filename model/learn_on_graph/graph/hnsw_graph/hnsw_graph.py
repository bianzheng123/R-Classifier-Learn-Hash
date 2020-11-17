from model.learn_on_graph.graph import base_graph
from model.learn_on_graph.graph.hnsw_graph import factory, graph_element
import numpy as np
import heapq
import random


class HNSWGraph(base_graph.BaseGraph):
    def __init__(self, config):
        super(HNSWGraph, self).__init__(config)
        config['select_neighbors_config']['k_graph'] = config['k_graph']
        self.select_neighbor = factory.get_select_neighbor_algorithm(config['select_neighbors_config'])
        self.k_graph = config['k_graph']
        self.efConstruction = config['efConstruction']
        self.graph_info = None
        # self.graph_info = (vertices, edges, result_graph)
        # save_dir

    '''
    输入base文件
    输出点，边的数量以及加权图
    读取bvecs文件, 使用暴力算出每一个点的最邻近距离, 转换成图, 输出为文本
    '''

    def create_graph(self, base):
        vertices = len(base)
        if vertices < self.k_graph + 1:
            print("error, 输入数据量太少, 不能满足边的数量")
            return
        # base_idx打乱后的索引, insert_idx插入时需要按照的顺序, 全部数组都是1-based
        base_idx, insert_order_l = self.shuffle_index(vertices)
        # insert_order_l = np.arange(vertices) + 1
        graph = []
        for i in range(vertices):
            graph.append(set())

        # 根据insert_idx的顺序插入图, 这里随机选取开始walk的点
        for i, insert_point_idx in enumerate(insert_order_l, 0):
            # 初始确定开始walk的索引, 此时表中数据为空
            if i == 0:
                continue
            if i % 500 == 0:
                print("build graph chunk", i)
            # 不是随机选取start_walk_idx会跑的非常慢, siftsmall 不随机选取要跑8分半, 随机选取值只需要两秒
            enter_point_idx = insert_order_l[random.randint(0, i - 1)]
            # enter_point_idx = insert_order_l[0]
            # candidate_elements是大根堆, 堆顶存放的是离query最远的点的idx
            candidate_elements = self.search_layer(insert_point_idx, enter_point_idx, base, graph)
            # print([ele.__str__() for ele in candidate_elements])

            candidate_to_conn = self.select_neighbor.select_neighbors(graph, base, candidate_elements, insert_point_idx)
            # print([ele.__str__() for ele in candidate_to_conn])

            # 得到最近的k_graph个节点, 连接他们
            for ele in candidate_to_conn:
                graph[ele.idx - 1].add(insert_point_idx)
                graph[insert_point_idx - 1].add(ele.idx)

            # print(graph)
            # continue
            for neighbor in candidate_to_conn:
                neighbor_conned = graph[neighbor.idx - 1]
                if len(neighbor_conned) > self.k_graph:
                    # 构建一个neighbor的neighbor的大根堆
                    neighbor_neighbor_max_heap = []
                    for neighbor_neighbor_idx in neighbor_conned:
                        neighbor_neighbor_distance = np.linalg.norm(
                            base[neighbor.idx - 1] - base[neighbor_neighbor_idx - 1])
                        heapq.heappush(neighbor_neighbor_max_heap,
                                       graph_element.GraphElement(neighbor_neighbor_idx, -neighbor_neighbor_distance))
                    # print(neighbor_neighbor_max_heap)
                    neighbor_to_conn = self.select_neighbor.select_neighbors(graph, base, neighbor_neighbor_max_heap,
                                                                             neighbor.idx)
                    # if i == 3:
                    #     print("neighbor_connected", neighbor_conned)
                    #     print(graph, neighbor)
                    #     print("neighbor_to_conn", [ele.__str__() for ele in neighbor_to_conn])
                    # 删除原有的边, 就是删除neighbor_conned的边, 增加neighbor_to_conn的边
                    for idx in neighbor_conned:
                        # print("graph", graph[idx - 1], idx - 1)
                        if neighbor.idx not in graph[idx - 1]:
                            print("双向边不存在")
                        graph[idx - 1].remove(neighbor.idx)
                    graph[neighbor.idx - 1] = set()
                    # 增加新的节点
                    # print(neighbor_to_conn)
                    for ele in neighbor_to_conn:
                        if ele.idx == neighbor.idx:
                            continue
                        graph[ele.idx - 1].add(neighbor.idx)
                        graph[neighbor.idx - 1].add(ele.idx)
            # if i == 3 or i == 2:
            #     print(graph)

        edges = 0
        for ele in graph:
            edges += len(ele)
        if edges % 2 != 0:
            # print(graph)
            raise Exception("边的出现bug")
        edges = edges / 2
        self.graph_info = (vertices, edges, graph)

    def search_layer(self, query_idx, enter_point_idx, base, graph):
        # 存放的是访问过元素的idx
        visited_elements = set()
        visited_elements.add(enter_point_idx)
        distance = np.linalg.norm(base[query_idx - 1] - base[enter_point_idx - 1])
        # candidates_l需要提取最近的元素, 所以是小根堆
        candidates_l = [graph_element.GraphElement(enter_point_idx, distance)]
        # nearest_candidates_l需要提取最远的元素, 所以是大根堆
        nearest_candidates_l = [graph_element.GraphElement(enter_point_idx, -distance)]
        while len(candidates_l) > 0:
            ele_candidates = heapq.heappop(candidates_l)
            ele_nearest_candidate = nearest_candidates_l[0]
            if ele_candidates.distance > -ele_nearest_candidate.distance:
                break
            # 更新candidates_l和nearest_candidates_l
            for neighbor in graph[ele_candidates.idx - 1]:
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    ele_nearest_candidate = nearest_candidates_l[0]
                    neighbor_distance = np.linalg.norm(base[query_idx - 1] - base[neighbor - 1])
                    if neighbor_distance < -ele_nearest_candidate.distance or len(
                            nearest_candidates_l) < self.efConstruction:
                        heapq.heappush(candidates_l, graph_element.GraphElement(neighbor, neighbor_distance))
                        heapq.heappush(nearest_candidates_l, graph_element.GraphElement(neighbor, -neighbor_distance))
                        while len(nearest_candidates_l) > self.efConstruction:
                            heapq.heappop(nearest_candidates_l)

        return nearest_candidates_l
