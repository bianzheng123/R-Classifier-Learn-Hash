from model.learn_on_graph.graph import base_graph
from model.learn_on_graph.graph.hnsw_graph import factory
import numpy as np
import heapq
import random


class HNSWGraph(base_graph.BaseGraph):
    def __init__(self, config):
        super(HNSWGraph, self).__init__(config)
        config['select_neighbors_config']['k_graph'] = config['k_graph']
        self.select_neighbor = factory.get_select_neighbor_algorithm(config['select_neighbors_config'])
        self.k_graph = config['k_graph']
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
        graph = []
        for i in range(vertices):
            graph.append(set())

        edges = 0

        # 根据insert_idx的顺序插入图, 这里随机选取开始walk的点
        for i, insert_point_idx in enumerate(insert_order_l, 0):
            # 初始确定开始walk的索引, 此时表中数据为空
            if i == 0:
                continue
            # 不是随机选取start_walk_idx会跑的非常慢, siftsmall 不随机选取要跑8分半, 随机选取值只需要两秒
            start_walk_idx = insert_order_l[random.randint(0, i - 1)]  # 先赋值为任意值
            # print("start_walk_idx", start_walk_idx)
            # start_walk_idx = insert_order_l[0]

            insert_info = (insert_point_idx, start_walk_idx)
            candidate_to_conn = self.select_neighbor.insert(graph, base, insert_info)

            # 得到最近的k_graph个节点, 连接他们
            for tmp_tuple in candidate_to_conn:
                nearest_idx, distance = tmp_tuple
                graph[nearest_idx - 1].add(insert_point_idx)
                graph[insert_point_idx - 1].add(nearest_idx)
                edges += 2
        edges = edges / 2
        # 还需要进行剪枝
        self.graph_info = (vertices, edges, graph)
