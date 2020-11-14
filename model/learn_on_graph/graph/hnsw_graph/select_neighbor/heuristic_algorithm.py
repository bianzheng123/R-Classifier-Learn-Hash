from model.learn_on_graph.graph.hnsw_graph.select_neighbor import base_algorithm


class HeuristicAlgorithm(base_algorithm.BaseAlgorithm):
    def __init__(self, config):
        super(HeuristicAlgorithm, self).__init__(config)
        # self.k_graph

    def insert(self, graph, n_edges, base, idx_to_insert):
        # 返回的是插入后的节点
        pass
