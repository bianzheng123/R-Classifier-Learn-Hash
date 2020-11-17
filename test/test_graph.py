import _init_paths
from model.learn_on_graph.graph import knn_graph
from model.learn_on_graph.graph.hnsw_graph import hnsw_graph
from util import vecs_io
import numpy as np
from time import time
import os


def preprocess():
    start_time = time()
    base_dir = '/home/bz/SIFT/siftsmall/siftsmall_base.fvecs'
    base = vecs_io.fvecs_read_mmap(base_dir)[0]
    # base = np.array([[1, 2], [1, 5], [2, 2], [3, 5], [4, 5], [5, 10], [10, 15], [2, 2]])
    # base = np.array([[1, 2], [1, 2], [2, 2], [2,5]])
    base = base.astype(np.float32)
    config = {
        # "select_neighbors_config": {
        #     "type": "simple"
        # },
        "select_neighbors_config": {
            "type": "heuristic",
            "extend_candidates": True,
            "keep_pruned_connections": True
        },
        'save_dir': '/home/bz/R-Classifier-Learn-Hash/test/graph.graph',
        "efConstruction": 100,
        "k_graph": 2
    }
    # graph_ins = knn_graph.KNNGraph(config)
    graph_ins = hnsw_graph.HNSWGraph(config)
    graph_ins.create_graph(base)
    graph_ins.save_graph()

    end_time = time()
    print("time_to_process", (end_time - start_time))
    return graph_ins.graph_info


graph_ins = preprocess()
vertices, edges, graph = graph_ins


class TestGraph:

    def test_vertices(self):
        assert vertices == len(graph)

    def test_edges(self):
        test_edge = 0
        for each_edge in graph:
            test_edge += len(each_edge)
        assert test_edge / 2 == edges

    def test_self_loop(self):
        for i, row in enumerate(graph, 1):
            assert i not in row

# /home/bz/KaHIP/deploy/graphchecker /home/bz/R-Classifier-Learn-Hash/test/graph.graph
# 检测是否符合规范
