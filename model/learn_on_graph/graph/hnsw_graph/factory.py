from model.learn_on_graph.graph.hnsw_graph.select_neighbor import heuristic_algorithm, simple_algorithm


def get_select_neighbor_algorithm(config):
    algorithm_type = config['type']
    if algorithm_type == 'simple':
        return simple_algorithm.SimpleAlgorithm(config)
    elif algorithm_type == 'heuristic':
        return heuristic_algorithm.HeuristicAlgorithm(config)
    else:
        return Exception('遇到无法解析类型的选择邻居算法')
