from model.learn_on_graph.graph import knn_graph
from model.learn_on_graph.graph.hnsw_graph import hnsw_graph
from model.learn_on_graph.classifier import neural_network


def create_graph(config, save_dir):
    graph_type = config['type']
    config['save_dir'] = save_dir
    if graph_type == 'knn':
        return knn_graph.KNNGraph(config)
    elif graph_type == 'hnsw':
        return hnsw_graph.HNSWGraph(config)
    else:
        raise Exception('遇到无法解析类型的图')


def create_classifier(config, save_dir):
    classifier_type = config['type']
    config['partition_save_dir'] = save_dir
    if classifier_type == 'neural_network':
        return neural_network.NeuralNetwork(config)
    else:
        return Exception('遇到无法解析类型的分类器')
