from model.learn_on_graph.classifier import classifier, factory
from model.learn_on_graph.classifier.trainnode import nn
from model.learn_on_graph.classifier.datanode import datanode
import numpy as np


class NeuralNetwork(classifier.Classifier):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__(config)
        # self.partition_dir
        self.datanode = factory.create_datanode(config)
        config['train_config']['use_neighbor'] = config['use_neighbor']
        config['train_config']['n_output'] = config['n_cluster']
        self.trainnode = nn.Trainnode(config['train_config'])

    def train(self, base, partition):
        super(NeuralNetwork, self).train(base, partition)
        self.datanode.process_data(base, partition)
        # 准备好数据, 进行训练
        self.trainnode.train(base, self.datanode)

    def eval(self, query, n_bucket_visit):
        super(NeuralNetwork, self).eval(query, n_bucket_visit
                                        )
        # 对每一个query, 得到一个不同桶的概率
        print(query.shape)
        vecs = self.trainnode.eval(query)
        print(vecs)
        # return np.array([[1, 2]])
