from model.learn_on_graph import factory
from model import base_model
from util import read_data
import torch
import numpy as np
import os


class LearnOnGraph(base_model.BaseModel):

    def __init__(self, config):
        super(LearnOnGraph, self).__init__(config)
        # self.has_train, self.save_dir
        print(config)
        self.graph_path = '%s/graph.graph' % self.save_dir
        self.partition_path = '%s/partition.txt' % self.save_dir
        config['classifier_config']['k'] = config['k']
        config['classifier_config']['n_cluster'] = config['n_cluster']
        self.graph_ins = factory.create_graph(config['graph_config'], self.graph_path)
        self.classifier = factory.create_classifier(config['classifier_config'], self.partition_path)
        self.kahip_config = config['kahip']
        self.kahip_config['kahip_dir'] = config['kahip_dir']

    def train(self, base):
        print('start training {}'.format(self.save_dir.split('/')[-1]))
        super(LearnOnGraph, self).train(base)
        # TODO
        # 搞出一个knn图
        self.graph_ins.create_graph(base)
        self.graph_ins.save_graph()
        # 调用kahip, 保存kahip结果
        self.build_and_save_partition()
        # 读取partition信息
        partition = read_data.read_partition(self.partition_path)
        partition = np.array(partition)
        self.get_labels(partition)
        # 然后训练一个nn
        self.classifier.train(base, partition)
        print('finish training {}'.format(self.save_dir.split('/')[-1]))

    # queryset, 二维numpy数组, n * d, sift中d = 128
    # n_bucker_visit, 需要访问n个桶
    # 支持批量处理
    # 返回的是这个桶内各个点在base上的index
    def eval(self, queryset, n_bucket_visit):
        super(LearnOnGraph, self).eval(queryset, n_bucket_visit)
        predict_label_distribution = self.classifier.eval(queryset, n_bucket_visit)
        _, sort_index = predict_label_distribution.topk(n_bucket_visit, largest=True, sorted=True)
        predict_label_idx = []
        for i in range(len(queryset)):
            tmp_predict_label = set()
            for j in range(n_bucket_visit):
                # print(set(self.label[sort_index[i][j].item()]))
                tmp_predict_label = tmp_predict_label | set(self.label[sort_index[i][j].item()])
            predict_label_idx.append(tmp_predict_label)
        # print(len(predict_label_idx[0]))
        return predict_label_idx
        # return np.array([[1, 2]])

    def build_and_save_partition(self):
        print("perform kahip partitioning")
        os.system("%s/deploy/kaffpa %s --preconfiguration=%s --output_filename=%s --k=%d" % (
            self.kahip_config['kahip_dir'], self.graph_path, self.kahip_config['preconfiguration'], self.partition_path,
            self.n_cluster))

    def __str__(self):
        return 'KMeans n_cluster:%d save_dir: %s' % (self.n_cluster, self.save_dir)
