from model.learn_on_graph.classifier.datanode import datanode
from util import vecs_util
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset


class UseNeighborDatanode(datanode.Datanode):
    def __init__(self, config):
        super(UseNeighborDatanode, self).__init__(config)
        # self.batch_size, self.shuffle, self.train_split, self.nn_mult, self.n_cluster
        self.trainloader = None
        self.valloader = None

    '''
    Input:
    -base
    -partition: kahip后的信息
    '''

    def process_data(self, base, partition):
        super(UseNeighborDatanode, self).process_data(base, partition)
        # 取前nn_mult * k个gnd作为训练集的标签
        ranks = vecs_util.get_gnd_numpy(base, base, self.nn_mult * self.k)
        base_idx = torch.arange(0, base.shape[0])
        partition = torch.LongTensor(partition)
        datalen = len(partition)
        cur_split = int(datalen * self.train_split)
        ranks = torch.from_numpy(ranks)

        partition_exp = partition.unsqueeze(0).expand(datalen, -1)
        # datalen x opt.k (or the number of nearest neighbors to take for computing acc)
        neigh_cls = torch.gather(partition_exp, 1, ranks)
        '''
        neigh_cls是一个二维数组
        存放的是每一个节点中相邻节点的partition信息
        每一行的最后一个再加上自己的partition信息
        5000 * 51
        '''
        neigh_cls = torch.cat((neigh_cls, partition.unsqueeze(-1)), dim=1)
        cls_ones = torch.ones(datalen, neigh_cls.size(-1))
        cls_distr = torch.zeros(datalen, self.n_cluster)
        '''
        每一个节点的相邻节点在每一个partition的分布, 包括自己
        每一行相加为51
        '''
        cls_distr.scatter_add_(1, neigh_cls, cls_ones)
        cls_distr /= neigh_cls.size(-1)

        trainset = TensorDataset(base_idx[:cur_split], partition[:cur_split], cls_distr[:cur_split])
        self.trainloader = DataLoader(dataset=trainset, batch_size=self.batch_size,
                                      shuffle=self.shuffle)

        # validation set
        valset = TensorDataset(base_idx[cur_split:], partition[cur_split:])
        self.valloader = DataLoader(dataset=valset, batch_size=self.batch_size, shuffle=False)
