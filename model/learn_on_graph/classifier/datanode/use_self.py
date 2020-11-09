from model.learn_on_graph.classifier.datanode import datanode
from util import vecs_util
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class UseSelfDatanode(datanode.Datanode):
    def __init__(self, config):
        super(UseSelfDatanode, self).__init__(config)
        # self.batch_size, self.shuffle, self.train_split, self.nn_mult, self.n_cluster
        self.trainloader = None
        self.valloader = None

    '''
    Input:
    -base
    -partition: kahip后的信息
    '''

    def process_data(self, base, partition):
        super(UseSelfDatanode, self).process_data(base, partition)
        cur_split = int(datalen * train_split)
        base_idx = np.arange(base.shape[0])
        trainset = TensorDataset(base_idx[:cur_split], partition[:cur_split])
        # print("trainset", trainset)
        self.trainloader = DataLoader(dataset=trainset, batch_size=self.batch_size,
                                      shuffle=self.shuffle)

        # validation set
        valset = TensorDataset(base_idx[cur_split:], partition[cur_split:])
        # print("valset", valset)
        self.valloader = DataLoader(dataset=valset, batch_size=self.batch_size, shuffle=False)
