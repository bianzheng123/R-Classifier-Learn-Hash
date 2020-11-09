import torch.nn as nn
import torch
import numpy as np


class Trainnode:
    def __init__(self, config):
        self.trained = False
        self.candidate_count_l = None
        self.model = NNModel(config)
        self.n_cluster = config['n_output']
        self.n_epochs = config['n_epochs']
        self.acc_threshold = config['acc_threshold']
        lr = config['lr']
        milestones = [10, 17, 24, 31, 38, 45, 50, 55, 60, 70]
        weight_decay = 10 ** (-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.21)

    '''
    得到训练数据
    Input:
    -base 就是数据集
    -dsnode: dataset node, 就是datanode的一个实例
    '''

    def train(self, base, dsnode):
        base = torch.from_numpy(base)
        trainloader = dsnode.trainloader
        valloader = dsnode.valloader
        self.model.train()

        y = trainloader.dataset.tensors[1]
        # 用于计数不同桶中点的数量
        self.candidate_count_l = [(y == i).sum() for i in range(self.n_cluster)]

        X = []
        for epoch in range(self.n_epochs):
            correct = 0
            loss_l = []
            for i, data_blob in enumerate(trainloader):
                ds_idx, partition, neighbor_distribution = data_blob
                if i == 0:
                    X.extend(list(ds_idx))
                if len(ds_idx) == 1:
                    # since can't batchnorm over a batch of size 1
                    continue
                batch_sz = len(ds_idx)

                ds = base[ds_idx]
                predicted = self.model(ds)

                # 计算交叉熵
                pred = torch.log(predicted).unsqueeze(-1)
                loss = -torch.bmm(neighbor_distribution.unsqueeze(1), pred).sum()
                loss_l.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                correct += (predicted.argmax(dim=1) == partition).sum().item()
            cur_recall = correct / len(trainloader.dataset)

            # 从抽出的数据集计算recall
            val_cor = 0
            self.model.eval()
            with torch.no_grad():
                for i, (ds_idx, partition) in enumerate(valloader):
                    cur_data = base[ds_idx]
                    predicted = self.model(cur_data)
                    val_cor += (predicted.argmax(dim=1) == partition).sum().item()
                val_recall = val_cor / len(valloader.dataset)

            print(
                'epoch {} loss: {} train recall: {}    val recall: {} lr: {}'.format(epoch, np.mean(loss_l), cur_recall,
                                                                                     val_recall,
                                                                                     self.optimizer.param_groups[0][
                                                                                         'lr']))
            self.model.train()
            if cur_recall > self.acc_threshold:
                print('Stopping training as acc is now {}'.format(cur_recall))
                break
            self.scheduler.step()
        self.trained = True
        print('correct {} Final recall: {}'.format(correct, cur_recall))

    def eval(self, query):
        query = torch.tensor(query)
        with torch.no_grad():
            return self.model(query)


class NNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['n_input']
        hidden_dim_1 = config['n_hidden'][0]
        hidden_dim_2 = config['n_hidden'][1]
        output_dim = config['n_output']
        dropout_probability = config['p_dropout']
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),

            nn.Linear(hidden_dim_2, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
