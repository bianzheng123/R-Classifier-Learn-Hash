from base_model import Base_Model
import numpy as np
import sklearn.cluster as cls


class KMeans(Base_Model):

    def __init__(self, config):
        super(KMeans, self).__init__(config)
        # key是每一个类的编号, value是属于该类的点在base对应的索引
        self.label = {}
        # 该模型需要聚类的数量
        self.n_cluster = config['n_cluster']
        self.model = cls.KMeans(n_clusters=self.n_cluster, init='k-means++', max_iter=30)
        # self.base, self.save_dir

    def train(self, base):
        print('start training {}'.format(self.save_dir.split('/')[-1]))
        super(KMeans, self).train(base)
        self.model.fit(base)
        self.get_labels()
        print('finish training {}'.format(self.save_dir.split('/')[-1]))
        # print(self.base)

    # query, 二维numpy数组, n * d, sift中d = 128
    # n_bucker_visit, 需要访问n个桶
    # 支持批量处理
    # 返回的是这个桶内各个点在base上的index
    def eval(self, query, n_bucket_visit):
        super(KMeans, self).eval(query, n_bucket_visit)
        sort_idx = self.model.transform(query)
        # 得到了每一个query在不同类的索引值, 二维数组
        sort_idx = np.argsort(sort_idx)[:, :n_bucket_visit]

        res = []
        for single_query_cluster_idx in sort_idx:
            base_idx = np.array([]).astype(np.int)
            for cluster_i in single_query_cluster_idx:
                base_idx = np.append(base_idx, self.label[cluster_i])
            res.append(base_idx)
        res = np.array(res)
        # print(res.shape) # 对于单个query, shape为1 * candidates数量
        return res

    # 填充self.label
    def get_labels(self):
        labels = self.model.labels_
        for cluster_i in range(self.n_cluster):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label[cluster_i] = base_idx_i

    def __str__(self):
        return 'KMeans n_cluster:%d save_dir: %s' % (self.n_cluster, self.save_dir)
