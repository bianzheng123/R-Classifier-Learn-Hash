class BaseAlgorithm:
    def __init__(self, config):
        self.k_graph = config['k_graph']

    '''
    改函数修改了graph的内容
    返回的是需要插入的节点, 结构为数组, 第一个是最近点的索引, 第二个是与query的距离
    '''
    def select_neighbors(self, graph, base, idx_to_insert):
        pass
