import numpy as np


class BaseGraph:
    def __init__(self, config):
        self.save_dir = config['save_dir']

    '''
    输入base文件
    输出点，边的数量以及加权图
    '''

    def create_graph(self, base):
        pass

    def save_graph(self):
        pass

    # 返回的是建图的idx, 用于打乱顺序建图
    def shuffle_index(self, base_len):
        base_idx = np.arange(base_len)
        # 用于表示打乱base后原位置插入的先后顺序
        np.random.shuffle(base_idx)
        # insert_idx用于优化遍历查找插入元素的流程
        insert_idx = np.zeros(base_len)
        for i in range(len(insert_idx)):
            insert_idx[base_idx[i]] = i
        return base_idx + 1, insert_idx.astype(np.int) + 1

    # 将图保存为graph.graph
    def save_graph(self):
        vertices, edges, result_graph = self.graph_info
        with open(self.save_dir, 'w') as f:
            f.write("%d %d\n" % (vertices, edges))
            for nearest_index in result_graph:
                row_index = ""
                for item in nearest_index:
                    row_index += str(item) + " "
                # print(row_index)
                f.write(row_index + '\n')
        print("extract vector hahip complete")
