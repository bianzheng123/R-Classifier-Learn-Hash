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
