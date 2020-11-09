import pickle


class BaseModel:
    def __init__(self, config):
        # 保存该模型参数的地址
        self.save_dir = config['project_data_dir']
        # 用于存放单个模型运行结果的文件名
        self.model_save_fname = config['model_save_fname']
        self.has_train = False

    def train(self, base):
        self.has_train = True

    def eval(self, query, n_bucket_visit):
        if not self.has_train:
            print("Warning, the model has not been trained")

    # 保存模型参数
    def save_para(self):
        if not self.has_train:
            print("Warning, the model has not been trained")
        save_dir = '%s/%s' % (self.save_dir, 'train_para.bin')
        with open(save_dir, 'wb') as f:
            pickle.dump(self, f)

    # 读取模型参数
    @staticmethod
    def load_para(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
