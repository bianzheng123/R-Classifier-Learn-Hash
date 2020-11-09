class Classifier:
    def __init__(self, config):
        self.partition_dir = config['partition_save_dir']

    def train(self, base, partition):
        pass

    def eval(self, query, n_bucket_visit):
        pass
