class Datanode:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.train_split = config['train_split']
        self.nn_mult = config['nn_mult']
        self.k = config['k']
        self.n_cluster = config['n_cluster']

    def process_data(self, base, partition):
        pass
