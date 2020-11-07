import _init_paths
import numpy as np
from model import kmeans
import os


def classifier_factory(config, base):
    models = []
    classifiers = config['classifiers']
    for method in classifiers:
        tmp_model = None
        if method == 'kmeans':
            for i, method_config in enumerate(classifiers[method], 0):
                fname = 'kmeans_%d' % i
                method_config['dest_dir'] = '%s/%s' % (config['dest_dir'], fname)
                # print(method_config)
                os.system('mkdir %s/%s' % (config['dest_dir'], fname))
                tmp_model = kmeans.KMeans(method_config)
                models.append(tmp_model)
        elif method == 'hnsw':
            pass
        elif method == 'learn-to-hash':
            pass
    return models
