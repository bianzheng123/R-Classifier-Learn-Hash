# import _init_paths
import numpy as np
from model import kmeans
from model.learn_on_graph import learn_on_graph
import os


def build_model(config, method_name):
    if method_name == 'kmeans':
        return kmeans.KMeans(config)
    elif method_name == 'learn-on-graph':
        print(learn_on_graph)
        return learn_on_graph.LearnOnGraph(config)
    else:
        return None


def classifier_factory(config, base):
    models = []
    classifiers = config['classifiers']

    for method in classifiers:
        for i, method_config in enumerate(classifiers[method], 0):
            fname = '%s_%d' % (method, i)
            method_config['project_data_dir'] = '%s/%s' % (config['project_data_dir'], fname)
            method_config['model_save_fname'] = '%s.json' % fname
            method_config['kahip_dir'] = config['kahip_dir']
            method_config['k'] = config['k']
            # print(method_config)
            os.system('mkdir %s' % method_config['project_data_dir'])
            tmp_model = build_model(method_config, method)
            models.append(tmp_model)

    if len(models) == 0:
        raise Exception('模型个数为0')
    return models
