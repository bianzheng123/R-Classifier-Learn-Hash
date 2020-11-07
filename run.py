import _init_paths
from procedure import get_base_query_gnd, get_init_models, train_eval
import json
import torch
import numpy as np
import os

if __name__ == '__main__':

    with open('config.json', 'r') as f:
        all_config = json.load(f)

    source_data_dir = all_config['source_data_dir']
    dest_dir = all_config['dest_dir']
    dataset_type = all_config['dataset_type']
    classifiers = all_config['classifiers']
    recall_threshold = all_config['recall_threshold']
    n_bins_l = range(all_config['n_bins_l'][0], all_config['n_bins_l'][1], all_config['n_bins_l'][2])
    k = all_config['k']
    source_data_fname = all_config['source_data_fname']

    # 如果没有已经训练完成就删除
    if os.path.isdir(dest_dir):
        command = 'rm -rf %s' % dest_dir
        print(command)
        os.system(command)

    '''
    数据准备
    创建文件夹, 提取base, query, 转换成图, 调用kahip
    '''
    config_base_query_gnd = {
        "dataset_type": dataset_type,
        'source_data_fname': source_data_fname,

        'source_data_dir': source_data_dir,
        'dest_dir': dest_dir,

        'k_gnd': k
    }
    base, query, gnd = get_base_query_gnd.get_base_query_gnd(config_base_query_gnd)

    config_init_models = {
        'dest_dir': dest_dir,
        'classifiers': classifiers
    }
    # 每一个模型建立自己的文件夹并初始化模型对象
    models = get_init_models.classifier_factory(config_init_models, base)

    # 训练完成时, 新建目录并在对应的目录下存放训练参数
    train_eval.train(models, base)

    config_eval_models = {
        'dest_dir': dest_dir,
        # 采样多少个桶
        'n_bins_l': n_bins_l,
        'k': k,
        'recall_threshold': recall_threshold
    }
    # 进行评估时需要在主文件下记录多少个文件夹
    train_eval.evaluate(models, config_eval_models, base, query, gnd)
