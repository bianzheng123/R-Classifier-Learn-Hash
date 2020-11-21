from procedure import get_base_query_gnd, get_init_models, train_eval
import json
import torch
import numpy as np
import os
import copy


def delete_dir_if_exist(dir):
    if os.path.isdir(dir):
        command = 'rm -rf %s' % dir
        print(command)
        os.system(command)


if __name__ == '__main__':
    long_config_dir = 'config/long_term_config.json'
    short_config_dir = 'config/kmeans_and_graph/knn.json'

    # 设置两个配置文件, 方便批量执行
    with open(long_config_dir, 'r') as f:
        long_term_config = json.load(f)

    with open(short_config_dir, 'r') as f:
        short_term_config = json.load(f)

    with open(short_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    this_program_dir = short_term_config['this_program_dir']

    classifiers = short_term_config['classifiers']
    n_bins_l = range(short_term_config['n_bins_l'][0], short_term_config['n_bins_l'][1],
                     short_term_config['n_bins_l'][2])

    project_data_dir = '%s/data/%s' % (long_term_config['project_dir'], this_program_dir)
    # 如果没有已经训练完成就删除
    delete_dir_if_exist(project_data_dir)

    project_result_dir = '%s/result/%s' % (long_term_config['project_dir'], this_program_dir)
    # 如果之前已经出现结果就删除
    delete_dir_if_exist(project_result_dir)

    '''
    数据准备
    创建文件夹, 提取base, query, 转换成图, 调用kahip
    '''
    config_base_query_gnd = {
        "dataset_type": long_term_config['dataset_type'],
        'source_data_fname': long_term_config['source_data_fname'],

        'source_data_dir': long_term_config['source_data_dir'],
        'project_data_dir': project_data_dir,

        'k_gnd': long_term_config['k']
    }
    base, query, gnd = get_base_query_gnd.get_base_query_gnd(config_base_query_gnd)

    '''
    每一个模型建立自己的文件夹并初始化模型对象
    '''
    config_init_models = {
        'project_data_dir': project_data_dir,
        'kahip_dir': long_term_config['kahip_dir'],
        'classifiers': classifiers,
        # 用于分类器的k表示
        'k': long_term_config['k']
    }
    models = get_init_models.classifier_factory(config_init_models, base)

    '''
    训练
    完成时新建目录并在对应的目录下存放训练参数
    '''
    train_eval.train(models, base)

    '''
    评估
    在result/创建文件夹,放置结果
    '''
    config_eval_models = {
        'project_result_dir': project_result_dir,
        # 采样多少个桶
        'n_bins_l': n_bins_l,
        'k': long_term_config['k'],
        'recall_threshold': long_term_config['recall_threshold'],
        'eval_separate': short_term_config['eval_separate'],
        'long_term_config': long_term_config,
        'short_term_config': short_term_config,
        'short_term_config_before_run': short_term_config_before_run
    }
    train_eval.evaluate(models, config_eval_models, (base, query, gnd))
