from util import vecs_util
import json
import numpy as np
import torch
import os


def idx_local2global(local, local2global):
    global_idx = np.zeros(len(local)).astype(np.int)
    for i, idx in enumerate(local, 0):
        global_idx[i] = local2global[idx]
    return global_idx


'''
计算单一query的recall
gnd, experiment都是一个一维数组, 代表baseset中的索引
'''


def count_recall(experiment, gnd, k):
    gnd = gnd[:k]
    matches = [index for index in experiment if index in gnd]
    recall = float(len(matches)) / k
    return recall


def union_models_label(models, queryset, n_bins):
    # label的顺序是partition后的顺序, 存放的值是baseset对应的索引
    label = []
    for i in range(len(queryset)):
        label.append(set())
    for model in models:
        # label, 列表里面包着set()
        # 列表长度queryset的长度, set存放的是压缩后各个点的索引
        pred_index = model.eval(queryset, n_bins)
        for i in range(len(pred_index)):
            label[i] = label[i] | pred_index[i]
    return label


def intersect_models_label(models, queryset, n_bins):
    # label的顺序是partition后的顺序, 存放的值是baseset对应的索引
    label = None
    for i, model in enumerate(models, 0):
        # label, 列表里面包着set()
        # 列表长度queryset的长度, set存放的是压缩后各个点的索引
        pred_index = model.eval(queryset, n_bins)

        if i == 0:
            label = pred_index
        else:
            for j in range(len(pred_index)):
                label[j] = label[j] & pred_index[j]
    return label


def separate_models_label(models, queryset, n_bins):
    # label, 列表里面包着set()
    # 列表长度queryset的长度, set存放的是压缩后各个点的索引
    label = models[0].eval(queryset, n_bins)
    return label


'''
models指的是每一个模型对象
baseset就是数据集
queryset就是测试集
gndset, groundtruth
n_bins, 需要访问的桶数量
k, 就是kNN中k的大小
eval_func, 确定模型之间的关系, 是交集并集还是单个评估
'''


def eval_models(models, dataset, n_bins, k, label):
    baseset, queryset, gndset = dataset
    n_candidates_l = [len(label_i) for label_i in label]
    recall_l = list()
    label_l = []
    for i in range(len(label)):
        tmp_label = list(label[i])
        tmp_label.sort()
        label_l.append(tmp_label)
    del label
    for i in range(len(label_l)):
        # 根据label存放的baseset索引找到切分数据集后的东西
        partition_items = np.array([baseset[candi_idx] for candi_idx in label_l[i]])
        # print(partition_items.shape)

        # 在partition后的空间内暴力搜索
        query = queryset[i]
        query = query[np.newaxis, :]
        result_local = vecs_util.get_gnd_numpy(partition_items, query, k)[0]
        # 将result的idx转换成全局的idx
        result_global_idx = idx_local2global(result_local, label_l[i])
        # 计算recall
        recall = count_recall(result_global_idx, gndset[i], k)
        recall_l.append(recall)

    recall_avg = np.mean(recall_l)
    n_candidates_avg = np.mean(n_candidates_l)

    # compute 95th percentile probe count
    n05 = int(len(recall_l) * 0.05)
    val, idx = torch.topk(torch.FloatTensor(n_candidates_l), k=n05, dim=0, largest=True)
    n_candidates_95 = val[-1].item()
    print('n_bins: {}, recall: {}, n_candidates_avg: {}, n_candidates_95: {}'.format(n_bins, recall_avg,
                                                                                     n_candidates_avg,
                                                                                     n_candidates_95))

    return recall_avg, n_candidates_avg, n_candidates_95


def train(models, base):
    for model in models:
        model.train(base)
        model.save_para()
        print(model)


def save_json(save_dir, result_fname, json_file):
    print('save fname: %s' % result_fname)
    with open('%s/%s' % (save_dir, result_fname), 'w') as f:
        json.dump(json_file, f)


def eval_diff_relationship(models, config, dataset, result_fname, eval_models_func):
    n_bins_l = config['n_bins_l']
    baseset, queryset, gndset = dataset
    write_data_buffer = []
    for n_bins in n_bins_l:
        label = eval_models_func(models, queryset, n_bins)
        recall_avg, n_candidates_avg, n_candidates_95 = eval_models(models, dataset, n_bins, config['k'], label)
        json_res = {
            'n_bins': n_bins,
            'recall': recall_avg,
            'n_candidates_avg': n_candidates_avg,
            'n_candidates_95': n_candidates_95
        }
        write_data_buffer.append(json_res)
        if recall_avg > config['recall_threshold']:
            break
    save_json(config['project_result_dir'], result_fname, write_data_buffer)


def evaluate(models, config, dataset):
    os.system('mkdir %s' % (config['project_result_dir']))
    config_dir = '%s/config' % config['project_result_dir']
    os.system('mkdir %s' % config_dir)
    save_json(config_dir, 'long_term_config.json', config['long_term_config'])
    save_json(config_dir, 'short_term_config.json', config['short_term_config'])
    save_json(config_dir, 'short_term_config_before_run.json', config['short_term_config_before_run'])
    eval_diff_relationship(models, config, dataset, 'union_result.json', union_models_label)
    eval_diff_relationship(models, config, dataset, 'intersect_result.json', intersect_models_label)
    if config['eval_separate'] is True:
        for model in models:
            eval_diff_relationship([model], config, dataset, model.model_save_fname, separate_models_label)
