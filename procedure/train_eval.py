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


def eval_models(models, baseset, queryset, gndset, n_bins, k):
    # label的顺序是partition后的顺序, 存放的值是baseset对应的索引
    label = []
    for i in range(len(queryset)):
        label.append(set())
    for model in models:
        # 这里取交集还是并集不太清楚
        # tmp_index, 列表里面包着set()
        pred_index = model.eval(queryset, n_bins)
        for i in range(len(pred_index)):
            label[i] = label[i] | pred_index[i]
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


def evaluate_separate(models, config, baseset, queryset, gnd, result_fname):
    n_bins_l = config['n_bins_l']
    write_data_buffer = []
    for n_bins in n_bins_l:
        recall_avg, n_candidates_avg, n_candidates_95 = eval_models(models, baseset, queryset, gnd, n_bins, config['k'])
        json_res = {
            'n_bins': n_bins,
            'recall': recall_avg,
            'n_candidates_avg': n_candidates_avg,
            'n_candidates_95': n_candidates_95
        }
        write_data_buffer.append(json_res)
        if recall_avg > config['recall_threshold']:
            break
    print('save result fname: %s' % result_fname)
    with open('%s/%s' % (config['project_result_dir'], result_fname), 'w') as f:
        json.dump(write_data_buffer, f)

def evaluate(models, config, baseset, queryset, gnd):
    os.system('mkdir %s' % (config['project_result_dir']))
    evaluate_separate(models, config, baseset, queryset, gnd, 'result.json')
    if config['eval_separate'] is True:
        for model in models:
            evaluate_separate([model], config, baseset, queryset, gnd, model.model_save_fname)
