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


def eval_single_query(models, baseset, query, gnd, n_bins, k):
    # label的顺序是partition后的顺序, 存放的值是baseset对应的索引
    label = np.array([]).astype(np.int)
    query = query[np.newaxis, :]
    for model in models:
        # 这里取交集还是并集不太清楚
        # np的一维数组, 返回的是桶中每一个点的索引(相对于base)
        tmp_index = model.eval(query, n_bins)
        # 将二维数组中外层维度为1的数组去掉
        tmp_index = tmp_index.squeeze()
        # print(tmp_index.shape)
        label = np.union1d(label, tmp_index)

    # print(label)
    n_candidates = label.shape[0]
    # print("n_candidates", n_candidates)
    # 根据label存放的baseset索引找到切分数据集后的东西
    partition_items = np.array([baseset[candi_idx] for candi_idx in label])
    # print(partition_items)
    # print("partition_items shape", partition_items.shape)  # x * 128
    # 在partition后的空间内暴力搜索
    result_local = vecs_util.get_gnd_numpy(partition_items, query, k)[0]
    # 将result的idx转换成全局的idx
    result_global_idx = idx_local2global(result_local, label)

    # 计算recall
    recall = count_recall(result_global_idx, gnd, k)
    return n_candidates, recall


def train(models, base):
    for model in models:
        model.train(base)
        model.save_para()
        print(model)


def evaluate(models, config, baseset, queryset, gnd):
    n_bins_l = config['n_bins_l']
    write_data_buffer = []
    for n_bins in n_bins_l:
        n_candidates_l = []
        recall_l = []
        for i in range(len(queryset)):
            n_candidates, recall = eval_single_query(models, baseset, queryset[i], gnd[i], n_bins, config['k'])
            n_candidates_l.append(n_candidates)
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
        json_res = {
            'n_bins': n_bins,
            'recall': recall_avg,
            'n_candidates_avg': n_candidates_avg,
            'n_candidates_95': n_candidates_95
        }
        write_data_buffer.append(json_res)
        if recall_avg > config['recall_threshold']:
            break

    os.system('mkdir %s' % (config['project_result_dir']))
    with open('%s/%s' % (config['project_result_dir'], 'result.json'), 'w') as f:
        json.dump(write_data_buffer, f)
