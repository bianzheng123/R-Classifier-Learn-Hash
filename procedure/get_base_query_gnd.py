# from procedure import _init_paths
import numpy as np
import faiss
from util import vecs_io, vecs_util
from time import time
import os

'''
提取vecs, 输出numpy文件
'''


def vecs2numpy(fname, new_file_name, file_type, file_len=None):
    if file_type == 'bvecs':
        vectors, dim = vecs_io.bvecs_read_mmap(fname)
    elif file_type == 'ivecs':
        vectors, dim = vecs_io.ivecs_read_mmap(fname)
    elif file_type == 'fvecs':
        vectors, dim = vecs_io.fvecs_read_mmap(fname)
    if file_len is not None:
        vectors = vectors[:file_len]
    vectors = vectors.astype(np.float32)
    np.save(new_file_name, vectors)
    return vectors


'''
创建文件夹, 提取base, query, gnd
'''


def get_base_query_gnd(config):
    os.system("mkdir %s" % (config['project_data_dir']))
    print("创建文件夹")

    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['base'])
    base_npy_dir = '%s/%s' % (config['project_data_dir'], 'dataset.npy')
    base = vecs2numpy(base_dir, base_npy_dir, config['dataset_type'])
    print("提取base")

    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['query'])
    query_npy_dir = '%s/%s' % (config['project_data_dir'], 'queries.npy')
    query = vecs2numpy(query_dir, query_npy_dir, config['dataset_type'])
    print("提取query")

    gnd_npy_dir = '%s/%s' % (config['project_data_dir'], 'answers.npy')
    # print(base_npy_dir)
    # print(query_npy_dir)
    # print(gnd_npy_dir)
    gnd = vecs_util.get_gnd_numpy(base, query, config['k_gnd'], gnd_npy_dir)
    print("提取gnd")
    return base, query, gnd


if __name__ == '__main__':
    fname = '/home/bz/learn-to-hash/data/sift/sift_dataset_unnorm.npy'
    new_fname = '/home/bz/learn-to-hash/data/sift/sift_graph_10/test_graph.txt'
    get_NN_graph(fname, new_fname, 10)
    a = '/home/bz/KaHIP/deploy/graphchecker'
    b = '/home/bz/learn-to-hash/data/sift/sift_graph_10/test_graph.txt'
