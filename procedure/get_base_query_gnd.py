import _init_paths
import numpy as np
import faiss
from util import vecs_io
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


"""
通过相似性搜索得到gnd文件
输入: base数组, query数组, k(多少个临近的结果)
输出: numpy文件
"""


def get_gnd_numpy(base, query, k, save_dir=None):
    base_dim = base.shape[1]
    index = faiss.IndexFlatL2(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    if save_dir is not None:
        np.save(save_dir, gnd_idx)
    return gnd_idx


'''
不再使用, 废弃
'''


# 读取bvecs文件, 使用暴力算出每一个点的最邻近距离, 输出为文本
def get_NN_graph(fname, new_fname, edges_per_vertices):
    # 在数据集中读取base.vecs, 使用暴力算法算出每一个点的距离, 写入文件
    start_time = time()
    vectors = np.load(fname)
    dim = vectors.shape[1]
    vertices = len(vectors)
    if vertices < edges_per_vertices + 1:
        print("error, 输入数据量太少, 不能满足边的数量")
        return

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    distance, index_arr = index.search(vectors, edges_per_vertices + 1)  # 第一个索引一定是自己, 所以+1
    index_arr = index_arr[:, -edges_per_vertices:] + 1  # 要求序号从1开始, 就集体加1
    result = index_arr.copy().tolist()
    print("get the nearest k result")
    if 192427 in result[192427]:
        print("在建立相似性搜索时没有排除self-loop")
    print("建图前")
    print(result[192427])

    for i in range(len(result)):
        for vertices_index in result[i]:
            if (i + 1) not in result[vertices_index - 1]:
                result[vertices_index - 1].append(i + 1)
    print("change to graph complete")

    print("建图后")
    print(result[192427])

    edges = 0
    for index_edge in result:
        edges += len(index_edge)
    edges = edges / 2
    with open(new_fname, 'w') as f:
        f.write("%d %d\n" % (vertices, edges))
        for nearest_index in result:
            row_index = ""
            for item in nearest_index:
                row_index += str(item) + " "
            # print(row_index)
            f.write(row_index + '\n')
    print("extract vector hahip complete")
    end_time = time()
    print("use", end_time - start_time, "s")


'''
创建文件夹, 提取base, query, gnd
'''


def get_base_query_gnd(config):
    os.system("mkdir %s" % (config['dest_dir']))
    print("创建文件夹")

    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['base'])
    base_npy_dir = '%s/%s' % (config['dest_dir'], 'dataset.npy')
    base = vecs2numpy(base_dir, base_npy_dir, config['dataset_type'])
    print("提取base")

    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['query'])
    query_npy_dir = '%s/%s' % (config['dest_dir'], 'queries.npy')
    query = vecs2numpy(query_dir, query_npy_dir, config['dataset_type'])
    print("提取query")

    gnd_npy_dir = '%s/%s' % (config['dest_dir'], 'answers.npy')
    # print(base_npy_dir)
    # print(query_npy_dir)
    # print(gnd_npy_dir)
    gnd = get_gnd_numpy(base, query, config['k_gnd'], gnd_npy_dir)
    print("提取gnd")
    return base, query, gnd


if __name__ == '__main__':
    fname = '/home/bz/learn-to-hash/data/sift/sift_dataset_unnorm.npy'
    new_fname = '/home/bz/learn-to-hash/data/sift/sift_graph_10/test_graph.txt'
    get_NN_graph(fname, new_fname, 10)
    a = '/home/bz/KaHIP/deploy/graphchecker'
    b = '/home/bz/learn-to-hash/data/sift/sift_graph_10/test_graph.txt'
