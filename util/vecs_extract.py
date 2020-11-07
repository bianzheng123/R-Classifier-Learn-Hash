import _init_paths
import numpy as np
import faiss
import vecs_io
from time import time

'''
提取文本, 输出为二进制文件
'''


def extract_fvecs(fname, new_file_name, file_len):
    vectors, dim = vecs_io.fvecs_read_mmap(fname)
    if len(vectors) < file_len:
        print("larger than file length, program would break")
        return
    new_vectors = vectors[:file_len]
    vecs_io.fvecs_write(new_file_name, new_vectors)


def extract_bvecs(fname, new_file_name, file_len):
    vectors, dim = vecs_io.bvecs_read_mmap(fname)
    if len(vectors) < file_len:
        print("larger than file length, program would break")
        return
    new_vectors = vectors[:file_len]
    vecs_io.bvecs_write(new_file_name, new_vectors)

# extract_bvecs('/home/bz/SIFT/bigann/bigann_base_part/bigann_base_1M.bvecs', '/home/bz/self_code/learn-to-hash/data/bigann_base_10K.bvecs',
#               10000)
# print("get the 10K data")

# extract_bvecs('/home/bz/SIFT/bigann/bigann_base_part/bigann_base_1M.bvecs',
#               '/home/bz/SIFT/bigann/bigann_base_part/bigann_base_10K.bvecs', 10000)
# save_gnd_bvecs('/home/bz/SIFT/bigann/bigann_base_part/bigann_base_10K.bvecs', '/home/bz/SIFT/bigann/bigann_query.bvecs', '/home/bz/SIFT/bigann/gnd/idx_10K.ivecs', 1000)
