
'''
Adds relevant directories to path
'''
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

cur_dir = osp.dirname(__file__)

add_path(osp.join(cur_dir, '../_kahip'))
add_path(osp.join(cur_dir, '../procedure'))
add_path(osp.join(cur_dir, '../model'))
add_path(osp.join(cur_dir, '..'))
