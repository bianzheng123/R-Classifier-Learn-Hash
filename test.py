import numpy as np
import torch
import heapq
import random
import copy
from graph_element import GraphElement

if __name__ == '__main__':
    a = {1,2,3}
    for ele in a:
        a.remove(ele)

    # ele1 = GraphElement(3, -2.3)
    # ele2 = GraphElement(1, -1.2)
    # ele3 = GraphElement(2, -10.1)
    # a = []
    # heapq.heappush(a, ele1)
    # heapq.heappush(a, ele2)
    # heapq.heappush(a, ele3)
    # b = copy.copy(a)
    # heapq.heappush(b, GraphElement(10, 20.1))
    # for ele in a:
    #     print(ele)
    # print(b)
    # print(a)
    # print(GraphElement(3, 2.3) in a)
