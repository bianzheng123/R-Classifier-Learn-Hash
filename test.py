import numpy as np
import torch

class A:
    def __init__(self):
        pass
    def func(self, a):
        print('a')

class B(A):
    def __init__(self):
        pass
    def func1(self):
        self.func('fsd')

tmp = B()
tmp.func1()