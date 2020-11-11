import numpy as np
import torch


def A(a, b):
    print(a, b)


def B(func):
    func(1, 2)


B(A)
