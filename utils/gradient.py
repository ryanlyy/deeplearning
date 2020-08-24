# coding: utf-8
import numpy as np

def numercial_diff(f, x, idx):
    h = 1e-4 #0.0001
    
    tmp_val = x[idx]

    x[idx] = float(tmp_val) + h
    fxh1 = f(x)
    x[idx] = tmp_val -h
    fxh2 = f(x)
    nd = (fxh1 - fxh2) / (2 * h)

    x[idx] = tmp_val

    return nd

def _numerical_gradient_1d(f, x):
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        grad[idx] = numercial_diff(f, x, idx)
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        grad[idx] = numercial_diff(f, x, idx)
        it.iternext()   
        
    return grad