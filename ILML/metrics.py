# 度量
import numpy as np 
from math import sqrt

def accuracy_score(y_true, y_predict):
    """计算准确率"""
    assert y_predict.shape[0] == y_true.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)

def mean_sqared_error(y_true, y_predict):
    assert len(y_predict) == len(y_true), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_predict - y_true)**2) / len(y_true)

def root_mean_sqared_error(y_true, y_predict):
    return sqrt(mean_sqared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    assert len(y_predict) == len(y_true), \
        "the size of y_true must be equal to the size of y_predict"

    return  np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    """R Square"""
    return 1 - mean_sqared_error(y_true, y_predict)/np.var(y_true)