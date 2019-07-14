import numpy as np 
from .metrics import r2_score

class SimpleLinearRegression:
    def __init__(self):
        """初始化"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "simple linear regressor can only solve single feature training data"
        assert y_train.ndim == 1, \
            "simple linear regressor can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # self.a_ = sum((x_train-x_mean) * (y_train-y_mean)) / sum((x_train-x_mean)**2)
        # self.b_ = y_mean - self.a_ * x_mean

        # 向量化计算 
        num = (x_train - x_mean).dot(y_train - y_mean) # 分子
        d = (x_train - x_mean).dot(x_train - x_mean) # 分母
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "simple linear regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before transform!"

        return self.a_ * x_predict + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression"


# a用向量化计算, 性能起飞
class SimpleLinearRegression2:
    def __init__(self):
        """初始化"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "simple linear regressor can only solve single feature training data"
        assert y_train.ndim == 1, \
            "simple linear regressor can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = sum((x_train-x_mean) * (y_train-y_mean)) / sum((x_train-x_mean)**2)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "simple linear regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before transform!"

        return self.a_ * x_predict + self.b_
 