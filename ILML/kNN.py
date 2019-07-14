import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k):
        """初始化kNN分类器"""
        assert k>=1, \
            "k must be valid"

        self.k = k
        self._X_train = None
        self._y_train = None
    
    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练分类器"""
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k"
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"

        self._X_train = X_train
        self._y_train = y_train
        return self # 返回自身
        
    def predict(self, X_predict): # X_predict为二维数组
        """给定待预测数据集X_predict, 返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict"
        assert self._X_train.shape[1] == X_predict.shape[1], \
            "the feature number X_predict must be equal to X_train"
        
        y_predict = [self._predict(x) for x in X_predict] # x为一维数组
        return np.array(y_predict)
   
    def _predict(self, x):
        """给定单个待预测数据x, 返回x的预测结果值"""
        assert self._X_train.shape[1] == x.shape[0], \
            "the feature number x must be equal to X_train"
            
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]
        
    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    
    def __repr__(self):
        # str()与repr()函数的区别: str() 的输出追求可读性，输出格式要便于理解，适合用于输出内容到用户终端, repr() 的输出追求明确性，除了对象内容，还需要展示出对象的数据类型信息，适合开发和调试阶段使用。
        return "KNN(k=%d)" % self.k
