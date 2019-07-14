# 均值方差归一化
import numpy as np 

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值方差"""
        assert X.ndim == 2, "the dimension of X must be 2" # ndim表示维度, 得用几个数字确定一个元素的坐标就是几维
        self.mean_ = np.mean(X, axis=0) # 求每一列的均值
        self.scale_ = np.std(X, axis=0) # 求每一列的方差

        return self

    def transform(self, X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim == 2, "the dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, "must fit before transform!" # 必须先执行了fit
        assert X.shape[1] == len(self.mean_), "the feature number of X must be equal to mean_ and scale_"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col] # 把一列的每个行元素归一化
        
        return resX
    def __repr__(self):
        "StandardScaler, mean=%d, std=%d" % self.mean_, self.scale_