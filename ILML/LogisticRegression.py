# 逻辑回归, 处理分类问题
import numpy as np 
from .metrics import accuracy_score

class LogisticRegression:
    def __init__(self):
        """初始化"""
        self.coef_ = None # 系数 \theta1 ~ \thetan
        self.interception_ = None # 截距 \theta0
        self._theta = None # 私有变量, 为所有\theta 列向量

    def _sigmoid(self, t): # 私有函数
        return 1 / (1 + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """使用梯度下降法训练逻辑回归模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"

        # 损失函数的大小
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta)) # 相当于演算中的 p_hat
            try:
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        # 损失函数对各个特征值求偏导, 计算梯度
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta)) # 开辟空间
            # res[0] = np.sum(X_b.dot(theta) - y) # 第一个没有带X, 独立求
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:,i]) #?? 列向量点乘列向量, 不, 两个的shape都是(100,), 已经退化为一维数组了, 数组和数组点乘, 得一个整数, 所以np.num()也不需要了, .dot()就是对应乘积求和
            # return res * 2 / len(X_b)
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b) # 向量化 转置只是为了和学的线代相符
 
        # 最多运行次数n_iters, 精度epsilon
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                    
                cur_iter += 1
            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果概率向量""" # 结果转为概率形式
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta)) # y_predict 转为概率形式

    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int') # 一个0 1 向量

    def score(self, X_test, y_test):
        """确定模型准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)


    def __repr__(self):
        return "LogisticRegression()"