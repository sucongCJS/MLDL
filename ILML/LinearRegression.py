# 支持多元线性回归
import numpy as np 
from .metrics import r2_score

class LinearRegression:
    def __init__(self):
        """初始化"""
        self.coef_ = None # 系数 \theta1 ~ \thetan
        self.interception_ = None # 截距 \theta0
        self._theta = None # 私有变量, 为所有\theta 列向量

    def fit_normal(self, X_train, y_train):
        """使用正规方程解训练模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train" # y是列向量, 或者是是一维数组
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 为矩阵的左边加一列1
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train) 

        self.interception_ = self._theta[0] 
        self.coef_ = self._theta[1:] 

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """使用梯度下降法训练线性回归模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"

        # 损失函数的大小
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta))**2) / len(X_b)
            except:
                return float('inf')

        # 损失函数对各个特征值求偏导, 计算梯度
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta)) # 开辟空间
            # res[0] = np.sum(X_b.dot(theta) - y) # 第一个没有带X, 独立求
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:,i]) #?? 列向量点乘列向量, 不, 两个的shape都是(100,), 已经退化为一维数组了, 数组和数组点乘, 得一个整数, 所以np.num()也不需要了, .dot()就是对应乘积求和
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b) # 向量化 转置只是为了和学的线代相符
 
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

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """使用梯度下降法"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):
            t0 = 5
            t1 = 50
            def learning_rate(t):
                return t0 / (t + t1)
            
            theta = initial_theta
            m = len(X_b) # 样本数

            for cur_iter in range(n_iters): # 把所有样本看 n_iters 遍
                # rand_i = np.random.randint(m) # 随机一个样本, 但这样不能保证每个样本都被取到过
                indexes = np.random.permutation(m) # 将X_b的索引打乱
                X_b_new = X_b[indexes] # 用已经打乱的索引花式索引, 打乱元素
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i]) # 这个样本的梯度
                    theta = theta - learning_rate(cur_iter * m + i) * gradient
                # 不能用前后两次的搜索的差距小来作为跳出循环的条件, 因为梯度改变方向是随机的, 不能保证损失函数一直减少
            
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"