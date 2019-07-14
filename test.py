from ILML.kNN import KNNClassifier
from sklearn import datasets
from ILML.model_selection import train_test_split

iris = datasets.load_iris()
X = iris['data'] # 特征矩阵
y = iris['target'] # 结果标签对应的向量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

clf = KNNClassifier(k=3)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))