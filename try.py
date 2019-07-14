import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import  TfidfVectorizer


# , 'ya allah meri sister affia ki madad farma'
data_all = np.array(['jo bhi ap se'])
count_vec = TfidfVectorizer(min_df=1, # ?? 如果df(document frequency)低于这个阈值, 忽略掉这个词 [0.0,1.0]
                            analyzer='word',
                            ngram_range=(1, 2),
                            use_idf=1,
                            smooth_idf=1,
                            sublinear_tf=1,
                            stop_words='english'
                            )
count_vec.fit(data_all)
print(count_vec.vocabulary_)
data_all=count_vec.transform(data_all)
print(data_all)