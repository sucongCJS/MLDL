# PTB数据集
import os
import datetime
import sys
import argparse
import collections

import numpy as np
import tensorflow as tf

# 数据集目录
data_path = "../../resource/data/simple-examples/data"

# 保存训练所得的模型参数文件的目录
save_path = './save'

parser = argparse.ArgumentParser() # 参数解析器
parser.add_argument('--data_path', type=str, default=data_path, help='The path of the data for training and testing') # 用来指定程序需要接受的命令参数
args = parser.parse_args()

# 词向量编码, 按照词频给编码

# 分割
def read_words(filename):
    with tf.io.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split() # 分成一个个单词, 从空格断, 从句末断

# 构造从单词到唯一整数值的映射
# 后面的其他数的整数值按照它们在数据集里出现的次数多少来排序，出现较多的排前面
# 单词 the 出现频次最多，对应整数值是 0
# <unk> 表示 unknown（未知），第二多，整数值为 1  
def build_vocab(filename):
    data = read_words(filename)
    
    counter = collections.Counter(data)
    counter_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) # 先按数值从大到小排, 再按单词首字母排

    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存. 可以使用list()转换成列表. 利用 * 号操作符，可以将元组解压为列表
    words, _ = list(zip(*counter_pairs)) # 只要排序后的单词

    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

# 将文件里的单词都替换成独一的整数
def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# 加载所有数据，读取所有单词，把其转成唯一对应的整数值
def load_data():
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = build_vocab(train_path)

    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id) #?? 没有对应数值的单词怎么办
    test_data = file_to_word_ids(test_path, word_to_id)

    # 所有不重复单词的个数
    vocab_size = len(word_to_id)

    # 反转一个词汇表：为了之后从 整数 转为 单词
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(word_to_id)
    print("===================")
    print(vocab_size)
    print("===================")
    print(train_data[:10])
    print("===================")
    print(" ".join([id_to_word[x] for x in train_data[:10]]))
    print("===================")
    return train_data, valid_data, test_data, vocab_size, id_to_word

def generate_batches(raw_data, batch_size, num_steps):
    # 将数据转为 Tensor 类型
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size

    # 将数据形状转为 [batch_size, batch_len], 一列为一个批次的
    data = tf.reshape(raw_data[0:batch_size*batch_len], [batch_size, batch_len]) #?? [:] ??行列为何如此
    
    epoch_size = (batch_len - 1) // num_steps # ??

    # range_input_producer 可以用多线程异步的方式从数据集里提取数据
    # 用多线程可以加快训练，因为 feed_dict 的赋值方式效率不高
    # shuffle 为 False 表示不打乱数据而按照队列先进先出的方式提取数据
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue() # ??

    x = data[:, i*num_steps:(i+1)*num_steps]
    x.set_shape

if __name__ == "__main__":
    load_data()