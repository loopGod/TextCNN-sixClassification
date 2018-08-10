# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

import pandas as pd
import json
import numpy as np


try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data'
vocab_dir = os.path.join(base_dir, 'myvocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    file_sour_Pred = open('data/source/testing.txt','r',encoding='utf-8') #validation.txt原始预测集获取id
    predcsv=pd.DataFrame()
    pred_text=[]
    id_list=[]
    count=0
    for line in file_sour_Pred.readlines():
        count+=1
        if count %1000 ==0:
          print(count)
        dic = json.loads(line)
        con_=dic['内容']
        s1=''.join([i for i in con_ if i not in '\n ▲▼●◆■★▶◀♠   '])
        seg = s1.strip()
        pred_text.append(seg+seg[::-1]+seg[::-3])
        id_=dic['id']
        id_list.append(id_)
    print('id len:',len(id_list))
    print('text len:',len(pred_text))
    
    file_sour_Pred.close()
    predcsv['id']=id_list

    count=0
    pred_type=[]
    for i in pred_text:
        count+=1
        if count %1000 ==0:
          print(count)
        pred_type.append(cnn_model.predict(i))
    predcsv['result']=pred_type
    
    pred_name='my_cnn9788_test.csv'
    predcsv.to_csv('data/result/{kk}'.format(kk=pred_name),index=False,header=False)
    #df=pd.read_csv('data/result/{kk}'.format(kk=pred_name), encoding = 'gb18030')
    #df.to_csv('data/result/{kk}'.format(kk=pred_name),index=False,encoding='utf-8')
    print('Done')
