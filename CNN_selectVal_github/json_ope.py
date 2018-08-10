import jieba
import pandas as pd
import glob
import json
import jieba.posseg
from numba import jit
import numpy as np

'''
import tensorflow as tf
#电脑GPU测试-运行下段程序如果有GPU输出则可以CNN GPU运行
with tf.device('/cpu:0'):
   a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
   b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:0'):
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print (sess.run(c))
error(10) #错误停止
'''
#3-自动摘要  2-机器翻译 0-人类作者  1-机器作者

count=0
file = open('data/source/training.txt','r')
write_file = open('data/mytraining.txt','w',encoding='utf-8')#训练集
test_file = open('data/mytesting.txt','w',encoding='utf-8')  #测试集
val_file = open('data/myval.txt','w',encoding='utf-8')       #验证集=测试集
datacsv=pd.DataFrame()
for line in file.readlines():
    count+=1
    if count %1000 ==0:
      print(count)
    dic = json.loads(line)
    con_=dic['内容']
    lab_=dic['标签']
    s1=''.join([i for i in con_ if i not in '\n ▲▼●◆■★▶◀♠   '])
    seg = s1.strip()
    seg_addlabel=lab_+'\t'+seg
    if count>136000:
        test_file.write(seg_addlabel+'\n')
        val_file.write(seg_addlabel+'\n')
    else:
        write_file.write(seg_addlabel+'\n')
file.close()
write_file.close()
test_file.close()
val_file.close()

print('traing testing TXT Finished...')

count=0
file = open('data/source/validation.txt','r')
write_file = open('data/mypred.txt','w',encoding='utf-8') #预测集
datacsv=pd.DataFrame()
for line in file.readlines():
    count+=1
    if count %1000 ==0:
      print(count)
    dic = json.loads(line)
    con_=dic['内容']
    s1=''.join([i for i in con_ if i not in '\n ▲▼●◆■★▶◀♠   '])
    seg = s1.strip()
    seg_addlabel=seg
    write_file.write(seg_addlabel+'\n')
file.close()
write_file.close()

print('predict TXT Finished...')


'''
count=0
file = open('data/source/testing.txt','r')
write_file = open('data/mytest.txt','w',encoding='utf-8') #预测集
datacsv=pd.DataFrame()
for line in file.readlines():
    count+=1
    if count %1000 ==0:
      print(count)
    dic = json.loads(line)
    con_=dic['内容']
    s1=''.join([i for i in con_ if i not in '\n ▲▼●◆■★▶◀♠   '])
    seg = s1.strip()
    seg_addlabel=seg
    write_file.write(seg_addlabel+'\n')
file.close()
write_file.close()

print('predict TXT Finished...')
'''