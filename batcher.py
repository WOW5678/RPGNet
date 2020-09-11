# -*- coding:utf-8 -*-
"""
@Time: 2019/09/25 16:16
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
from random import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

class GenBatcher(object):
    def __init__(self,data,args):
        self.data=data
        self.args=args

        self.ehr_queue=self.fill_example_queue()
        # 分割训练集，验证集和测试集
        self.train_data,self.valid_data=train_test_split(self.ehr_queue,test_size=(1/3.0))
        self.valid_data,self.test_data=train_test_split(self.valid_data,test_size=(1.0/2))

        self.train_batches=self.create_batch(mode='train')
        self.valid_batches=self.create_batch(mode='valid')
        self.test_batches=self.create_batch(mode='test')

    def create_batch(self,mode='train',shuffleis=True):
        all_batch=[]

        if mode=='train':
            num_batches=int(len(self.train_data)/self.args.batch_size)
            if shuffleis:
                shuffle(self.train_data)
        elif mode=='valid':
            num_batches=int(len(self.valid_data)/self.args.batch_size)
        elif mode=='test':
            num_batches=int(len(self.test_data)/self.args.batch_size)

        for i in range(0,num_batches):
            batch=[]
            if mode=='train':
                batch=(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size])
            elif mode=='valid':
                batch=(self.valid_data[i*self.args.batch_size:(i+1)*self.args.batch_size])
            elif mode=='test':
                batch=(self.test_data[i*self.args.batch_size:(i+1)*self.args.batch_size])
            all_batch.append(batch)
        return all_batch


    def get_batches(self,mode='train'):
        if mode=='train':
            self.train_batches = self.create_batch(mode='train')
            #shuffle(self.train_batches)
            return self.train_batches
        elif mode=='valid':
            return self.valid_batches
        elif mode=='test':
            return self.test_batches

    def get_all_data(self,mode='train'):
        if mode=='train':
            return
        elif mode=='valid':
            pass
        elif mode=='test':
            pass



    def fill_example_queue(self):
        new_queue=[]

        for ehr,labels in zip(self.data.patientDescribs,self.data.labels):
            example=Example(ehr,labels,self.args,self.data._word2id)
            new_queue.append(example)
        return new_queue

class Example(object):
    def __init__(self,ehr,labels,args,word2id):

        # 对ehr进行id化
        self.ehr=[word2id.get(item,word2id.get('[UNK]')) for item in ehr.split()]
        self.ehr_len = len(self.ehr)
        #print('labels:',labels)
        self.labels=[args.node2id.get(item) for item in labels.split(';') if len(item.strip())>0]

        self.hier_labels=[args.hier_dicts.get(item) for item in self.labels]
        # if 1033 in self.labels:
            # print('labels_ids:', self.labels)
            # print('hier_labels:',self.hier_labels)
        self.label_len=len(labels)
        self.padding_ehr(args.padded_len_ehr)

        # 将电子病历 padding到固定的维度

    def padding_ehr(self,max_len):
        if self.ehr_len<= max_len:  # 需要padding
            zeroes = list(np.zeros(max_len - self.ehr_len))
            self.ehr = self.ehr + zeroes
        elif self.ehr_len > max_len:
            self.ehr = self.ehr[:max_len]





