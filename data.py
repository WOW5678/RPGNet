# -*- coding:utf-8 -*-
"""
@Time: 2019/09/25 15:27
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import csv
from nltk.corpus import stopwords
from string  import punctuation
import os
from  collections import Counter
import pickle

PAD_TOKEN='[PAD]'
UNKNOWN_TOKEN='[UNK]'
ROOT='ROOT'

def processEHR(str):
    #对字符串进行预处理操作
    str=str.lower() #全部转化为小写
    str=' '.join([word for word in str.split() if word not in stopwords.words('english')]) #删除停用词
    str=' '.join([word for word in str.split() if word not in  punctuation]) #删除标点符号
    str=' '.join([word for word in str.split() if word.isalpha()]) #移除数字
    str=' '.join([word for word in str.split() if len(word)>=2])
    return str

class Data(object):
    def __init__(self,datafolder,vocab_size,labels,samples):
        self._word2id={}
        self._id2word={}

        # 读数据
        patientWords = []  # 非结构化数据中所有的单词
        self.patientDescribs=[]
        self.labels=[]

        if os.path.exists(os.path.join(datafolder,'patientDescribes_labels_%d_sample%d.pkl'%(labels,samples))):
        #if os.path.exists(os.path.join(datafolder, 'patientDescribes_labels_50_sampleAll.pkl')):
            #print('datafolder:',datafolder)
            with open(os.path.join(datafolder,'patientDescribes_labels_%d_sample%d.pkl'%(labels,samples)),'rb') as f:
            #with open(os.path.join(datafolder, 'patientDescribes_labels_50_sampleAll.pkl'), 'rb') as f:
                self.patientDescribs, self.labels=pickle.load(f)
            with open(os.path.join(datafolder,'patientWords_%d_sample%d.pkl'%(labels,samples)),'rb') as f:
            #with open(os.path.join(datafolder, 'patientWords_50_sampleAll.pkl'), 'rb') as f:
                patientWords=pickle.load(f)
        else:

            with open(os.path.join(datafolder, 'filter_top_%d_sample%d.csv'%(labels,samples))) as f:
            #with open(os.path.join(datafolder, 'note_labeled.csv')) as f:
                reader = csv.reader(f)
                next(reader)
                data = [row for row in reader]
                for row in data:
                    str = processEHR(row[2])
                    self.patientDescribs.append(str)
                    patientWords.extend(str.split())
                    self.labels.append(row[3].strip())

            # 将处理好的数据保存下来  进行第二次使用
            with open(os.path.join(datafolder,'patientDescribes_labels_%d_sample%d.pkl'%(labels,samples)),'wb') as f:
            #with open(os.path.join(datafolder, 'patientDescribes_labels_50_sampleAll.pkl'), 'wb') as f:
                pickle.dump([self.patientDescribs,self.labels],f)
            with open(os.path.join(datafolder,'patientWords_%d_sample%d.pkl'%(labels,samples)),'wb') as f:
            #with open(os.path.join(datafolder, 'patientWords_50_sampleAll.pkl'), 'wb') as f:
                pickle.dump(patientWords,f)


        # 对patientWords按照频率进行排序，将最频繁的max_size-2个单词加入到vocab中去
        patientWords = Counter(patientWords)
        sortedWords = patientWords.most_common(vocab_size-2)

        self._word2id = {w: i + 2 for i, (w, c) in enumerate(sortedWords)}  # 索引从2开始
        self._word2id[PAD_TOKEN]=0
        self._word2id[UNKNOWN_TOKEN]=1

        self._id2word = {id: w for w, id in self._word2id.items()}

        # icds=list(set(icds))

        # self.icd2id={id:icd+1 for id,icd in enumerate(icds)}
        # self.icd2id[ROOT]=0
        # self.id2icd={id:icd for icd,id in self.icd2id.items()}

    def word2id(self,word):
        if word not in self._word2id:
            return self._word2id[UNKNOWN_TOKEN]
        return self._word2id[word]

    def id2word(self,word_id):
        if word_id not in self._id2word:
            raise ValueError('Id not found in vocab:%d' %word_id)
        return self._id2word[word_id]
    def size(self):
        return len(self._word2id)
    


    

