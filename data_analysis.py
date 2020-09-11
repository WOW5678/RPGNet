# -*- coding:utf-8 -*-
"""
@Time: 2019/11/24 15:35
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import csv
from collections import Counter
import matplotlib.pyplot  as plt


def histogram(data1, n_bins1, data2, n_bins2,cumulative=False, x_label = "", y_label = "", title = ""):
    fontsize=13
    _, (ax1,ax2) = plt.subplots(1,2)
    ax1.hist(data1, bins = n_bins1, cumulative = cumulative, color = '#3da043',edgeColor='k')
    ax2.hist(data2, bins = n_bins2, cumulative = cumulative, color = '#3da043',edgeColor='k')
    ax1.set_ylabel(y_label,fontsize=fontsize)
    ax1.set_xlabel('Number of Labels(50)',fontsize=fontsize)
    ax1.set_title(title,fontsize=fontsize)

    ax2.set_ylabel(y_label,fontsize=fontsize)
    ax2.set_xlabel('Number of Labels(6918)',fontsize=fontsize)
    ax2.set_title(title,fontsize=fontsize)
    plt.tick_params(labelsize=12)

    plt.show()

def label_distri(filename):
    with open(filename) as f:
        reader=csv.reader(f)
        next(reader)
        labels=[]
        data=[row for row in reader]
        for row in data:
            sampleLabel=row[3].strip().split(';')
            labels.extend(list(set(sampleLabel)))
        label_count=Counter(labels).most_common()
        print('label_count:',label_count)

    # 画出标签的分布图
    # 给每个标签id化
    labels_set=list(set(labels))
    # label2id={label:id for id,label in enumerate(labels_set)}
    # labels=[label2id.get(item) for item in labels]
    #labels.sort(reverse = True )
    print('labels:',labels)
    label2id=dict()
    for i,tuple in enumerate(label_count):
        label2id[tuple[0]]=i
    labels = [label2id.get(item) for item in labels]
    return labels


if __name__ == '__main__':
    labels_6918=label_distri('data/filter_top_6918_sample50000.csv')
    labels_50 = label_distri('data/filter_top_50_sample50000.csv')
    histogram(labels_50, 50,labels_6918, 6918, y_label='Number of EHRs')