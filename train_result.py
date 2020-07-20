# -*- coding:utf-8 -*-
"""
@Time: 2019/11/20 22:06
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib
import  pandas as pd
import csv

print(matplotlib.get_backend() )


def plot_f_fill(epoch_f):
    ax=plt.gca()
    num_epochs=len(epoch_f)
    window = int(num_epochs / 50)
    print('window:', window)
    rolling_mean = pd.Series(epoch_f).rolling(window).mean()
    std = pd.Series(epoch_f).rolling(window).std()
    x = [i for i in range(len(epoch_f))]
    plt.plot(x,rolling_mean)
    ax.fill_between(range(len(epoch_f)), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)
    ax.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1)
    # ax为两条坐标轴的实例
    # 把x轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

def plot_f(epoch_f):
    ax = plt.gca()
    x = [i for i in range(len(epoch_f))]
    plt.plot(x,epoch_f)
    ax.set_title('Performance on valid set')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1)
    # ax为两条坐标轴的实例
    # 把x轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

def plot_loss(losses):
    # 从文件中读出loss值
    ax = plt.gca()
    x=[i for i in range(len(losses))]
    plt.plot(x,losses)
    ax.set_title('Performance on valid set')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss')
    # ax为两条坐标轴的实例
    # 把x轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


if __name__ == '__main__':
    # plot results
    # 画图
    # 把x轴的刻度间隔设置为1，并存在变量里
    x_major_locator = MultipleLocator(1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    y_major_locator = MultipleLocator(0.1)

    with open('data/train_losses.csv','r') as f:
        reader=csv.reader(f)
        loss=[float(row[0]) for row in reader]

    plot_loss(loss)

    with open('data/valid_f.csv','r') as f:
        reader=csv.reader(f)
        valid_f=[float(row[0]) for row in reader]
    plot_f(valid_f)
    plot_f_fill(valid_f)