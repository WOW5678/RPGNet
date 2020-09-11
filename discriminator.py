# -*- coding:utf-8 -*-
"""
@Time: 2019/10/09 8:55
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from modelUtils import VanillaConv

class Discriminator(nn.Module):
    def __init__(self,args,data):
        super(Discriminator, self).__init__()

        self.args = args
        self.hidden_size = args.d_hidden_size
        self.embedding = nn.Embedding(len(args.node2id), args.node_embedding_size)
        self.gru = nn.GRU(self.args.node_embedding_size, self.hidden_size,num_layers=2, bidirectional=False, dropout=0.5)
        #self.transform1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru2hidden = nn.Linear(2 * 2 * self.hidden_size, self.hidden_size)
        self.dropout_linear = nn.Dropout(p=0.5)
        self.hidden2out = nn.Linear(self.hidden_size+300, 1)

        self.ehrEncoder=VanillaConv(args,vocab_size=data.size())
        self.optimizer=optim.Adam(self.parameters(),0.001)

    # def init_hidden(self, batch_size):
    #     h = torch.zeros(2*2*1, batch_size, self.hidden_size).to(self.args.device)
    #     return h

    def forward(self,g_model,paths,hidden,ehrReps):
        # input dim                                                # batch_size x seq_len
        # 使用GRU对路径进行编码
        #current_node= torch.Tensor(paths).to(self.args.device)  # []
        # print('current_node:',current_node)
        emb = self.embedding(paths)  # [64,2,100]
        #emb = emb.permute(1, 0, 2) #[2,64,100]
        # print('emb.size:',emb.shape)
        # print('hidden,size:',hidden.shape)
        out, hidden = self.gru(emb, hidden)  # 4 x batch_size x hidden_di
        out=out[:,-1,:]
        #hidden = hidden.permute(1, 0, 2).contiguous()  # batch_size x 4 x hidden_dim
        #out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_size))  # batch_size x 4*hidden_dim
        out = torch.cat((ehrReps, out), 1)  #
        out = torch.tanh(out)
        #out = self.dropout_linear(out)
        out = self.hidden2out(out)  # batch_size x 1
        out = F.sigmoid(out)
        return out

    def batchClassify(self,g_model,ehrs,inp): # 不能光判断路径是否真实  还应该与样本对应起来
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        # 对电子病历数据的处理
        ehrRrep = self.ehrEncoder(ehrs)  # [64,300]
        inp= torch.Tensor(inp).long().to(self.args.device)  # [64,2]
        #h = self.init_hidden(inp.size()[0]) #[4,64,500]
        h=None
        out = self.forward(g_model,inp, h,ehrRrep)

        #将out 进行规整成-1到1之间
        #out=2*out-1
        return out

    def train_d_model(self,g_model,ehrs_hops,paths_n,ground_truths_hops,paths_p,mode='BCE'):
        import random

        #sampleCount=0
        for hop in  range(len(paths_n)):
            loss_fn = nn.BCELoss()
            # loss = torch.tensor(0).float().to(self.args.device)
            # 先从paths_n中找出那些生成的错误的样本(确保标签的准确性)
            filter_ehrs,filter_paths=filter_negative(ehrs_hops[hop],paths_n[hop],ground_truths_hops[hop])

            paths= filter_paths+paths_p[hop]

            #取batch size个样本进行训练（因为数据规模太大 无法都）
            paths = torch.Tensor(paths).long().to(self.args.device)
            target=torch.Tensor([0]*len(filter_paths)+[1]*len(paths_p[hop])).float().unsqueeze(dim=-1).to(self.args.device)

            ehrs=filter_ehrs+ehrs_hops[hop].tolist()
            ehrs=torch.Tensor(ehrs).long().to(self.args.device)


            ehrReps=self.ehrEncoder(ehrs)
            #h = self.init_hidden(paths.size()[0])
            h=None
            out = self.forward(g_model,paths, h,ehrReps)
            loss =loss_fn(out, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

def filter_negative(ehrs,paths,ground_truths):
    filter_ehrs=[]
    filter_paths=[]
    for i in range(len(paths)):
        if ground_truths[i][0]<=0: # 说明生成的确实是负样本
            filter_paths.append(paths[i])
            filter_ehrs.append(ehrs[i])
    return filter_ehrs,filter_paths

