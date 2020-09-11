# -*- coding:utf-8 -*-
"""
@Time: 2019/09/10 21:04
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import torch
import torch.nn  as nn
from collections import deque
import numpy as np
import random
import torch.autograd as autograd
import torch.nn.functional as F
from gcn import GCNNet
from torch.distributions import Categorical,Bernoulli
import math
from scipy.stats import entropy
from sklearn.metrics import matthews_corrcoef,accuracy_score
import generator

class ActionSelection(nn.Module):
    def __init__(self,args):
        super(ActionSelection, self).__init__()
        self.args=args

        self.layers=nn.Linear(self.args.state_size,len(args.node2id))
        nn.init.xavier_normal_(self.layers.weight)

        self.alpha=0
        self.brotherMatch = nn.Linear(args.node_embedding_size, 1)
        nn.init.xavier_normal_(self.layers.weight)

        self.pathFusion = nn.Linear(args.node_embedding_size * 6, args.node_embedding_size)
        nn.init.xavier_normal_(self.pathFusion.weight)

        self.path_attention = PathAttention(args)
        self.brother_attention = SelfAttention(args)
        self.highwayUnit = HighwayUnit(args)

        self.best_threshold = [0.5]*len(args.node2id)
        self.accuracies = [0] *len(args.node2id) #保留当前最好的acc指标 只有大于当前的指标 才更新阈值


        # 全局调控网络
        self.layers_one=nn.Linear(args.node_embedding_size,1024)
        self.layers_two=nn.Linear(1024,args.node_embedding_size)

        # 全局与局部表示融合网络
        self.global_local=nn.Linear(args.node_embedding_size,len(args.node2id))
        self.alpha=0.2

    def forward(self,x):
        return self.layers(x)

    def act(self,pathEncoder,ehrRrep,pathRep,action_space,hop):
        '''
        :param state:
        :param action_space: 为[64,229]
        :param children_len: [64]
        :param eval_flag:
        :return:
        '''
        if hop==0:
            state=ehrRrep
        else:
            feature = torch.cat((pathRep, ehrRrep), 1)
            # feature = torch.cat((feature, ehrRrep), 1)
            # feature = torch.cat((feature, pathRep), 1)
            feature = torch.cat((feature, ehrRrep * pathRep), 1)
            feature = torch.cat((feature, ehrRrep - pathRep), 1)
            feature = torch.cat((feature, pathRep - ehrRrep), 1)
            feature = torch.cat((feature, ehrRrep + pathRep), 1)
            state = F.tanh(self.pathFusion(feature))  # [64,300]

        children = torch.Tensor([[i] for i in range(len(self.args.node2id))]).long().to(self.args.device)
        children = pathEncoder.embedding(children).squeeze(1)  # [123,300]
        # MP2: parient-to-children
        parient_to_children = torch.zeros((len(state), len(self.args.node2id), self.args.node_embedding_size)).to(
            self.args.device)
        for i in range(len(state)):
            parient_to_children[i] = state[i].unsqueeze(0) * children
        # MP3: brother-to-brother
        brother_to_brother = self.brother_attention(parient_to_children,action_space)  # [64,300]
        highwayRep = self.highwayUnit(ehrRrep, brother_to_brother)
        #highwayRep=ehrRrep+brother_to_brother

        # 得到局部的预测表示
        global_feature = F.relu(self.layers_two(F.relu(self.layers_one(ehrRrep),inplace=True)),inplace=True)  # [64,300]
        if hop==0:
            logists_global = F.sigmoid(self.global_local(global_feature))*self.args.level_0_mask  #
        elif hop==1:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_1_mask  #
        elif hop==2:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_2_mask  #
        elif hop==3:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_3_mask  #
        # 基于highway MP之后的结果进行预测
        logists_local = F.sigmoid(self.layers(highwayRep).squeeze(-1))*action_space
        logists= self.alpha * logists_local+(1-self.alpha) * logists_global

        c = Categorical(logists)
        action=c.sample()
        log_action=-c.log_prob(action)
        return action.detach().cpu().tolist(),log_action



    def eval_act(self, pathEncoder, ehrRrep,pathRep, action_space, tr_y,hop=1,train_flag=0):

        if hop==0:
            state=ehrRrep
        else:
            feature = torch.cat((pathRep, ehrRrep), 1)
            # feature = torch.cat((feature, ehrRrep), 1)
            # feature = torch.cat((feature, pathRep), 1)
            feature = torch.cat((feature, ehrRrep * pathRep), 1)
            feature = torch.cat((feature, ehrRrep - pathRep), 1)
            feature = torch.cat((feature, pathRep - ehrRrep), 1)
            feature = torch.cat((feature, ehrRrep + pathRep), 1)
            state = F.tanh(self.pathFusion(feature))  # [64,300]

       
        children = torch.Tensor([[i] for i in range(len(self.args.node2id))]).long().to(self.args.device)
        children = pathEncoder.embedding(children).squeeze(1)  # [123,300]
        # MP2: parient-to-children
        parient_to_children = torch.zeros((len(state), len(self.args.node2id), self.args.node_embedding_size)).to(
            self.args.device)
        for i in range(len(state)):
            parient_to_children[i] = state[i].unsqueeze(0) * children
        # MP3: brother-to-brother
        brother_to_brother = self.brother_attention(parient_to_children,action_space)  # [64,123,300]
        highwayRep = self.highwayUnit(ehrRrep, brother_to_brother)
        #highwayRep = ehrRrep + brother_to_brother

        # 得到局部的预测表示
        global_feature = F.relu(self.layers_two(F.relu(self.layers_one(ehrRrep),inplace=True)),inplace=True)  # [64,300]
        if hop==0:
            logists_global = F.sigmoid(self.global_local(global_feature))*self.args.level_0_mask  #
        elif hop==1:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_1_mask  #
        elif hop==2:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_2_mask  #
        elif hop==3:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_3_mask  #
        # 基于highway MP之后的结果进行预测
        logists_local = F.sigmoid(self.layers(highwayRep).squeeze(-1)) * action_space
        logists = self.alpha * logists_global + (1 - self.alpha) * logists_local

        logists=logists.detach().cpu().numpy()
        if train_flag and hop==0:
            # 每训练一次 调整一次阈值
            self.set_threshold(logists, tr_y,hop=hop)
        y_pred = np.array([[1 if logists[i, j] >= self.best_threshold[j] else 0 for j in range(logists.shape[1])] for i in range(len(logists))])
        preds = [np.nonzero(row)[0].tolist() for row in y_pred]
        #print('y_preds:',y_pred)
        for i in range(len(preds)):
            if len(preds[i]) == 0:
                preds[i] = [np.argmax(logists[i])]
        return preds

    def pre_act(self,pathEncoder,ehrRrep,pathRep,tr_y,action_space,hop=1):
        '''
        :param state:
        :param action_space: 为[64,229]
        :param children_len: [64]
        :param eval_flag:
        :return:
        '''

        if hop==0:
            state=ehrRrep

        else:
            feature = torch.cat((pathRep, ehrRrep), 1)
            # feature = torch.cat((feature, ehrRrep), 1)
            # feature = torch.cat((feature, pathRep), 1)
            feature = torch.cat((feature, ehrRrep * pathRep), 1)
            feature = torch.cat((feature, ehrRrep - pathRep), 1)
            feature = torch.cat((feature, pathRep - ehrRrep), 1)
            feature = torch.cat((feature, ehrRrep + pathRep), 1)
            state = F.tanh(self.pathFusion(feature))  # [64,300]

        children=torch.Tensor([[i] for i in range(len(self.args.node2id))]).long().to(self.args.device)
        children=pathEncoder.embedding(children).squeeze(1) #[123,300]
        # MP2: parient-to-children
        parient_to_children=torch.zeros((len(state),len(self.args.node2id),self.args.node_embedding_size)).to(self.args.device)
        for i in range(len(state)):
            parient_to_children[i]=state[i].unsqueeze(0)*children
        # MP3: brother-to-brother
        brother_to_brother=self.brother_attention(parient_to_children,action_space) #[64,123,300]
        highwayRep = self.highwayUnit(ehrRrep, brother_to_brother)
        #highwayRep = ehrRrep + brother_to_brother
        # 得到局部的预测表示
        global_feature = F.relu(self.layers_two(F.relu(self.layers_one(ehrRrep),inplace=True)),inplace=True)  # [64,300]
        if hop==0:
            logists_global = F.sigmoid(self.global_local(global_feature))*self.args.level_0_mask  #
        elif hop==1:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_1_mask  #
        elif hop==2:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_2_mask  #
        elif hop==3:
            logists_global = F.sigmoid(self.global_local(global_feature)) * self.args.level_3_mask  #
        # 基于highway MP之后的结果进行预测
        logists_local = F.sigmoid(self.layers(highwayRep).squeeze(-1)) * action_space
        logists = self.alpha * logists_global + (1 - self.alpha) * logists_local

        return logists

    def set_threshold(self,logists,tr_y,hop):
        # 多标签分类要为每个标签设定阈值  以下是阈值的选择方法
        threshold = np.arange(0.1, 0.9, 0.1)
        if hop==0:
            space=self.args.level_0
        elif hop==1:
            space=self.args.level_1
        elif hop==2:
            space=self.args.level_2
        else:
            space=self.args.level_3
        for i in space:
            acc = []
            y_prob = np.array(logists[:, i])
            for t in threshold:
                y_pred = [1 if prob >= t else 0 for prob in y_prob]
                y_true=tr_y[:,i]
                acc.append(accuracy_score(y_true, y_pred))
            acc = np.array(acc)
            # print('acc:', acc)
            index = np.argmax(acc)
            if acc[index]>self.accuracies[i]:
                self.best_threshold[i] = threshold[index]
                self.accuracies[i]=np.max(acc)

class VanillaConv(nn.Module):
    '''


    def __init__(self,args,vocab_size):
        super(VanillaConv, self).__init__()
        self.args=args
        chanel_num = 1
        filter_num = self.args.num_filter_maps
        filter_sizes = [4]

        # self.embedding = nn.Embedding(vocab_size,args.cnn_embedding_size)
        # self.conv1 = nn.ModuleList([nn.Conv2d(chanel_num, filter_num, (filter_size,self.args.cnn_embedding_size)) for filter_size in filter_sizes])
        # self.dropout1 = nn.Dropout(self.args.dropout_rate)
        #############################################################
        modules = []
        for kernel_size in filter_sizes:
            conv_word = nn.Conv1d(args.cnn_embedding_size, filter_num, kernel_size, padding=int(kernel_size / 2))
            nn.init.xavier_uniform_(conv_word.weight)
            modules.append(conv_word)

        self.convs_word = nn.ModuleList(modules=modules)

        self.embed_drop = nn.Dropout(p=0.2)
        # if init_embeds is None:
        self.embed = nn.Embedding(vocab_size + 2, args.cnn_embedding_size, padding_idx=0)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)

        self.word_U = nn.Linear(filter_num, len(args.node2id))
        nn.init.xavier_uniform_(self.word_U.weight)

        self.final = nn.Linear(filter_num, args.state_size)
        nn.init.xavier_uniform_(self.final.weight)

    def forward(self, x):
        # # 将每个电子病历转化成Tensor对象
        # x = self.embedding(x)
        # #print('x:',x.shape)
        # x = x.unsqueeze(1) #[batch_size,1,200,emb_size]
        # #print('x_unsequeeze:',x.shape)
        # #print(F.relu(self.convs[0](x)).shape) # 每个不同的filter卷积之后为：[batch_size,32,198,1],[batch_size,32,197,1],[batch_size,32,196,1]
        # #print('x:',x.shape) #[49, 1, 100]
        # # 多通道的图卷积操作（单层卷积）
        # x=[F.relu(conv(x),inplace=False) for conv in self.conv1]
        #
        # x_ = [item.squeeze(3) for item in x]
        # x_ = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x_]
        # x_ = torch.cat(x_, 1)
        # x_ = self.dropout1(x_)
        # return x_
        docs = self.embed(x)
        docs = self.embed_drop(docs)
        docs = docs.transpose(1, 2)
        multi_word_features = []
        for module in self.convs_word:
            word_features = module(docs)
            word_features = F.tanh(word_features)
            multi_word_features.append(word_features)
        multi_word_features = torch.stack(multi_word_features, dim=0).transpose(0,1)  ## batch_size * kernel_num *  num_filter_maps * words


        ####################
        word_u = self.word_U.weight
        ## max-pooling over kernel
        word_features, _ = torch.max(multi_word_features, dim=1)  ## batch_size * num_filter_maps * words
        alpha = F.softmax(word_u.matmul(word_features), dim=2)

        context = alpha.matmul(word_features.transpose(1, 2))
        #print('context:', context.shape)
        #y = self.final.weight.mul(context).sum(dim=2)
        return context
        '''

    def __init__(self, args, vocab_size):
        super(VanillaConv, self).__init__()
        self.args = args
        chanel_num = 1
        filter_num = self.args.num_filter_maps
        filter_sizes = [3, 4,5]

        self.embedding = nn.Embedding(vocab_size, args.cnn_embedding_size)
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (filter_size, self.args.cnn_embedding_size)) for filter_size in
             filter_sizes])
        self.dropout1 = nn.Dropout(self.args.dropout_rate)

    def forward(self, x):
        # 将每个电子病历转化成Tensor对象
        x = self.embedding(x)
        # print('x:',x.shape)
        x = x.unsqueeze(1)  # [batch_size,1,200,emb_size]
        # print('x_unsequeeze:',x.shape)
        # print(F.relu(self.convs[0](x)).shape) # 每个不同的filter卷积之后为：[batch_size,32,198,1],[batch_size,32,197,1],[batch_size,32,196,1]
        # print('x:',x.shape) #[49, 1, 100]
        # 多通道的图卷积操作（单层卷积）
        x = [F.relu(conv(x), inplace=True) for conv in self.conv1]

        x_ = [item.squeeze(3) for item in x]
        x_ = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x_]
        x_ = torch.cat(x_, 1)
        x_ = self.dropout1(x_)
        return x_

# 对选择出的路径进行编码，返回路径的表示
# 目前的做法是：将当前的节点以及上一步的节点以及中间的边作为当前时刻的路径
class PathEncoder(nn.Module):
    def __init__(self,args,g):
        self.args=args
        super(PathEncoder, self).__init__()

        #self.hidden_size=args.d_hidden_size
        self.embedding=nn.Embedding(len(args.node2id),args.node_embedding_size)
        #self.gru = nn.GRU(self.args.node_embedding_size, self.hidden_size, num_layers=2, bidirectional=True,
        #                  dropout=0.5)
        #self.gru2hidden = nn.Linear(2 * 2 * self.hidden_size, self.hidden_size)
        # self.gcn=GCNNet()
        # self.w_path=nn.Linear(args.node_embedding_size*2,args.node_embedding_size)
        # self.embedding.weight = nn.Parameter(self.gcn(g, self.embedding.weight))
        #self.w_path = nn.Linear(args.node_embedding_size * 2, args.node_embedding_size)

    # def init_hidden(self, batch_size):
    #     h = torch.zeros(2*2*1, batch_size, self.hidden_size).to(self.args.device)
    #     return h

    # 另一种编码路径的方法（et_1,et-et_1）
    def forward(self,actionList):
        #print('actionList:',actionList)

        current_node = [[sample[-1]] for sample in actionList]
        # print('current_node:',current_node)
        current_node = torch.Tensor(current_node).long().to(self.args.device)
        # print('current_node:',current_node)
        current_node_emb = self.embedding(current_node).squeeze(1)  # [64,100]
        # path_emb = torch.cat((current_node_emb,current_node_emb-current_node_emb),dim=1)
        # if len(actionList[0])>=2:
        #     last_node=[[sample[-2]] for sample in actionList]
        #     last_node = torch.Tensor(last_node).long().to(self.args.device)
        #     # print('current_node:',current_node)
        #     last_node_emb = self.embedding(last_node).squeeze(1)  # [64,100]
        #     path_emb = torch.cat((current_node_emb,current_node_emb-last_node_emb),dim=1)
        return current_node_emb

        # paths = torch.Tensor(paths).long().to(self.args.device) #[265,4]
        # hidden = self.init_hidden(paths.size()[0])  # [4,265,500]
        # emb = self.embedding(paths)  # [64,2,300]
        # emb = emb.permute(1, 0, 2)  # [4,265,300]
        # # print('emb.size:',emb.shape)
        # # print('hidden,size:',hidden.shape)
        # _, hidden = self.gru(emb, hidden)  # 4 x batch_size x hidden_dim
        # hidden = hidden.permute(1, 0, 2).contiguous()  # batch_size x 4 x hidden_dim [265,4,500]
        # out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_size))  # batch_size x 4*hidden_dim
        # return out #[265,500]


class PathDecoder(nn.Module):
    def __init__(self,args):
        self.args=args
        super(PathDecoder, self).__init__()
        self.hidden_size=args.path_hidden_size
        self.max_length=args.hops

        self.gru=nn.GRU(self.hidden_size,self.hidden_size)
        self.hidden=self.initHidden(args)

    # 其中x是对路径编码之后的表示 所以不需要再进行embedding
    def forward(self,input): # 其中input为decoder_input,hidden为decoder_hidden
        # print('before-self.hidden:',self.hidden.shape)
        output,self.hidden=self.gru(input,self.hidden)
        return output


    def initHidden(self,args):
        hidden = torch.zeros((1,self.args.k,args.path_hidden_size))
        hidden = torch.Tensor(hidden).to(args.device)
        return hidden


class EncoderRNN(nn.Module):
    def __init__(self, args, num_layers, vocab_size):
        super(EncoderRNN, self).__init__()
        self.args=args
        self.hidden_size = 150
        self.num_layers = num_layers
        self.lstm = nn.LSTM(args.cnn_embedding_size, self.hidden_size , num_layers, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(self.hidden_size * 2, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size
        self.embedding = nn.Embedding(vocab_size, args.cnn_embedding_size)

    def forward(self, x):
        # 初始话LSTM的隐层和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.args.device)  # 同样考虑向前层和向后层
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.args.device)

        # 前向传播 LSTM
        x=self.embedding(x) #[64,200,100]
        out, _ = self.lstm(x, (h0, c0))  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)

        # 解码最后一个时刻的隐状态
        #out = self.fc(out[:, -1, :])
        #out=out[:,-1,:]
        return out

class PathAttention(nn.Module):
    def __init__(self,args):
        super(PathAttention, self).__init__()
        self.args=args

    def forward(self,parient_node_emb,final_node_emb):
        hidden = final_node_emb.squeeze(1) #[1,300]
        #print('parient_node_emb,hidden:',parient_node_emb.shape,hidden.shape)
        attn_weights = torch.bmm(parient_node_emb, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        #print('soft_attn_weights:',soft_attn_weights)
        new_hidden_state = torch.bmm(parient_node_emb.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

class SelfAttention(nn.Module):
    def __init__(self,args):
        super(SelfAttention, self).__init__()
        self.args = args
        self.W_s1=nn.Linear(args.node_embedding_size,500)
        self.W_s2=nn.Linear(500,1)

    def forward(self,brother_embs,action_space):
        # brother_embs:[64,123,300]
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(brother_embs))) #[64,123,1]
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        action_space=action_space.unsqueeze(1)
        attn_weight_matrix=attn_weight_matrix*action_space
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        children=torch.bmm(attn_weight_matrix,brother_embs)
        return children.squeeze(1)

class HighwayUnit(nn.Module):
    def __init__(self,args):
        super(HighwayUnit, self).__init__()
        self.args=args
        self.layer_gate=nn.Linear(args.node_embedding_size,args.node_embedding_size)

    def forward(self,ehrRep,childrenRep):
        gate_layer_result = F.sigmoid(self.layer_gate(ehrRep))
        #print('gate_layer_result:',gate_layer_result)
        multiplyed_gate_and_match = childrenRep * (1-gate_layer_result)  # [5860, 100])
        multiplyed_gate_and_input = ehrRep * gate_layer_result
        # # 两部分相加
        m_t = multiplyed_gate_and_match + multiplyed_gate_and_input
        return m_t