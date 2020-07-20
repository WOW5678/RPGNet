# -*- coding:utf-8 -*-
"""
@Time: 2019/09/25 17:01
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from modelUtils import ActionSelection,VanillaConv,PathEncoder
import torch.optim as optim
import numpy as np
import math
from discriminator import *
from torch.distributions import Bernoulli
import discriminator
from scipy.stats import entropy
from torch.optim import lr_scheduler

class Generator(nn.Module):
    def __init__(self,args,data,g):
        super(Generator, self).__init__()
        self.args=args
        self.data=data
        self.g=g
        self.cnn =VanillaConv(args,vocab_size=data.size())
        #self.cnn=EncoderRNN(args, num_layers=1, vocab_size=data.size())
        self.pathEncoder = PathEncoder(args,self.g)
        self.pathHist = []  # 保存着已经选择的路径（只保留最后的一个ICD）


        # 与强化学习有关的模块
        #self.buffer =
        self.ActionSelection = ActionSelection(args)
        self.gamma = 0.99  # 计算未来奖励的折扣率
        self.lamda= 1.0

        self.optimizer=optim.Adam(self.parameters(),lr=args.lr,weight_decay = 0.0001)
        self.schedule=lr_scheduler.StepLR(self.optimizer,10,0.1)

    def forward(self,d_model,ehrs,hier_labels):

        paths_hops=[]
        ehrs_hops=[]
        ground_truths_hops=[]
        # 随机选出batch条路径
        ehrs, randomPaths = random_sample(hier_labels, ehrs)
        true_label_level_0 = [row[1] for row in randomPaths]
        true_label_level_1 = [row[2] for row in randomPaths]
        true_label_level_2 = [row[3] for row in randomPaths]
        true_label_level_3 = [row[4] for row in randomPaths]

        # 针对batch 中的每个样本(每个电子病历)进行单独的取样
        # 1.得到电子病历的表示
        ehrs = torch.Tensor(ehrs).long().to(self.args.device)
        ehrRrep = self.cnn(ehrs) #[64,300]
        #print('ehrRrep:',ehrRrep)
        paths = [[self.args.node2id.get('ROOT')] for i in range(len(ehrs))]
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(paths, self.args) #[64,2023] [64]
        rewards_episode=[[] for i in range(len(ehrs))]
        log_action_episode=torch.zeros((len(ehrs),self.args.hops)).float().to(self.args.device)
        batchRewards=0.0

        # hop == 0
        actions,log_action= self.ActionSelection.act(self.pathEncoder,ehrRrep,pathRep,children,hop=0)  # [64,229],[64]
        # print('actions:',actions)
        # print('log_action:',log_action)
        # 执行选择出的action 得到状态和reward值
        paths=[row+[action] for row,action in zip(paths,actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_0)
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist() #[64,1]
        #rewards=rewards_groundTruth
        #rewards=rewards
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        #print('log_action_episode[:,0]:',log_action_episode[:,0])
        log_action_episode[:,0]=log_action

        # hop==1
        #根据新的状态向下选择action
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(actions, self.args) #[64,2023] [64]
        actions,log_action = self.ActionSelection.act(self.pathEncoder,ehrRrep, pathRep,children, hop=1)  # [64,229],[64,1]
        # 执行选择出的action 得到状态和reward值
        paths = [row + [action] for row, action in zip(paths, actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_1)
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist()
        #rewards = rewards_groundTruth
        # print('reward:', rewards)
        # print('true:', rewards_groundTruth)
        batchRewards =batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        log_action_episode[:,1]=log_action

        # hop==2
        #根据新的状态向下选择action
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(actions, self.args) #[64,2023] [64]
        actions,log_action = self.ActionSelection.act(self.pathEncoder,ehrRrep,pathRep,children, hop=2)  # [64,229],[64,1]
        # print('hop:2,actions:', actions)
        # print('hop:2,state:', state)
        # 执行选择出的action 得到状态和reward值
        paths=[row+[action] for row,action in zip(paths,actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_2)
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist()
        #rewards = rewards_groundTruth
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        log_action_episode[:,2]=log_action

        # hop==3
        #根据新的状态向下选择action
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(actions, self.args) #[64,2023] [64]
        actions,log_action  = self.ActionSelection.act(self.pathEncoder,ehrRrep, pathRep,children,hop=3)  # [64,229],[64,1]
        # print('hop:3,actions:', actions)
        # print('hop:3,state:', state)
        # 执行选择出的action 得到状态和reward值
        paths=[row+[action] for row,action in zip(paths,actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())

        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_3)
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist()
        #rewards = rewards_groundTruth
        batchRewards =batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        log_action_episode[:,3]=log_action

        #print('generated paths:',paths)

        #最终得到的paths就是agent 为每个样本预测出的paths
        return batchRewards,rewards_episode,log_action_episode,ehrs_hops,paths_hops,ground_truths_hops


    def getReward(self,actions,hier_labels):
        # 通过层次计算是否给予奖励值
        rewards=[[1] if p==h else [0] for p,h in zip(actions,hier_labels)]
        return rewards

    def update_policy(self,d_model,rewards_episode,log_action,ehrs_hops, paths_n):

        # batch_rewards=[]
        # # 针对该batch中的每一个样本
        # for i in range(len(rewards_episode)):
        #     R = 0
        #     rewards = []
        #     # Discount future rewards back to the present using gamma
        #     for r in rewards_episode[i][::-1]:
        #         R = r + self.gamma * R
        #         rewards.insert(0, R)
        #     batch_rewards.append(rewards)

        # Scale rewards
        batch_rewards = torch.FloatTensor(rewards_episode).float().to(self.args.device) #[64,4]

        # Calculate loss
        pg_loss = torch.mean(torch.mul(log_action, batch_rewards))
        self.optimizer.zero_grad()
        pg_loss.backward()

        self.optimizer.step()
        return pg_loss.item()


    # 传入真实的action 得到真实的（state,path）对
    def teacher_force(self,d_model,ehrs,hier_labels):
        ###############################################
        paths_hops=[]
        ehrs_hops=[]
        ground_truths_hops=[]
        # 随机选出batch条路径
        ehrs, randomPaths = random_sample(hier_labels, ehrs)
        true_label_level_0 = [row[1] for row in randomPaths]
        true_label_level_1 = [row[2] for row in randomPaths]
        true_label_level_2 = [row[3] for row in randomPaths]
        true_label_level_3 = [row[4] for row in randomPaths]

        # 针对batch 中的每个样本(每个电子病历)进行单独的取样
        # 1.得到电子病历的表示
        ehrs = torch.Tensor(ehrs).long().to(self.args.device)
        ehrRrep = self.cnn(ehrs) #[64,300]
        #print('ehrRrep:',ehrRrep)
        paths = [[self.args.node2id.get('ROOT')] for i in range(len(ehrs))]
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(paths, self.args) #[64,2023] [64]
        rewards_episode=[[] for i in range(len(ehrs))]
        log_action_episode=torch.zeros((len(ehrs),self.args.hops)).float().to(self.args.device)
        batchRewards=0.0

        # hop == 0
        actions,log_action= self.ActionSelection.act(self.pathEncoder,ehrRrep , pathRep,children,hop=0)  # [64,229],[64]

        # 执行选择出的action 得到状态和reward值
        paths=[row+[action] for row,action in zip(paths,true_label_level_0)]

        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_0 )
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist() #[64,1]
        #rewards = rewards_groundTruth
        #rewards=rewards
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        #print('log_action_episode[:,0]:',log_action_episode[:,0])
        log_action_episode[:,0]=log_action

        # hop==1
        #根据新的状态向下选择action
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(true_label_level_0, self.args) #[64,2023] [64]
        actions,log_action = self.ActionSelection.act(self.pathEncoder,ehrRrep, pathRep,children,hop=1)  # [64,229],[64,1]
        # print('hop:1,actions:', actions)
        # print('hop:1,state:', state)
        # 执行选择出的action 得到状态和reward值
        paths = [row + [action] for row, action in zip(paths, true_label_level_1)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_1)
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist()
        #rewards = rewards_groundTruth
        batchRewards =batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        log_action_episode[:,1]=log_action

        # hop==2
        #根据新的状态向下选择action
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(true_label_level_1, self.args) #[64,2023] [64]
        actions,log_action = self.ActionSelection.act(self.pathEncoder,ehrRrep, pathRep,children,hop=2)  # [64,229],[64,1]
        # print('hop:2,actions:', actions)
        # print('hop:2,state:', state)
        # 执行选择出的action 得到状态和reward值
        paths=[row+[action] for row,action in zip(paths,true_label_level_2)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_2)
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist()
        #rewards = rewards_groundTruth
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        log_action_episode[:,2]=log_action


        # hop==3
        #根据新的状态向下选择action
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(true_label_level_2, self.args) #[64,2023] [64]
        actions,log_action  = self.ActionSelection.act(self.pathEncoder,ehrRrep, pathRep,children, hop=3)  # [64,229],[64,1]
        # print('hop:3,actions:', actions)
        # print('hop:3,state:', state)
        # 执行选择出的action 得到状态和reward值
        paths=[row+[action] for row,action in zip(paths,true_label_level_3)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())

        #得到reward
        rewards_groundTruth=self.getReward(actions,true_label_level_3)
        ground_truths_hops.append(rewards_groundTruth)
        rewards=d_model.batchClassify(self,ehrs,paths).detach().cpu().numpy().tolist()
        #rewards = rewards_groundTruth
        batchRewards =batchRewards + sum([sum(r) for r in rewards])
        rewards_episode=[row+r for row,r in zip(rewards_episode,rewards)]
        log_action_episode[:,3]=log_action

        return batchRewards,rewards_episode,log_action_episode,ehrs_hops,paths_hops,ground_truths_hops


def getHopAction(hier_labels,hop):
    selected_actions = [list(set([row[hop] for row in sample])) for sample in
                        hier_labels]  # selected_actions:[[**,**,**],[**,**,**,**],..]
    return selected_actions

def getHopActionNext(paths, hier_labels,hop,args):
    new_hier_labels = []
    for sample in hier_labels:
        new_hier_labels.append([row[:hop + 1] for row in sample])
    next_paths=[]
    next_actions=[]
    #print('new_hier_labels:',new_hier_labels)
    for i in range(len(paths)): # 找见以每一条selected_actions为开头的路径（注：可能有多条）
        #print('selected_actions:',selected_actions[i])
        selected_paths=[row for row in new_hier_labels[i] if row[:hop]==paths[i][1:]]
        selected_path = random.choice(selected_paths)
        selected_path = [args.node2id.get('ROOT')] +selected_path
        next_paths.append(selected_path)
        next_actions.append([selected_path[-1]])

    return next_paths,next_actions


def action_space(parients,args):
    childrens=torch.zeros((len(parients),len(args.node2id))).float().to(args.device) #[64,229]
    children_lens=torch.zeros(len(parients)).long().to(args.device) #[64]
    for sample_i in range(len(parients)):
        children=args.adj[parients[sample_i]]
        children_len = len(torch.nonzero(children))
        childrens[sample_i]=children
        children_lens[sample_i]=children_len
    return childrens,children_lens


def generate_samples(gen_model,d_model,ehrs,hier_labels):
    # 从g_model中生成state,hidden(路径的表示)
    batchRewards_n,rewards_episode_n,log_action_episode_n,ehrs_hops,paths_n,ground_truths_hops= gen_model(d_model,ehrs,hier_labels)
    batchRewards_p, rewards_episode_p, log_action_episode_p, ehrs_hops, paths_p, ground_truths_hops=gen_model.teacher_force(d_model,ehrs, hier_labels)

    return batchRewards_n,log_action_episode_n,log_action_episode_p,rewards_episode_n,rewards_episode_p,ehrs_hops, paths_n,ground_truths_hops,paths_p

def random_sample(paths,ehrs):
    selected_paths=[]
    ehrs_=[]
    # 从每个样本的paths中选择出一条路径
    for i in range(len(paths)):
        # for j in range(len(paths[i])):
        path = random.choice(paths[i])
        selected_paths.append(path)
        ehrs_.append(ehrs[i])
    return ehrs_,selected_paths





