# -*- coding:utf-8 -*-
'''
Create time: 2019/11/1 0:46
@Author: 大丫头
'''
import torch
import torch.nn.functional as F
import generator
from keras.utils import to_categorical
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import matthews_corrcoef

from collections import Counter
def balance_sample(ehrs,hier_labels,index=1):
    ground_truths=[]
    all_labels=[]
    label_sample_dict = {}

    for ehr,sample in zip(ehrs,hier_labels):
        label=list(set([row[index] for row in sample]))
        ground_truths.append(label)
        all_labels.extend(label)
        for l in label:
            if l not in label_sample_dict:
                label_sample_dict[l]=[]
                label_sample_dict[l].append((ehr,sample))
            else:
                #不能按照顺序插入 而应该是按照标签分布进行进行插入
                if len(label)==1:
                    label_sample_dict[l].insert(0,(ehr,sample))
                else:
                    label_sample_dict[l].append((ehr, sample))


    label_counter=Counter(all_labels).most_common()
    #print('ground_truths:',ground_truths)
    #print('label_counter:',label_counter)
    label_counter = sorted(label_counter, key=lambda tup:tup[1], reverse=False)
    #print('label_counter:', label_counter)
    mini_label,mini_num=label_counter[0][0],label_counter[0][1]
    current_mini_num=mini_num
    #print(mini_label, mini_num)

    selected_sample=[]
    for tuple in label_counter:
        def label_static(selected_sample):
            # 统计一下index的label分布 看是否平衡了一些
            all_labels = []
            for sample in selected_sample:
                label = list(set([row[index] for row in sample[1]]))
                all_labels.extend(label)
            label_counter = Counter(all_labels).most_common()
            #转化为字典
            label_dict={key:value for key,value in label_counter}
            return label_dict

        if len(selected_sample)>0:
            curent_label_dict = label_static(selected_sample)
            if curent_label_dict.get(tuple[0]):
                if curent_label_dict.get(tuple[0]) >= mini_num:
                    continue
                else:
                    current_mini_num = mini_num - curent_label_dict.get(tuple[0])
            else:
                current_mini_num=mini_num
                # 找出tuple[0]对应的样本
        samples=label_sample_dict.get(tuple[0])
        # 从value中任选mini_label个样本
        selected_sample= selected_sample + samples[:current_mini_num]
        # 统计此时的标签分布 若有些标签已经达到最大的个数则停止继续选择
    # 从selected_sample中解析出ehr 和hier_labels
    selected_ehrs = [row[0] for row in selected_sample]
    selected_hier_labels = [row[1] for row in selected_sample]
    print('batch Number:', len(selected_sample))
    return selected_ehrs,selected_hier_labels

def run_pre_train_step_v2(gen_model,g_batcher,epochs):
    for epoch in range(epochs):
        #gen_model.schedule.step()
        print('lr:', gen_model.optimizer.state_dict()['param_groups'][0]['lr'])

        batches = g_batcher.get_batches(mode='train')
        for batch_num in range(len(batches)):
            # for batch_num in range(len(batches)):
            current_batch = batches[batch_num]

            ehrs = [example.ehr for example in current_batch]
            # 对hier_labels进行填充
            hier_labels = [example.hier_labels for example in current_batch]
            #采样 将每个标签的训练样本个数保持均衡
            #ehrs_0,hier_labels_0=balance_sample(ehrs,hier_labels,index=1)
            ehrs, selected_paths, hier_labels  = random_sample(hier_labels,ehrs)
            # hop == 0
            # 1.得到电子病历的表示
            ehrs_0_tensor = torch.Tensor(ehrs).long().to(gen_model.args.device)
            ehrRrep = gen_model.cnn(ehrs_0_tensor)  # [64,300]
            paths= [[gen_model.args.node2id.get('ROOT')] for i in range(len(ehrs))]
            pathRep = gen_model.pathEncoder(paths)  # [64,600]
            children, children_len =generator. action_space(paths, gen_model.args)  # [64,2023] [64]
            # 初始state  没有路径 仅适用EHR的表示
            parients = [gen_model.args.node2id.get('ROOT') for i in range(len(ehrs))]
            true_label_level_0_all = find_true_children(parients, hier_labels, hop=0)
            true_label_level_0_all=label_one_hot(true_label_level_0_all,class_num=len(gen_model.args.node2id))
            log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder,ehrRrep,pathRep,true_label_level_0_all,children,hop=0)  # [64,229],[64]

            loss=focal_loss(gen_model, log_action, true_label_level_0_all)

            
            # hop==1
            #ehrs,randomPaths,hier_labels=random_sample(hier_labels_0,ehrs_0)
            randomPaths_1=[[row[1]] for row in selected_paths]
            # 根据新的状态向下选择action
            pathRep = gen_model.pathEncoder(randomPaths_1)  # [64,600]

            parients= [row[-1] for row in randomPaths_1]
            true_label_level_1_all = find_true_children(parients, hier_labels, hop=1)
            true_label_level_1_all = label_one_hot(true_label_level_1_all, class_num=len(gen_model.args.node2id))
            children, children_len =generator. action_space(parients, gen_model.args)  # [64,2023] [64]
            log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder,ehrRrep,pathRep,true_label_level_1_all,children,hop=1)  # [64,229],[64,1]

            loss += focal_loss(gen_model, log_action, true_label_level_1_all)


            # hop==2

            # 根据新的状态向下选择action
            randomPaths_2 = [[row[2]] for row in selected_paths]
            pathRep = gen_model.pathEncoder(randomPaths_2)  # [64,100]
            parients = [row[-1] for row in randomPaths_2]
            children, children_len = generator.action_space(parients, gen_model.args)  # [64,2023] [64]

            #print('actions_2:', actions)
            true_label_level_2_all = find_true_children(parients, hier_labels, hop=2)
            true_label_level_2_all = label_one_hot(true_label_level_2_all, class_num=len(gen_model.args.node2id))
            log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder, ehrRrep, pathRep,true_label_level_2_all, children,hop=2)  # [64,229],[64,1]
            loss += focal_loss(gen_model, log_action, true_label_level_2_all)

            # hop==3
            randomPaths_3= [[row[3]] for row in selected_paths]
            pathRep = gen_model.pathEncoder(randomPaths_3)  # [64,600]
            parients = [row[-1] for row in randomPaths_3]
            children, children_len = generator.action_space(parients, gen_model.args)  # [64,2023] [64]


            true_label_level_3_all = find_true_children(parients, hier_labels, hop=3)
            true_label_level_3_all = label_one_hot(true_label_level_3_all, class_num=len(gen_model.args.node2id))
            log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder, ehrRrep, pathRep, true_label_level_3_all,children, hop=3)  # [64,229],[64,1]
            loss += focal_loss(gen_model, log_action, true_label_level_3_all)


            print('epoch:{},loss:{:.4f}'.format(epoch,loss.item()))
            gen_model.optimizer.zero_grad()
            loss.backward()
            gen_model.optimizer.step()

def run_pre_train_step_batch(gen_model,ehrs, hier_labels):
    ehrs, selected_paths, hier_labels = random_sample(hier_labels, ehrs)
    # hop == 0
    # 1.得到电子病历的表示
    ehrs_0_tensor = torch.Tensor(ehrs).long().to(gen_model.args.device)
    ehrRrep = gen_model.cnn(ehrs_0_tensor)  # [64,300]
    paths = [[gen_model.args.node2id.get('ROOT')] for i in range(len(ehrs))]
    pathRep = gen_model.pathEncoder(paths)  # [64,600]
    children, children_len = generator.action_space(paths, gen_model.args)  # [64,2023] [64]
    # 初始state  没有路径 仅适用EHR的表示
    parients = [gen_model.args.node2id.get('ROOT') for i in range(len(ehrs))]
    true_label_level_0_all = find_true_children(parients, hier_labels, hop=0)
    true_label_level_0_all = label_one_hot(true_label_level_0_all, class_num=len(gen_model.args.node2id))
    log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder, ehrRrep, pathRep, true_label_level_0_all,
                                                   children, hop=0)  # [64,229],[64]

    loss = focal_loss(gen_model, log_action, true_label_level_0_all)

    # hop==1
    # ehrs,randomPaths,hier_labels=random_sample(hier_labels_0,ehrs_0)
    randomPaths_1 = [[row[1]] for row in selected_paths]
    # 根据新的状态向下选择action
    pathRep = gen_model.pathEncoder(randomPaths_1)  # [64,600]

    parients = [row[-1] for row in randomPaths_1]
    true_label_level_1_all = find_true_children(parients, hier_labels, hop=1)
    true_label_level_1_all = label_one_hot(true_label_level_1_all, class_num=len(gen_model.args.node2id))
    children, children_len = generator.action_space(parients, gen_model.args)  # [64,2023] [64]
    log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder, ehrRrep, pathRep, true_label_level_1_all,
                                                   children, hop=1)  # [64,229],[64,1]

    loss += focal_loss(gen_model, log_action, true_label_level_1_all)

    # hop==2

    # 根据新的状态向下选择action
    randomPaths_2 = [[row[2]] for row in selected_paths]
    pathRep = gen_model.pathEncoder(randomPaths_2)  # [64,100]
    parients = [row[-1] for row in randomPaths_2]
    children, children_len = generator.action_space(parients, gen_model.args)  # [64,2023] [64]

    # print('actions_2:', actions)
    true_label_level_2_all = find_true_children(parients, hier_labels, hop=2)
    true_label_level_2_all = label_one_hot(true_label_level_2_all, class_num=len(gen_model.args.node2id))
    log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder, ehrRrep, pathRep, true_label_level_2_all,
                                                   children, hop=2)  # [64,229],[64,1]
    loss += focal_loss(gen_model, log_action, true_label_level_2_all)

    # hop==3
    randomPaths_3 = [[row[3]] for row in selected_paths]
    pathRep = gen_model.pathEncoder(randomPaths_3)  # [64,600]
    parients = [row[-1] for row in randomPaths_3]
    children, children_len = generator.action_space(parients, gen_model.args)  # [64,2023] [64]

    true_label_level_3_all = find_true_children(parients, hier_labels, hop=3)
    true_label_level_3_all = label_one_hot(true_label_level_3_all, class_num=len(gen_model.args.node2id))
    log_action = gen_model.ActionSelection.pre_act(gen_model.pathEncoder, ehrRrep, pathRep, true_label_level_3_all,
                                                   children, hop=3)  # [64,229],[64,1]
    loss += focal_loss(gen_model, log_action, true_label_level_3_all)

    print('loss:{:.4f}'.format(loss.item()))
    gen_model.optimizer.zero_grad()
    loss.backward()
    gen_model.optimizer.step()
    del loss, ehrs_0_tensor,ehrRrep ,pathRep

def train_level(gen_model,logist,true_labels_level):

    labels = label_one_hot(true_labels_level,len(gen_model.args.node2id))
    labels=torch.Tensor(labels).to(gen_model.args.device)
    #weight=torch.tensor(weights[i]).to(gen_model.args.device)
    #loss=loss + F.nll_loss(log_action[i].unsqueeze(0), ground_truth,weight=weight)
    loss = F.binary_cross_entropy(logist,labels)
    #loss=F.mse_loss(logist,labels)
    return loss

def focal_loss(gen_model, logist, true_labels_level, eps=1e-8, gamma=2, alpha = 0.1):
    labels = torch.Tensor(true_labels_level).to(gen_model.args.device)

    probs = torch.clamp(logist, eps, 1-eps)
    loss = -  ((1 - alpha) * torch.pow((1 - probs),gamma) * labels * torch.log(probs) + alpha * torch.pow(probs, gamma) * (1 - labels) * torch.log(1 - probs))
    loss = loss.sum()
    return loss / labels.size(0)

def find_true_children(parients,hier_label,hop):
    childrens=[]
    for parient,rows in zip(parients,hier_label):
        true_children = []
        for row in rows:
            if parient==row[hop]:
                true_children.append(row[hop+1])
        childrens.append(list(set(true_children)))
    return  childrens

def random_sample(paths,ehrs):
    selected_paths=[]
    ehrs_=[]
    hier_labels=[]
    # 从每个样本的paths中选择出一条路径
    for i in range(len(paths)):
        # 进行数据扩展
        # for j in range(len(paths[i])):
        path = random.choice(paths[i])
        ehrs_.append(ehrs[i])
        selected_paths.append(path)
        hier_labels.append(paths[i])
    return ehrs_,selected_paths,hier_labels

def random_sample_d(paths):
    import random
    selected_paths = []
    for i in range(len(paths)):
        path=random.choice(paths[i])
        selected_paths.append(path)
    return selected_paths

def label_one_hot(patientLabels,class_num):
    # 将labels从ID转换为multi-hot编码
    labels = []
    for row in patientLabels:
        temp = np.zeros(class_num)
        temp[row] = 1
        labels.append(temp)
    return np.array(labels)