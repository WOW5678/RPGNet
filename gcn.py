# -*- coding:utf-8 -*-
"""
@Time: 2019/09/13 11:13
@Author: Shanshan Wang
@Version: Python 3.7
@Function:  使用DGL框架实现图卷积操作，得到每个节点（ICD CODE）的向量表示
"""
import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        #print('feature:',feature.weight) #Embedding(9139, 100)
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.gcn1 = GCNLayer(300, 300, F.relu)
        self.gcn2 = GCNLayer(300, 300, F.relu)

    def forward(self, g, features):
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())

        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
# net = Net()
# print(net)