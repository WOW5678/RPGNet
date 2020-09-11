# -*- coding:utf-8 -*-
"""
@Time: 2019/12/22 19:55
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

x=[[1,2],[3,4,5]]
x=[torch.tensor(row) for row in x]
x=[F.pad(row,(0,3-len(row))) for row in x]
print(x)