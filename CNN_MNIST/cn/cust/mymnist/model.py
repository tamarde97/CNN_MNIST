#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File :           model.py
# @CreatTime :      2022/3/24 15:55
# @Author  :        Xu Jun
# @Function: this file used for  创建模型

import torch.nn as nn


class LeNet(nn.Module):

	def __init__(self):
		super(LeNet, self).__init__()
		# 第一层卷积池化
		self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2, stride=2))
		# 第二层卷积池化
		self.layer2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2, stride=2))
		# 数据reshape
		self.flatten = nn.Flatten()
		# 全连接层，输入是手动算出来的，隐藏层神经元100个，输出层神经元10个，写死
		self.linear = nn.Linear(32*7*7, 100)
		self.relu = nn.ReLU()
		# self.dropout = nn.Dropout()	TODO
		self.softmax = nn.Linear(100, 10)


	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.flatten(out)
		out = self.linear(out)
		out = self.relu(out)
		# out = self.dropout(out)	TODO
		out = self.softmax(out)

		return out
