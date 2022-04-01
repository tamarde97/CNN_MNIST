#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File :           train.py
# @CreatTime :      2022/3/24 15:54
# @Author  :        Xu Jun
# @Function: this file used for  训练模型并保存

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import sys

sys.path.append("./model.py")
from model import LeNet


batch_size = 128
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNet().to(device)

# print("device: ", device, "\nmodel: ", model)

train_dataset = torchvision.datasets.MNIST("./data",
										   download=False,
										   train=True,
										   transform=transforms.ToTensor())

train_data_loader = Data.DataLoader(train_dataset,
									batch_size=batch_size,
									shuffle=True)

# 计算损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optim = torch.optim.Adam(model.parameters())

# 开始训练模型
for epoch in range(epochs):

	for i, (images, labels) in enumerate(train_data_loader):

		images = images.to(device)
		labels = labels.to(device)

		out = model(images)

		loss = criterion(out, labels)

		optim.zero_grad()

		loss.backward()

		optim.step()

	print(f"第{epoch + 1}轮 loss: {loss.item()}")

# 训练好模型，保存
torch.save(model, "./LeNet.pkl")
print("模型已保存")
