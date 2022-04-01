#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File :           test.py
# @CreatTime :      2022/3/24 15:55
# @Author  :        Xu Jun
# @Function: this file used for  测试模型，计算准确率

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data


batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型，选择在cpu还是gpu上运行
model = torch.load("./LeNet.pkl", map_location=torch.device(device))
model = model.to(device)
# print("device: ", device, "\nmodel: ", model)

test_dataset = torchvision.datasets.MNIST("./data",
										  download=False,
										  train=False,
										  transform=transforms.ToTensor())

test_data_loader = Data.DataLoader(test_dataset,
								   batch_size=batch_size,
								   shuffle=True)

# test
model.eval()
with torch.no_grad():

	# 输出正确个数
	correct_cnt = 0
	# 采样总数
	total_cnt = 0

	for i, (images, labels) in enumerate(test_data_loader):

		images = images.to(device)
		labels = labels.to(device)

		out = model(images)

		out_flag = torch.argmax(out, axis=1)
		correct_cnt += (out_flag == labels).sum().item()

		total_cnt += labels.shape[0]

	print("准确率：", correct_cnt / total_cnt)

