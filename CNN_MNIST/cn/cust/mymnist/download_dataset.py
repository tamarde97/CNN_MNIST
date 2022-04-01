#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File :           download_dataset.py
# @CreatTime :      2022/3/24 15:54
# @Author  :        Xu Jun
# @Function: this file used for  下载数据集


import torchvision


torchvision.datasets.MNIST("./data", download=True)