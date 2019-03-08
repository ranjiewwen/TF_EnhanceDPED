#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/8 16:07
# @Author  : Whu_DSP
# @File    : print_graph.py

# https://blog.csdn.net/haima1998/article/details/80297710

from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path = os.path.join("MEON-0")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

vars = []
for key in var_to_shape_map:
    print("tensor_name: ", key)
    vars.append(key)

print(len(vars))
