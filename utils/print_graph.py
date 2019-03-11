#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/8 16:07
# @Author  : Whu_DSP
# @File    : print_graph.py

# https://blog.csdn.net/haima1998/article/details/80297710

'''
1. 获取graph变量
        # optimize parameters of image enhancement (generator) and discriminator networks
        generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
        discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]
        meon_vars = [v for v in tf.global_variables() if v.name.startswith("conv") or v.name.startswith("subtask")]
        
2. 
        # 得到该网络中，所有可以加载的参数; 用var_list = tf.contrib.framework.get_variables(scope_name)获取指定scope_name下的变量，
        variables = tf.contrib.framework.get_variables_to_restore()
        # 删除output层中的参数
        variables_to_resotre = [v for v in variables if v.name.startswith("conv")]
        # 构建这部分参数的
        pre_saver = tf.train.Saver(variables_to_resotre)
3.
        3.1.获取某个操作之后的输出;用graph.get_operations()获取所有op
        3.2.获取指定的var的值;用GraphKeys获取变量;tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)返回指定集合的变量
        3.3.获取指定scope的collection;tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope='common_conv_xxx_net.final_logits')

'''

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
