#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/7 16:55
# @Author  : Whu_DSP
# @File    : vgg19_loss.py

# reference: https://github.com/hyeongyuy/CT-WGAN_VGG_tensorflow/blob/master/WGAN_VGG/code/wgan_vgg_model.py

import os
import tensorflow as tf
import numpy as np


class Vgg19:
    def __init__(self, size = 100, vgg_path = '.'):
        self.size = size
        self.VGG_MEAN = [103.939, 116.779, 123.68]

        vgg19_npy_path = os.path.join(vgg_path, "vgg19.npy")
        self.data_dict  = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")


    def extract_feature(self, rgb):
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert green.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert blue.get_shape().as_list()[1:] == [self.size, self.size, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        print(bgr.get_shape().as_list()[1:])
        assert bgr.get_shape().as_list()[1:] == [self.size, self.size, 3]


        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')
        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')
        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, 'pool3')
        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4, 'pool4')
        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        return conv5_4


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")