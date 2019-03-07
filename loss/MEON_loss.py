#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created MEON_loss.py by rjw at 19-1-21 in WHU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
from scipy import misc
from scipy import stats
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import gdn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MEON_eval(object):
    def __init__(self, height=256, width=256, channel=3,
                 dist_num=5, checkpoint_dir='./weights/'):
        """

                Args:
                    height: height of image
                    width: width of image
                    channel: number of color channel
                    dist_num: number of distortion types
                    checkpoint_dir: parameter saving directory

                """
        self.sess = tf.Session()
        self.height = height
        self.width = width
        self.channel = channel
        self.checkpoint_dir = checkpoint_dir
        self.dist_num = dist_num
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name='test_image')

        self.build_model()

    def build_model(self):

        # params for convolutional layers
        width1 = 5
        height1 = 5
        stride1 = 2
        depth1 = 8

        width2 = 5
        height2 = 5
        stride2 = 2
        depth2 = 16

        width3 = 5
        height3 = 5
        stride3 = 2
        depth3 = 32

        width4 = 3
        height4 = 3
        stride4 = 1
        depth4 = 64

        # params for fully-connected layers
        sub1_fc1 = 128
        sub1_fc2 = self.dist_num

        sub2_fc1 = 256
        sub2_fc2 = self.dist_num

        # convolution layer 1
        with tf.variable_scope('conv1'):
            weights = tf.get_variable(name='weights',
                                      shape=[height1, width1, self.channel, depth1],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth1],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(self.x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride1, stride1, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='pool_x')

        # convolution layer 2
        with tf.variable_scope('conv2'):
            weights = tf.get_variable(name='weights',
                                      shape=[height2, width2, depth1, depth2],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth2],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(pool_x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride2, stride2, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_x')

        # convolution layer 3
        with tf.variable_scope('conv3'):
            weights = tf.get_variable(name='weights',
                                      shape=[height3, width3, depth2, depth3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(pool_x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride3, stride3, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_x')

        # convolution layer 4
        with tf.variable_scope('conv4'):
            weights = tf.get_variable(name='weights',
                                      shape=[height4, width4, depth3, depth4],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth4],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))

            conv_x = tf.nn.conv2d(input=pool_x, filter=weights, padding='VALID', strides=[1, stride4, stride4, 1],
                                  name='conv_x') +biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            conv_out_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_x')

        # subtask 1
        with tf.variable_scope('subtask1'):
            with tf.variable_scope('fc1'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1,depth4, sub1_fc1],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(depth4))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub1_fc1],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                fc_x = tf.nn.conv2d(input=conv_out_x, filter=weights, padding='VALID', strides=[1,1,1,1], name='fc_x') +biases
                gdn_x = gdn(inputs=fc_x, inverse=False, data_format='channels_last', name='gdn_x')

            with tf.variable_scope('fc2'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1, sub1_fc1,sub1_fc2],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(sub1_fc1))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub1_fc2],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                out_x = tf.squeeze(tf.nn.conv2d(input=gdn_x, filter=weights, padding='VALID', strides=[1,1,1,1])+biases,
                                   name='out_x')

            self.probs = tf.nn.softmax(out_x,name='dist_prob')

        # subtask 2
        with tf.variable_scope('subtask2'):
            with tf.variable_scope('fc1'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1,depth4, sub2_fc1],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(depth4))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub2_fc1],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                fc_x = tf.nn.conv2d(input=conv_out_x, filter=weights, padding='VALID', strides=[1, 1, 1, 1],
                                    name='fc_x') + biases
                gdn_x = gdn(inputs=fc_x , inverse=False, data_format='channels_last', name='gdn_x')

            with tf.variable_scope('fc2'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1,sub2_fc1, sub2_fc2],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / math.sqrt(float(sub2_fc1))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub2_fc2],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                self.q_scores = tf.squeeze(tf.nn.conv2d(input=gdn_x, filter=weights, padding='VALID',
                                                        strides=[1,1,1,1])+biases, name='q_scores')
                self.out_q = tf.reduce_sum(tf.multiply(self.probs, self.q_scores),
                                           axis=1, keep_dims=False, name='out_q')

            self.saver = tf.train.Saver()


    def initialize(self):
        ckpt_path = self.checkpoint_dir

        could_load, checkpoint_counter = self.__load__(ckpt_path)
        if could_load:
            counter = checkpoint_counter
            print('Load successfully!')
        else:
            raise IOError('Fail to load the pretrained model')

        check_init = tf.report_uninitialized_variables()
        assert self.sess.run(check_init).size == 0


    def predict(self, img_names=None, quiet=True):
        '''
        Args:
            param img_names: list of test image names (with directory)
            quiet: keep quiet while running
        return:
            predicted_distortion_types: A list of the the length as img_names
            predicted scores: A list of the same length as img_names

        '''

        if not isinstance(img_names, str) and not isinstance(img_names, list):
            raise ValueError('The arg is neither an image name nor a list of image names')
        if isinstance(img_names, str):
            img_names = [img_names,]

        p_labels = []
        p_scores = []
        for file_name in img_names:
            try:
                img = misc.imread(file_name)
            except Exception:
                raise IOError('Fail to load image: %s' % file_name)

            patches = self.__generate_patches__(img, input_size=self.height)
            patch_probs, patch_qs = self.sess.run([self.probs,self.out_q],feed_dict={self.x: patches})
            patch_types = [p.argmax() for p in patch_probs]
            img_type, _ = stats.mode(patch_types)

            p_labels.append(img_type[0]+1)
            p_scores.append(np.mean(patch_qs))
            if not quiet:
                print("%s: %.4f" % (file_name, p_scores[-1]))

        return p_labels, p_scores


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format('CSIQ', self.dist_num, 'distortions', 'final')  # needs further revision


    def __load__(self, ckpt_dir):
        import re
        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(ckpt_dir, self.model_dir)
        checkpoint_dir = ckpt_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [!] Failed to find a checkpoint")
            return False, 0

    def __generate_patches__(self, img, input_size=256, type=np.float32):
        img_shape = img.shape
        img = img.astype(dtype=type)
        if len(img_shape) == 2:
            H, W = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img, ] * 3, dtype=img.dtype)
            ch = 3


        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches = [img[hId:hId + input_size, wId:wId + input_size, :]
                   for hId in hIdx
                   for wId in wIdx]

        return patches


    def close(self):
        self.sess.close()
        tf.reset_default_graph()

def demo():
    MEON_evaluate_model = MEON_eval()        # build the tensorflow Graph of MEON
    MEON_evaluate_model.initialize()         # initialize with pretrained weights

    p_labels, p_scores = MEON_evaluate_model.predict('./imgs/0.jpg')   # Example of predicting quality of one image
    print(p_labels)                # [5]
    print(p_scores)                # [2.4722667]

    img_list = ['./imgs/{}.bmp'.format(i+1) for i in range(4)]
    p_labels, p_scores = MEON_evaluate_model.predict(img_list)  # Example of predicting quality of a list of images
    print(p_labels)                 # [5, 2, 1, 3]
    print(p_scores)                 # [2.4722667, 45.239239, 77.436516, 92.17942]

    MEON_evaluate_model.close()     # close session and reset default graph before delete this MEON model instance
    del MEON_evaluate_model


def MEON_loss(file_name, checkpoint_dir):
    try:
        img = misc.imread(file_name)
    except Exception:
        raise IOError('Fail to load image: %s' % file_name)
    MEON_evaluate_model = MEON_eval(checkpoint_dir=checkpoint_dir)
    MEON_evaluate_model.initialize()  # initialize with pretrained weights

    patches = MEON_evaluate_model.__generate_patches__(img, input_size=MEON_evaluate_model.height)
    patch_probs, patch_qs = MEON_evaluate_model.sess.run([MEON_evaluate_model.probs, MEON_evaluate_model.out_q], feed_dict={MEON_evaluate_model.x: patches})
    patch_types = [p.argmax() for p in patch_probs]
    img_type, _ = stats.mode(patch_types)

    return np.mean(patch_qs)


if __name__ == '__main__':
    # demo()
    p_scores = MEON_loss('../demo/imgs/1.bmp','../data/weights/')
    print(p_scores)
