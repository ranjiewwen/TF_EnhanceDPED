#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/10 11:15
# @Author  : Whu_DSP
# @File    : dped_dataloader.py

import multiprocessing as mtp
import os
import cv2
import numpy as np
from scipy import misc


def parse_data(filename):
    I = np.asarray(misc.imread(filename))
    I = np.float16(I) / 255
    return I

class Dataloader:

    def __init__(self, dped_dir, type_phone, batch_size, is_training, im_shape):
        self.works = mtp.Pool(10)
        self.dped_dir = dped_dir
        self.phone_type = type_phone
        self.batch_size = batch_size
        self.is_training = is_training
        self.im_shape = im_shape
        self.image_list, self.dslr_list = self._get_data_file_list()

        self.num_images = len(self.image_list)
        self._cur = 0
        self._perm = None
        self._shuffle_index() # init order

    def _get_data_file_list(self):
        if self.is_training:
            directory_phone = os.path.join(self.dped_dir, str(self.phone_type), 'training_data', str(self.phone_type))
            directory_dslr = os.path.join(self.dped_dir, str(self.phone_type), 'training_data', 'canon')
        else:
            directory_phone = os.path.join(self.dped_dir, str(self.phone_type), 'test_data', 'patches',
                                           str(self.phone_type))
            directory_dslr = os.path.join(self.dped_dir, str(self.phone_type), 'test_data', 'patches', 'canon')

        # num_images = len([name for name in os.listdir(directory_phone) if os.path.isfile(os.path.join(directory_phone, name))])
        image_list = [os.path.join(directory_phone, name) for name in os.listdir(directory_phone)]
        dslr_list = [os.path.join(directory_dslr, name) for name in os.listdir(directory_dslr)]
        return image_list, dslr_list

    def _shuffle_index(self):
        '''randomly permute the train order'''
        self._perm = np.random.permutation(np.arange(self.num_images))
        self._cur = 0

    def _get_next_minbatch_index(self):
        """return the indices for the next minibatch"""
        if self._cur + self.batch_size > self.num_images:
            self._shuffle_index()
        next_index = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size
        return next_index

    def get_minibatch(self, minibatch_db):
        """return minibatch datas for train/test"""
        if self.is_training:
            jobs = self.works.map(parse_data, minibatch_db)
        else:
            jobs = self.works.map(parse_data, minibatch_db)
        index = 0
        images_data = np.zeros([self.batch_size, self.im_shape[0], self.im_shape[1], 3])
        for index_job in range(len(jobs)):
            images_data[index, :, :, :] = jobs[index_job]
            index += 1
        return images_data

    def next_batch(self):
        """Get next batch images and labels"""
        db_index = self._get_next_minbatch_index()
        minibatch_db = []
        for i in range(len(db_index)):
            minibatch_db.append(self.image_list[db_index[i]])

        minibatch_db_t = []
        for i in range(len(db_index)):
            minibatch_db_t.append(self.dslr_list[db_index[i]])

        images_data = self.get_minibatch(minibatch_db)
        dslr_data = self.get_minibatch(minibatch_db_t)

        return images_data, dslr_data


if __name__ == "__main__":

    data_dir = "F:\\ranjiewen\\TF_EnhanceDPED\\data\\dped"
    train_loader = Dataloader(data_dir, "iphone", 32, True,[100,100])
    test_loader = Dataloader(data_dir, "iphone", 32, False, [100, 100])

    for i in range(10):
        image_batch,label_batch = train_loader.next_batch()
        print(image_batch.shape,label_batch.shape)

        print("-------------------------------------------")
        image_batch,label_batch = test_loader.next_batch()
        print(image_batch.shape,label_batch.shape)
