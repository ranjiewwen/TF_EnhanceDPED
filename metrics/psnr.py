import tensorflow as tf
import numpy as np


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def PSNR_Orignal(enhanced, dslr_, PATCH_SIZE, batch_size):
    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])
    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2)) / (PATCH_SIZE * batch_size)
    loss_psnr = 20 * log10(1.0 / tf.sqrt(loss_mse))
    return loss_psnr

def PSNR(target,prediction,name=None):
    with tf.name_scope(name,default_name='psnr_op',values=[target,prediction]):
        squares=tf.square(target-prediction,name='squares')
        squares=tf.reshape(squares,[tf.shape(squares)[0],-1])
        #mean psnr over a batch
        p=tf.reduce_mean((-10/np.log(10))*tf.log(tf.reduce_mean(squares,axis=[1])))
    return p