import tensorflow as tf
import numpy as np
import scipy.io
import utils.utils as utils

IMAGE_MEAN = np.array([123.68 ,  116.779,  103.939])

def net(path_to_vgg_net, input_image):

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(path_to_vgg_net)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
        elif layer_type == 'pool':
            current = _pool_layer(current)
        net[name] = current

    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def preprocess(image):
    return image - IMAGE_MEAN


def content_loss(vgg_dir,enhanced,dslr_image,batch_size):
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = net(vgg_dir, preprocess(enhanced * 255))
    dslr_vgg = net(vgg_dir, preprocess(dslr_image * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size
    return loss_content


# content loss
def multi_content_loss(vgg_dir,enhanced,dslr_image,batch_size):

    CONTENT_LAYER = ['relu1_2','relu3_4','relu5_4']
    enhanced_vgg = net(vgg_dir, preprocess(enhanced * 255))
    dslr_vgg = net(vgg_dir, preprocess(dslr_image * 255))

    multi_content_loss=0.0

    for i in range(len(CONTENT_LAYER)):
        content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER[i]]) * batch_size
        loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER[i]] - dslr_vgg[CONTENT_LAYER[i]]) / content_size
        multi_content_loss += tf.reduce_mean(loss_content)

    return multi_content_loss/3*0.001