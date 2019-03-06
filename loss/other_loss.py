import tensorflow as tf
import utils.utils as utils

def color_loss(enhanced, dslr_image, batch_size):

    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)
    # dslr_blur=dslr_image
    # enhanced_blur=enhanced

    loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2)) / (2 * batch_size)
    return loss_color

def variation_loss(enhanced,PATCH_WIDTH,PATCH_HEIGHT,batch_size):
    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:, 1:, :, :])
    tv_x_size = utils._tensor_size(enhanced[:, :, 1:, :])
    y_tv = tf.nn.l2_loss(enhanced[:, 1:, :, :] - enhanced[:, :batch_shape[1] - 1, :, :])
    x_tv = tf.nn.l2_loss(enhanced[:, :, 1:, :] - enhanced[:, :, :batch_shape[2] - 1, :])
    loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size
    return loss_tv