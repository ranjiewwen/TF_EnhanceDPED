
import tensorflow as tf
import tensorflow.contrib.slim as slim

def Gaussian(input):
    return tf.exp(-tf.multiply(input,input))


def Modulecell(input_image,in_channels,out_channels,kernel_size):

    W1=weight_variable([kernel_size,kernel_size,in_channels,out_channels],name="W")
    b1=bias_variable([out_channels],name="b")
    x=conv2d(input_image,W1)+b1

    with tf.variable_scope("xUnit"):
        with tf.variable_scope('bn1'):
            #y1 = batch_normalization_layer(x, out_channels)
            y1=_instance_norm(x)
        y1 = tf.nn.relu(y1)
        W2 = weight_variable([kernel_size, kernel_size, out_channels, 1], name="W" )
        b2 = bias_variable([out_channels], name="b")
        y1 = tf.nn.depthwise_conv2d(y1, W2, [1, 1, 1, 1], padding="SAME") + b2
        with tf.variable_scope('bn2'):
            #y1 = batch_normalization_layer(y1, out_channels)
            y1=_instance_norm(y1)
        y = Gaussian(y1)

    out=tf.multiply(x,y)
    return out

def xResidualBlock(input_image,in_channels,out_channels,kernel):

    y1=Modulecell(input_image,in_channels,out_channels,kernel)

    W=weight_variable([kernel,kernel,in_channels,out_channels],name="W")
    b=bias_variable([out_channels],name="b")
    y1=conv2d(y1,W)+b
    #y1 = batch_normalization_layer(y1, out_channels)
    y1=_instance_norm(y1)

    return y1+input_image


def resnet_xUnit(input_image):

    with tf.variable_scope("generator"):

        # W1 = weight_variable([9, 9, 3, 16], name="W")
        # b1 = bias_variable([16], name="b")
        # x=tf.nn.conv2d(input_image, W1, strides=[1, 2, 2, 1], padding='SAME')

        x=slim.conv2d(input_image,16,[3,3])
        y=slim.conv2d(x,16,[3,3],stride=2)

        with tf.variable_scope('conv1'):
            c1=Modulecell(y,16,16,3)

        # residual 1
        with tf.variable_scope('res1'):
            c2=xResidualBlock(c1,16,16,3)

        # Convolutional
        with tf.variable_scope('conv3'):
            c7 = Modulecell(c2, 16, 16, 3)

        # Final
        with tf.variable_scope('out'):
            W = weight_variable([4, 4, 3, 16], name="W")
            # b = bias_variable([3], name="b")
        out=tf.nn.conv2d_transpose(y+c7, W,output_shape=tf.shape(input_image),strides=[1,2,2,1])
        out.set_shape([None, input_image.shape[1].value, input_image.shape[2].value, 3])

        out=slim.conv2d(out,3,[3,3],activation_fn=None)
        enhanced = tf.nn.tanh(out) * 0.58 + 0.5

    return enhanced


def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift


if __name__=="__main__":

    img1 = tf.constant(value=[[[[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]]]],
                       dtype=tf.float32)
    img2 = tf.constant(value=[[[[1], [1], [1], [1]], [[1], [1], [1], [1]], [[1], [1], [1], [1]], [[1], [1], [1], [1]]]],
                       dtype=tf.float32)
    img = tf.concat(values=[img1, img2,img1], axis=3)

    enhanced = resnet_xUnit(img)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        output = sess.run(enhanced)
        print(sess.run([tf.shape(img),tf.shape(enhanced)]))



