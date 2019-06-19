import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import random

from keras.layers import Input, Convolution2D, BatchNormalization, MaxPooling2D, ZeroPadding2D, Lambda, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = indices[i: min(i + batch_size, num_examples)]
        yield features[j], labels[j]

def conv2d(x, filters, kernel_size, strides=1, padding_size=0, batch_normalization=True, activation=LeakyReLU, use_bias=True, name='conv2d'):
    if padding_size > 0:
        # x = np.pad(x, ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant')
        x = ZeroPadding2D(padding=(padding_size, padding_size), data_format='channels_last')(x)

    x = Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', activation=None, use_bias=use_bias,name=name)(x)
    if batch_normalization:
        x = BatchNormalization(axis=-1, momentum=0.9, trainable=False, name=f'{name}_bn')(x)
    if activation:
        x = activation()(x)
    return x

def max_pooling2d(x, pool_size, strides, name='maxpool'):
    return MaxPooling2D(pool_size=pool_size, strides=strides, name=name)(x)

def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)

def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2."""
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *input_shape[3]) if input_shape[1] else (input_shape[0], None, None,4 * input_shape[3])

def reorg(x, block_size):
    # print(x.shape[1])
    # result_size = (x.shape[0] * block_size ** 2, x.shape[1] / block_size, x.shape[2] / block_size, x.shape[3])
    # return np.reshape(x, result_size)
    return Lambda(space_to_depth_x2,
       output_shape=space_to_depth_x2_output_shape,
       name='space_to_depth')(x)


### YOLO V2使用的Darknet实现
class Darknet19(object):
    def __init__(self, images, labels):
        self.net = self.build_net(images)
        self.train_op = self.train(images, labels)

    def build_net(self, images, n_last_channels=425):
        input_shape = images[0].shape
        inputs = Input(shape=input_shape)
        net = conv2d(inputs, filters=32, kernel_size=3, padding_size=1,
                     name='conv1')  # 卷积层，卷积核数量32，大小为3*3，padding=1, 默认步长为1
        net = max_pooling2d(net, pool_size=2, strides=2, name='pool1')  # maxpooling, 减少特征图的维度一半，为112*112,因为siez=2*2,步长为2

        net = conv2d(net, filters=64, kernel_size=3, padding_size=1, name='conv2')  # 卷积层，卷积核数量为64，大小为3*3，padding=1,默认步长为1
        net = max_pooling2d(net, pool_size=2, strides=2, name='pool2')  # maxpooling，变成56*56

        net = conv2d(net, filters=128, kernel_size=3, padding_size=1, name='conv3_1')  # 卷积层，卷积核数量为128，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, filters=64, kernel_size=1, padding_size=0, name='conv3_2')  # 卷积层，卷积核数量为64，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, filters=128, kernel_size=3, padding_size=1, name='conv3_3')  # 卷积层，卷积核数量为128，大小为3*3，padding为1，默认步长为1
        net = max_pooling2d(net, pool_size=2, strides=2, name='pool3')  # maxpooling,变成28*28

        net = conv2d(net, filters=256, kernel_size=3, padding_size=1, name='conv4_1')  # 卷积层，卷积核数量为256，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, filters=128, kernel_size=1, padding_size=0, name='conv4_2')  # 卷积层，卷积核数量为128，大小为1*1，padding=0，默认步长为1
        net = conv2d(net, filters=256, kernel_size=3, padding_size=1, name='conv4_3')  # 卷积层，卷积核数量为256，大小为3*3，padding=1,默认步长为1
        net = max_pooling2d(net, pool_size=2, strides=2, name='pool4')  # maxpooling,变成14*14

        net = conv2d(net, filters=512, kernel_size=3, padding_size=1, name='conv5_1')  # 卷积层，卷积核数量为512，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, filters=256, kernel_size=1, padding_size=0, name='conv5_2')  # 卷积层，卷积核数量为256，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, filters=512, kernel_size=3, padding_size=1, name='conv5_3')  # 卷积层，卷积核数量为512，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, filters=256, kernel_size=1, padding_size=0, name='conv5_4')  # 卷积层，卷积核数量为256，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, filters=512, kernel_size=3, padding_size=1, name='conv5_5')  # 卷积层，卷积核数量为512，大小为1*1,padding=1,默认步长为1

        # 存储这一层特征图，以便后面passthrough层
        shortcut = net  # 大小为14*14
        net = max_pooling2d(net, pool_size=2, strides=2, name='pool5')  # maxpooling，变成7*7

        net = conv2d(net, filters=1024, kernel_size=3, padding_size=1, name='conv6_1')  # 卷积层，卷积核数量为1024,大小为3*3，padding=1,默认步长为1
        net = conv2d(net, filters=512, kernel_size=1, padding_size=0, name='conv6_2')  # 卷积层，卷积核数量为512，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, filters=1024, kernel_size=3, padding_size=1, name='conv6_3')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1，默认步长为1
        net = conv2d(net, filters=512, kernel_size=1, padding_size=0, name='conv6_4')  # 卷积层，卷积核数量为512，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, filters=1024, kernel_size=3, padding_size=1, name='conv6_5')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1,默认步长为1

        # 具体这个可以参考： https://blog.csdn.net/hrsstudy/article/details/70767950     Training for classification 和 Training for detection
        # 训练检测网络时去掉了分类网络的网络最后一个卷积层，在后面增加了三个卷积核尺寸为3 * 3，卷积核数量为1024的卷积层，并在这三个卷积层的最后一层后面跟一个卷积核尺寸为1 * 1
        # 的卷积层，卷积核数量是（B * （5 + C））。
        # 对于VOC数据集，卷积层输入图像尺寸为416 * 416
        # 时最终输出是13 * 13
        # 个栅格，每个栅格预测5种boxes大小，每个box包含5个坐标值和20个条件类别概率，所以输出维度是13 * 13 * 5 * （5 + 20）= 13 * 13 * 125。
        #
        # 检测网络加入了passthrough
        # layer，从最后一个输出为26 * 26 * 512
        # 的卷积层连接到新加入的三个卷积核尺寸为3 * 3
        # 的卷积层的第二层，使模型有了细粒度特征。

        # 下面这部分主要是training for detection
        net = conv2d(net, filters=1024, kernel_size=3, padding_size=1, name='conv7_1')  # 卷积层，卷积核数量为1024，大小为3*3,padding=1,默认步长为1
        net = conv2d(net, filters=1024, kernel_size=3, padding_size=1, name='conv7_2')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1,默认步长为1，大小为1024*7*7

        # 关于这部分细粒度的特征的解释，可以参考：https://blog.csdn.net/hai_xiao_tian/article/details/80472419
        # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
        # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图，可能是输入图片刚开始不是224，而是448，知道就好了,YOLOv2的输入图片大小为 416*416
        shortcut = conv2d(shortcut, filters=64, kernel_size=1, padding_size=0, name='conv_shortcut')  # 卷积层，卷积核数量为64，大小为1*1，padding=0,默认步长为1，变成26*26*64
        shortcut = reorg(shortcut, block_size=2)  # passthrough, 进行Fine-Grained Features，得到13*13*256
        # 连接之后，变成13*13*（1024+256）
        net = Concatenate(axis=-1)([shortcut, net])  # channel整合到一起，concatenated with the original features，passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上，
        net = conv2d(net, filters=1024, kernel_size=3, padding_size=1, name='conv8')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1, 在连接的特征图的基础上做卷积进行预测。变成13*13*1024

        # detection layer: 最后用一个1*1卷积去调整channel，该层没有BN层和激活函数，变成: S*S*(B*(5+C))，在这里为：13*13*425
        outputs = conv2d(net, filters=n_last_channels, kernel_size=1, batch_normalization=False, activation=None,
                        use_bias=True, name='conv_dec')

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='mse',
            optimizer='adam'
        )
        model.summary()
        return model

    def train(self, X, y, batch_size=128):
        assert self.net is not None
        net = self.net
        net.fit(X, y, batch_size=batch_size)

if __name__ == '__main__':
    images = np.random.randint(0, 255, (5, 416, 416, 3))
    labels = np.random.randint(0, 9999, (5, 13, 13, 425))
    darknet = Darknet19(images, labels)