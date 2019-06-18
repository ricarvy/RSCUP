import numpy as np
import tensorflow as tf
import keras


### YOLO V2使用的Darknet实现
class Darknet19(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.net = self.build_net()

    def build_net(self):
        pass
