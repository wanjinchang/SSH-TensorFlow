#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: darknet53.py
@time: 18-7-4 下午4:19
@desc:
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
import numpy as np
from collections import namedtuple

from nets.network import Network
from model.config import cfg
conv_bn = ['beta', 'gamma', 'mean', 'variance', 'weights']

def darknet53_arg_scope(is_training=True,
                           stddev=0.09):
    batch_norm_params = {
        'is_training': False,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
        'trainable': False,
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(cfg.DARKNET.WEIGHT_DECAY)

    with slim.arg_scope([slim.conv2d],
                        trainable=is_training,
                        weights_initializer=weights_init,
                        activation_fn=leaky_relu(alpha=0.1),
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer) as sc:
                with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                    return sc

def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op

# def leaky_relu(alpha):
#     def op(inputs):
#         return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
#     return op

class Darknet53_ssh(Network):
    def __init__(self, darknet53_npz_path=None):
        Network.__init__(self)
        self._feat_branches = {'M1', 'M2', 'M3'}
        self._feat_stride = {'M1': 8, 'M2': 16, 'M3': 32}
        self._feat_layers = {'M1': ['res10', 'res18'], 'M2': 'res18',
                             'M3': 'res22'}
        self._Module_boxes = {'M1': 128, 'M2': 256, 'M3': 256}
        self.end_points = {}
        self._scope = 'Darknet53'
        self.darknet53_npz_path = darknet53_npz_path

    def _image_to_head(self, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            with slim.arg_scope(darknet53_arg_scope(is_training)):
                self.conv0 = slim.conv2d(self._image, 32, [3, 3], scope='conv_0')
                # Downsample
                self.conv1 = slim.conv2d(self.conv0, 64, [3, 3], stride=2, scope='conv_1')

                # 1 x residual_blocks
                self.conv2 = slim.conv2d(self.conv1, 32, [1, 1], scope='conv_2')
                self.conv3 = slim.conv2d(self.conv2, 64, [3, 3], scope='conv_3')
                self.res0 = self.conv3 + self.conv1

                # Downsample
                self.conv4 = slim.conv2d(self.res0, 128, [3, 3], stride=2, scope='conv_4')

                # 2 x residual_blocks
                self.conv5 = slim.conv2d(self.conv4, 64, [1, 1], scope='conv_5')
                self.conv6 = slim.conv2d(self.conv5, 128, [3, 3], scope='conv_6')
                self.res1 = self.conv6 + self.conv4
                self.conv7 = slim.conv2d(self.res1, 64, [1, 1], scope='conv_7')
                self.conv8 = slim.conv2d(self.conv7, 128, [3, 3], scope='conv_8')
                self.res2 = self.conv8 + self.res1  # 128

                # Downsample
                self.conv9 = slim.conv2d(self.res2, 256, [3, 3], stride=2, scope='conv_9')

                # 8 x residual_blocks
                self.conv10 = slim.conv2d(self.conv9, 128, [1, 1], scope='conv_10')
                self.conv11 = slim.conv2d(self.conv10, 256, [3, 3], scope='conv_11')
                self.res3 = self.conv11 + self.conv9
                self.conv12 = slim.conv2d(self.res3, 128, [1, 1], scope='conv_12')
                self.conv13 = slim.conv2d(self.conv12, 256, [3, 3], scope='conv_13')
                self.res4 = self.conv13 + self.res3
                self.conv14 = slim.conv2d(self.res4, 128, [1, 1], scope='conv_14')
                self.conv15 = slim.conv2d(self.conv14, 256, [3, 3], stride=1, scope='conv_15')
                self.res5 = self.conv15 + self.res4
                self.conv16 = slim.conv2d(self.res5, 128, [1, 1], scope='conv_16')
                self.conv17 = slim.conv2d(self.conv16, 256, [3, 3], scope='conv_17')
                self.res6 = self.conv17 + self.res5
                self.conv18 = slim.conv2d(self.res6, 128, [1, 1], scope='conv_18')
                self.conv19 = slim.conv2d(self.conv18, 256, [3, 3], scope='conv_19')
                self.res7 = self.conv19 + self.res6
                self.conv20 = slim.conv2d(self.res7, 128, [1, 1], scope='conv_20')
                self.conv21 = slim.conv2d(self.conv20, 256, [3, 3], scope='conv_21')
                self.res8 = self.conv21 + self.res7
                self.conv22 = slim.conv2d(self.res8, 128, [1, 1], scope='conv_22')
                self.conv23 = slim.conv2d(self.conv22, 256, [3, 3], scope='conv_23')
                self.res9 = self.conv23 + self.res8
                self.conv24 = slim.conv2d(self.res9, 128, [1, 1], scope='conv_24')
                self.conv25 = slim.conv2d(self.conv24, 256, [3, 3], scope='conv_25')
                self.res10 = self.conv25 + self.res9
                self.end_points['res10'] = self.res10

                # Downsample
                self.conv26 = slim.conv2d(self.res10, 512, [3, 3], stride=2, scope='conv_26')

                # 8 x residual_blocks
                self.conv27 = slim.conv2d(self.conv26, 256, [1, 1], scope='conv_27')
                self.conv28 = slim.conv2d(self.conv27, 512, [3, 3], scope='conv_28')
                self.res11 = self.conv28 + self.conv26
                self.conv29 = slim.conv2d(self.res11, 256, [1, 1], scope='conv_29')
                self.conv30 = slim.conv2d(self.conv29, 512, [3, 3], scope='conv_30')
                self.res12 = self.conv30 + self.res11
                self.conv31 = slim.conv2d(self.res12, 256, [1, 1], scope='conv_31')
                self.conv32 = slim.conv2d(self.conv31, 512, [3, 3], scope='conv_32')
                self.res13 = self.conv32 + self.res12
                self.conv33 = slim.conv2d(self.res13, 256, [1, 1], scope='conv_33')
                self.conv34 = slim.conv2d(self.conv33, 512, [3, 3], scope='conv_34')
                self.res14 = self.conv34 + self.res13
                self.conv35 = slim.conv2d(self.res14, 256, [1, 1], scope='conv_35')
                self.conv36 = slim.conv2d(self.conv35, 512, [3, 3], scope='conv_36')
                self.res15 = self.conv36 + self.res14
                self.conv37 = slim.conv2d(self.res15, 256, [1, 1], scope='conv_37')
                self.conv38 = slim.conv2d(self.conv37, 512, [3, 3], scope='conv_38')
                self.res16 = self.conv38 + self.res15
                self.conv39 = slim.conv2d(self.res16, 256, [1, 1], scope='conv_39')
                self.conv40 = slim.conv2d(self.conv39, 512, [3, 3], scope='conv_40')
                self.res17 = self.conv40 + self.res16
                self.conv41 = slim.conv2d(self.res17, 256, [1, 1], scope='conv_41')
                self.conv42 = slim.conv2d(self.conv41, 512, [3, 3], scope='conv_42')
                self.res18 = self.conv42 + self.res17
                self.end_points['res18'] = self.res18

                # Downsample
                self.conv43 = slim.conv2d(self.res18, 1024, [3, 3], stride=2, scope='conv_43')

                # 4 x residual_blocks
                self.conv44 = slim.conv2d(self.conv43, 512, [1, 1], scope='conv_44')
                self.conv45 = slim.conv2d(self.conv44, 1024, [3, 3], scope='conv_45')
                self.res19 = self.conv45 + self.conv43
                self.conv46 = slim.conv2d(self.res19, 512, [1, 1], scope='conv_46')
                self.conv47 = slim.conv2d(self.conv44, 1024, [3, 3], scope='conv_47')
                self.res20 = self.conv47 + self.res19
                self.conv48 = slim.conv2d(self.res20, 512, [1, 1], scope='conv_48')
                self.conv49 = slim.conv2d(self.conv48, 1024, [3, 3], scope='conv_49')
                self.res21 = self.conv49 + self.res20
                self.conv50 = slim.conv2d(self.res21, 512, [1, 1], scope='conv_50')
                self.conv51 = slim.conv2d(self.conv50, 1024, [3, 3], scope='conv_51')
                self.res22 = self.conv51 + self.res21
                self.end_points['res22'] = self.res22
            self._act_summaries.append(self.res22)
            self._layers['head'] = self.res22

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv_0/conv_weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def restored_from_npz(self, sess):
        if self.darknet53_npz_path is not None:
            data_dict = np.load(self.darknet53_npz_path)
            data_dict = self.fix_first_conv(data_dict)
        else:
            print('the darknet53_path does exist!!!')
            return
        conv_idx = conv_bn.index('weights')
        beta_idx = conv_bn.index('beta')
        gamma_idx = conv_bn.index('gamma')
        mean_idx = conv_bn.index('mean')
        variance_idx = conv_bn.index('variance')
        print('restored variables from npz!!!!')

        with tf.variable_scope(self._scope, reuse=True):
            for key, value in data_dict.items():
                sess.run(tf.get_variable(key + '/weights').assign(value[conv_idx][0]))
                # print(self._scope + '/' + key + '/weights' + ' Restored...')
                sess.run(tf.get_variable(key + '/BatchNorm/gamma').assign(value[gamma_idx]))
                # print(self._scope + '/' + key + 'BatchNorm/gamma' + ' Restored...')
                sess.run(tf.get_variable(key + '/BatchNorm/beta').assign(value[beta_idx]))
                # print(self._scope + '/' + key + '/BatchNorm/beta' + ' Restored...')
                sess.run(tf.get_variable(key + '/BatchNorm/moving_mean').assign(value[mean_idx]))
                # print(self._scope + '/' + key + '/BatchNorm/moving_mean' + ' Restored...')
                sess.run(tf.get_variable(key + '/BatchNorm/moving_variance').assign(value[variance_idx]))

    def fix_first_conv(self, data_dict):
        # fix the first conv layer channel from RGB to BGR
        print('Fix Darknet53 first conv layers..')
        conv_idx = conv_bn.index('weights')
        data_dict['conv_0'][conv_idx][0] = data_dict['conv_0'][conv_idx][0][:, :, ::-1, :]
        return data_dict

if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, 512, 512, 3])
    darknet53 = Darknet53()
