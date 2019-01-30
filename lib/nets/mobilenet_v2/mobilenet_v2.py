# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Mobilenet V2.

Architecture: https://arxiv.org/abs/1801.04381

The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
3.4 M parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from nets.mobilenet_v2 import conv_blocks as ops
# from lib.nets.mobilenet import mobilenet as mobile_lib
import nets.mobilenet_v2.mobilenet as mobile_lib

from nets.network import Network
from model.config import cfg

slim = tf.contrib.slim
op = mobile_lib.op

expand_input = ops.expand_input_by_factor

# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        #  use stride 1 for the 15th layer
        op(ops.expanded_conv, stride=2, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=320),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
)


# pyformat: enable

### Modified mobilenet_v2
@slim.add_arg_scope
def mobilenet(input_tensor,
              num_classes=1001,
              depth_multiplier=1.0,
              scope='MobilenetV2',
              conv_defs=None,
              finegrain_classification_mode=False,
              min_depth=None,
              divisible_by=None,
              **kwargs):
    """Creates mobilenet V2 network.

    Inference mode is created by default. To create training use training_scope
    below.

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

    Args:
      input_tensor: The input tensor
      num_classes: number of classes
      depth_multiplier: The multiplier applied to scale number of
      channels in each layer. Note: this is called depth multiplier in the
      paper but the name is kept for consistency with slim's model builder.
      scope: Scope of the operator
      conv_defs: Allows to override default conv def.
      finegrain_classification_mode: When set to True, the model
      will keep the last layer large even for small multipliers. Following
      https://arxiv.org/abs/1801.04381
      suggests that it improves performance for ImageNet-type of problems.
        *Note* ignored if final_endpoint makes the builder exit earlier.
      min_depth: If provided, will ensure that all layers will have that
      many channels after application of depth multiplier.
      divisible_by: If provided will ensure that all layers # channels
      will be divisible by this number.
      **kwargs: passed directly to mobilenet.mobilenet:
        prediction_fn- what prediction function to use.
        reuse-: whether to reuse variables (if reuse set to true, scope
        must be given).
    Returns:
      logits/endpoints pair

    Raises:
      ValueError: On invalid arguments
    """
    if conv_defs is None:
        conv_defs = V2_DEF
    if 'multiplier' in kwargs:
        raise ValueError('mobilenetv2 doesn\'t support generic '
                         'multiplier parameter use "depth_multiplier" instead.')
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        if depth_multiplier < 1:
            conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier

    depth_args = {}
    # NB: do not set depth_args unless they are provided to avoid overriding
    # whatever default depth_multiplier might have thanks to arg_scope.
    if min_depth is not None:
        depth_args['min_depth'] = min_depth
    if divisible_by is not None:
        depth_args['divisible_by'] = divisible_by

    with slim.arg_scope((mobile_lib.depth_multiplier,), **depth_args):
        return mobile_lib.mobilenet(
            input_tensor,
            num_classes=num_classes,
            conv_defs=conv_defs,
            scope=scope,
            multiplier=depth_multiplier,
            **kwargs)


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
    """Creates base of the mobilenet (no pooling and no logits) ."""
    return mobilenet(input_tensor,
                     depth_multiplier=depth_multiplier,
                     base_only=True, **kwargs)


def training_scope(**kwargs):
    """Defines MobilenetV2 training scope.

    Usage:
       with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
         logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

    with slim.

    Args:
      **kwargs: Passed to mobilenet.training_scope. The following parameters
      are supported:
        weight_decay- The weight decay to use for regularizing the model.
        stddev-  Standard deviation for initialization, if negative uses xavier.
        dropout_keep_prob- dropout keep probability
        bn_decay- decay for the batch norm moving averages.

    Returns:
      An `arg_scope` to use for the mobilenet v2 model.
    """
    return mobile_lib.training_scope(**kwargs)


__all__ = ['training_scope', 'mobilenet_base', 'mobilenet', 'V2_DEF']


class mobilenetv2_ssh(Network):
    def __init__(self):
        Network.__init__(self)
        # config which branch contained in the SSH  should be the format of ['M1', 'M2', 'M3']
        self._feat_branches = ['M1', 'M2', 'M3']
        self._feat_stride = {'M1': 8, 'M2': 16, 'M3': 32}
        self._feat_layers = {'M1': ['layer_5', 'layer_14'], 'M2': 'layer_14',
                             'M3': 'layer_19'}
        # self._feat_layers = {'M1': ['layer_5/expansion_output', 'layer_19'], 'M2': 'layer_19',
        #                      'M3': 'layer_19'}
        self._Module_boxes = {'M1': 128, 'M2': 256, 'M3': 256}
        self.end_points = {}
        self._depth_multiplier = cfg.MOBILENET_V2.DEPTH_MULTIPLIER
        self._min_depth = cfg.MOBILENET_V2.MIN_DEPTH
        self._scope = 'MobilenetV2'

    def _image_to_head(self, is_training, reuse=None):
        net_conv = self._image
        with slim.arg_scope(training_scope(is_training=is_training, bn_decay=0.9997)):
            net_conv, end_points = mobilenet_base(net_conv, conv_defs=V2_DEF, depth_multiplier=self._depth_multiplier,
                                             min_depth=self._min_depth)

        self.end_points['layer_5'] = end_points['layer_5']
        self.end_points['layer_14'] = end_points['layer_14']
        self.end_points['layer_19'] = end_points['layer_19']

        self._act_summaries.append(net_conv)
        self._layers['head'] = net_conv

        return net_conv

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/Conv/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix MobileNet V2 layers..')
        with tf.variable_scope('Fix_MobileNet_V2') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR, and match the scale by (255.0 / 2.0)
                Conv_rgb = tf.get_variable("Conv_rgb",
                                           [3, 3, 3, max(int(32 * self._depth_multiplier), 8)],
                                           trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/Conv/weights": Conv_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + "/Conv/weights:0"],
                                   tf.reverse(Conv_rgb / (255.0 / 2.0), [2])))
