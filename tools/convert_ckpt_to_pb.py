#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: convert_ckpt_to_pb.py
@time: 18-7-19 上午9:31
@desc:
'''
import logging
import os
import tempfile
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib

from nets.vgg16 import vgg16_ssh
from nets.resnet_v1 import resnetv1_ssh
from nets.mobilenet_v1 import mobilenetv1_ssh
from nets.darknet53 import Darknet53_ssh
from nets.mobilenet_v2.mobilenet_v2 import mobilenetv2_ssh

import tensorflow as tf

slim = tf.contrib.slim


def freeze_graph_with_def_protos(
        input_graph_def,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        clear_devices,
        initializer_nodes,
        variable_names_blacklist=''):
    """Converts all variables in a graph and checkpoint into constants."""
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError(
            'Input checkpoint "' + input_checkpoint + '" does not exist!')

    if not output_node_names:
        raise ValueError(
            'You must supply the name of a node to --output_node_names.')

    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ''

    # print('>>>>>input_graph_def.node', input_graph_def.node)
    with tf.Graph().as_default():
        tf.import_graph_def(input_graph_def, name='')
        config = tf.ConfigProto(graph_options=tf.GraphOptions())
        with session.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            var_list = {}
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()
            print('>>>>>>var_to_shape_map', var_to_shape_map)
            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ':0')
                except KeyError:
                    # This tensor doesn't exist in the graph (for example it's
                    # 'global_step' or a similar housekeeping element) so skip it.
                    continue
                var_list[key] = tensor
            saver = saver_lib.Saver(var_list=var_list)
            saver.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes)

            variable_names_blacklist = (variable_names_blacklist.split(',') if
                                        variable_names_blacklist else None)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(','),
                variable_names_blacklist=variable_names_blacklist)
            # if input_saver_def:
            #     print('>>>>>input_saver_def', input_saver_def)
            #     saver = saver_lib.Saver(saver_def=input_saver_def)
            #     saver.restore(sess, input_checkpoint)
            # else:
            #     var_list = {}
            #     reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            #     var_to_shape_map = reader.get_variable_to_shape_map()
            #     for key in var_to_shape_map:
            #         try:
            #             tensor = sess.graph.get_tensor_by_name(key + ':0')
            #         except KeyError:
            #             # This tensor doesn't exist in the graph (for example it's
            #             # 'global_step' or a similar housekeeping element) so skip it.
            #             continue
            #         var_list[key] = tensor
            #     saver = saver_lib.Saver(var_list=var_list)
            #     saver.restore(sess, input_checkpoint)
            #     if initializer_nodes:
            #         sess.run(initializer_nodes)
            #
            # variable_names_blacklist = (variable_names_blacklist.split(',') if
            #                             variable_names_blacklist else None)
            # output_graph_def = graph_util.convert_variables_to_constants(
            #     sess,
            #     input_graph_def,
            #     output_node_names.split(','),
            #     variable_names_blacklist=variable_names_blacklist)

    return output_graph_def


def replace_variable_values_with_moving_averages(graph,
                                                 current_checkpoint_file,
                                                 new_checkpoint_file):
    """Replaces variable values in the checkpoint with their moving averages.
    If the current checkpoint has shadow variables maintaining moving averages of
    the variables defined in the graph, this function generates a new checkpoint
    where the variables contain the values of their moving averages.
    Args:
      graph: a tf.Graph object.
      current_checkpoint_file: a checkpoint containing both original variables and
        their moving averages.
      new_checkpoint_file: file path to write a new checkpoint.
    """
    with graph.as_default():
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        ema_variables_to_restore = variable_averages.variables_to_restore()
        with tf.Session() as sess:
            read_saver = tf.train.Saver(ema_variables_to_restore)
            read_saver.restore(sess, current_checkpoint_file)
            write_saver = tf.train.Saver()
            write_saver.save(sess, new_checkpoint_file)

def _image_tensor_input_placeholder(input_shape=None):
    """Returns input placeholder and a 4-D uint8 image tensor."""
    if input_shape is None:
        input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(
        dtype=tf.uint8, shape=input_shape, name='image_tensor')
    return input_tensor, input_tensor

input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
}

# def _add_output_tensor_nodes(preprocess_scores, preprocess_rois, output_collection_name='inferece_op'):
#     """
#     Adds output nodes for detection boxes and scores.
#     :param preprocess_scores: a dictionary containing the all detection scores concat by
#         the network inference output;
#     :param preprocess_rois: a dictionary containing the all detection rois concat by
#         the network inference output;
#     :param output_collection_name: Name of collection to add output tensors to.
#     :return: A tensor dict containing the added output tensor nodes.
#     """
#     outputs = {}
#     outputs['all_scores'] = tf.identity(preprocess_scores, name='all_scores')
#     outputs['all_rois'] = tf.identity(preprocess_rois, name='all_rois')
#     for output_key in outputs.keys():
#         tf.add_to_collection(output_collection_name, outputs[output_key])
#     return outputs
def _add_output_tensor_nodes(net, preprocess_tensors, output_collection_name='inferece_op'):
    """
    Adds output nodes for all preprocess_tensors.
    :param preprocess_tensors: a dictionary containing the all predictions;
    :param output_collection_name: Name of collection to add output tensors to.
    :return: A tensor dict containing the added output tensor nodes.
    """
    outputs = {}
    outputs['roi_scores'] = tf.identity(net.all_rois_scores, name='rois_scores')
    outputs['rois']  = tf.identity(net.all_rois, name='rois')
    for output_key in outputs.keys():
        tf.add_to_collection(output_collection_name, outputs[output_key])
    return outputs

def write_frozen_graph(frozen_graph_path, frozen_graph_def):
    """Writes frozen graph to disk.
    Args:
      frozen_graph_path: Path to write inference graph.
      frozen_graph_def: tf.GraphDef holding frozen graph.
    """
    with gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
    logging.info('%d ops in the final graph.', len(frozen_graph_def.node))

def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):
    """Writes SavedModel to disk.
    If checkpoint_path is not None bakes the weights into the graph thereby
    eliminating the need of checkpoint files during inference. If the model
    was trained with moving averages, setting use_moving_averages to true
    restores the moving averages, otherwise the original set of variables
    is restored.
    Args:
      saved_model_path: Path to write SavedModel.
      frozen_graph_def: tf.GraphDef holding frozen graph.
      inputs: The input placeholder tensor.
      outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    with tf.Graph().as_default():
        with session.Session() as sess:
            tf.import_graph_def(frozen_graph_def, name='')

            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        detection_signature,
                },
            )
            builder.save()

def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
    """Writes the graph and the checkpoint into disk."""
    for node in inference_graph_def.node:
        node.device = ''
    with tf.Graph().as_default():
        tf.import_graph_def(inference_graph_def, name='')
        with session.Session() as sess:
            saver = saver_lib.Saver(saver_def=input_saver_def,
                                    save_relative_paths=True)
            saver.restore(sess, trained_checkpoint_prefix)
            saver.save(sess, model_path)


def _get_outputs_from_inputs(input_tensors, detection_model,
                             output_collection_name):
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(
        preprocessed_inputs, true_image_shapes)
    postprocessed_tensors = detection_model.postprocess(
        output_tensors, true_image_shapes)
    return _add_output_tensor_nodes(postprocessed_tensors,
                                    output_collection_name)


# def _build_detection_graph(input_type, detection_model, input_shape,
#                            output_collection_name, graph_hook_fn):
#     """Build the detection graph."""
#     if input_type not in input_placeholder_fn_map:
#         raise ValueError('Unknown input type: {}'.format(input_type))
#     placeholder_args = {}
#     if input_shape is not None:
#         if input_type != 'image_tensor':
#             raise ValueError('Can only specify input shape for `image_tensor` '
#                              'inputs.')
#         placeholder_args['input_shape'] = input_shape
#     placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](
#         **placeholder_args)
#     outputs = _get_outputs_from_inputs(
#         input_tensors=input_tensors,
#         detection_model=detection_model,
#         output_collection_name=output_collection_name)
#
#     # Add global step to the graph.
#     slim.get_or_create_global_step()
#
#     if graph_hook_fn: graph_hook_fn()
#
#     return outputs, placeholder_tensor

def _build_detection_graph(output_collection_name, graph_hook_fn):
    """Build the detection graph."""
    net = mobilenetv1_ssh()
    # net = vgg16_ssh()
    net.create_architecture("TEST", 2,
                            tag='default', anchor_scales={"M1": [1, 2], "M2": [4, 8], "M3": [16, 32]})
    placeholder_tensor = net._image
    outputs = net._predictions

    outputs = _add_output_tensor_nodes(net, outputs, output_collection_name)

    # Add global step to the graph.
    # slim.get_or_create_global_step()

    if graph_hook_fn: graph_hook_fn()

    return outputs, placeholder_tensor

def _export_inference_graph(trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            output_collection_name='inference_op',
                            graph_hook_fn=None,
                            write_inference_graph=False):
    """Export helper."""
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                     'mobilev1_ssh_group_training.pb')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'mobile_ssh_iter_320000.ckpt')

    outputs, placeholder_tensor = _build_detection_graph(
        output_collection_name=output_collection_name,
        graph_hook_fn=graph_hook_fn)

    print('>>>>>>outputs', outputs)

    saver_kwargs = {}

    checkpoint_to_use = trained_checkpoint_prefix

    saver = tf.train.Saver(**saver_kwargs)
    input_saver_def = saver.as_saver_def()

    # tf.import_graph_def(tf.get_default_graph().as_graph_def(), name='')

    # write_graph_and_checkpoint(
    #     inference_graph_def=tf.get_default_graph().as_graph_def(),
    #     model_path=model_path,
    #     input_saver_def=input_saver_def,
    #     trained_checkpoint_prefix=checkpoint_to_use)

    if write_inference_graph:
        inference_graph_def = tf.get_default_graph().as_graph_def()
        inference_graph_path = os.path.join(output_directory,
                                            'mobilev1_ssh_group_training.pbtxt')
        for node in inference_graph_def.node:
            node.device = ''
        with gfile.GFile(inference_graph_path, 'wb') as f:
            f.write(str(inference_graph_def))

    if additional_output_tensor_names is not None:
        output_node_names = ','.join(outputs.keys() + additional_output_tensor_names)
    else:
        output_node_names = ','.join(outputs.keys())

    frozen_graph_def = freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_checkpoint=checkpoint_to_use,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        clear_devices=True,
        initializer_nodes='')
    write_frozen_graph(frozen_graph_path, frozen_graph_def)
    write_saved_model(saved_model_path, frozen_graph_def,
                      placeholder_tensor, outputs)


def export_inference_graph(trained_checkpoint_prefix,
                           output_directory,
                           output_collection_name='mobilev1_ssh_group_training',
                           additional_output_tensor_names=None,
                           write_inference_graph=False):
    """Exports inference graph for the model specified in the pipeline config.
    Args:
      input_type: Type of input for the graph. Can be one of ['image_tensor',
        'encoded_image_string_tensor', 'tf_example'].
      pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
      trained_checkpoint_prefix: Path to the trained checkpoint file.
      output_directory: Path to write outputs.
      input_shape: Sets a fixed shape for an `image_tensor` input. If not
        specified, will default to [None, None, None, 3].
      output_collection_name: Name of collection to add output tensors to.
        If None, does not add output tensors to a collection.
      additional_output_tensor_names: list of additional output
        tensors to include in the frozen graph.
      write_inference_graph: If true, writes inference graph to disk.
    """
    _export_inference_graph(
        trained_checkpoint_prefix,
        output_directory,
        additional_output_tensor_names,
        output_collection_name,
        graph_hook_fn=None,
        write_inference_graph=write_inference_graph)

if __name__ == '__main__':
    # ckpt_path = '/home/oeasy/PycharmProjects/SSH-TensorFlow/output/mobile/wider_face_train/default/mobile_ssh_iter_400000.ckpt'
    ckpt_path = '/home/oeasy/PycharmProjects/tf-ssh_modify/output/pb/mobile/default_group_training/mobile_ssh_iter_320000.ckpt'
    output_dir = '/home/oeasy/PycharmProjects/tf-ssh_modify/output/pb/mobile/default_group_training'
    export_inference_graph(trained_checkpoint_prefix=ckpt_path, output_directory=output_dir)

