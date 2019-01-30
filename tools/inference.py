#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: inference.py
@time: 18-7-19 下午6:01
@desc:
'''
import tensorflow as tf
from tensorflow.python.platform import gfile

from model.nms_wrapper import nms
# from lib.model.nms import py_nms as nms
import numpy as np
import cv2
import os

from utils.blob import im_list_to_blob
from model.config import cfg, get_output_dir

CLASSES = ('__background__', 'face')
input_tensor = tf.placeholder(tf.float32, shape=[1, None, None, 3])

# PATH_TO_CKPT = '/home/oeasy/PycharmProjects/tf-ssh_modify/output/mobile/wider_face_train/pb_ckpt_1.4/mobilenetv1_ssh_two_branches.pb'
PATH_TO_CKPT = 'vgg16_ssh_three_branches.pb'

def load_model():
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(PATH_TO_CKPT)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

def init_ssh_network():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    with sess.as_default():
        load_model()
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {
            output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['roi_scores', 'rois']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        im_info_tensor = tf.get_default_graph().get_tensor_by_name('im_info:0')
        return lambda img, im_info: sess.run(tensor_dict,
                                                 feed_dict={image_tensor: img, im_info_tensor: im_info})

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # print(">>>>>>>>", im_shape[0], im_shape[1])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    print(">>>>>>>>", im.shape[0], im.shape[1])
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    ##### for fast test inference time ###################
    #blobs = {}
    #blobs['data'] = im[np.newaxis, :, :, :]
    #im_scale_factors = np.array([1.0])

    return blobs, im_scale_factors

def cv2_vis(im, class_name, dets):
    """Draw detected bounding boxes using cv2."""
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # im = im[:, :, ::-1].copy()
    print('>>>>>', dets.shape)
    if dets.shape[0] != 0:
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            cv2.rectangle(im, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (0,  0, 255), 2)
    cv2.imshow("demo", im)
    # cv2.imwrite(result_file, im)
    cv2.waitKey(0)

def run_inference_for_one_image(image):
    blobs, im_scales = _get_blobs(image)
    im_blob = blobs['data']
    im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    result_dict = model_ssh_fc(im_blob, im_info)
    boxes = result_dict['rois'][:, 1:5] / im_scales[0]
    scores = np.reshape(result_dict['roi_scores'], [result_dict['roi_scores'].shape[0], -1])

    boxes = np.tile(boxes, (1, scores.shape[1]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3

    inds = np.where(scores[:, 0] > CONF_THRESH)[0]
    scores = scores[inds, 0]
    boxes = boxes[inds, :]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return dets

if __name__ == '__main__':
    # detection_graph = import_graph()
    PATH_TO_TEST_IMAGES_DIR = 'keypoint_validation_images_20170911'
    im_names = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    model_ssh_fc = init_ssh_network()

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(im_name))
        im = cv2.imread(os.path.join(PATH_TO_TEST_IMAGES_DIR, im_name))
        # im = cv2.imread(os.path.join(PATH_TO_TEST_IMAGES_DIR, '4e01c586c8cd5fa03b5d09cb0d827451781938e9.jpg'))
        print('>>>>>', im.shape[0], im.shape[1])
        boxes = run_inference_for_one_image(im)
        print('>>>>>faces:', boxes.shape[0])