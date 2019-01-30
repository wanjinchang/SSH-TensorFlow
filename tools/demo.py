#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tools._init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16_ssh
from nets.resnet_v1 import resnetv1_ssh
from nets.mobilenet_v1 import mobilenetv1_ssh
from nets.darknet53 import Darknet53_ssh
from nets.mobilenet_v2.mobilenet_v2 import mobilenetv2_ssh

CLASSES = ('__background__', 'face')

NETS = {'vgg16': ('vgg16_ssh_iter_300000.ckpt',), 'res101': ('res101_ssh_iter_110000.ckpt',),
        'mobile': ('mobile_ssh_iter_400000.ckpt',), 'res50': ('res50_ssh_iter_305000.ckpt',)}
DATASETS= {'wider_face': ('wider_face_train',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='green', linewidth=2)
            )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def video_demo(sess, net, image):
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, image)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.85
    NMS_THRESH = 0.3

    inds = np.where(scores[:, 0] > CONF_THRESH)[0]
    scores = scores[inds, 0]
    boxes = boxes[inds, :]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return dets
    # vis_detections(image, CLASSES[1], dets, thresh=CONF_THRESH)

def cv2_vis(im, class_name, dets):
    """Draw detected bounding boxes using cv2."""
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # im = im[:, :, ::-1].copy()
    if dets.shape[0] != 0:
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            cv2.rectangle(im, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (0,  0, 255), 2)
    cv2.imshow("demo", im)
    # cv2.imwrite(result_file, im)
    cv2.waitKey(0)

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    once_time = 0


    im = cv2.imread(img_path)
    # print('>>>>>>>', im.shape[0], im.shape[1])

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    once_time = timer.total_time
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.85
    NMS_THRESH = 0.3


    inds = np.where(scores[:, 0] > CONF_THRESH)[0]
    scores = scores[inds, 0]
    boxes = boxes[inds, :]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    print('>>>>>num_faces:', dets.shape[0])
    cv2_vis(im, CLASSES[1], dets)
    return once_time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow SSH demo')
    parser.add_argument('--backbone', dest='backbone', help='Backbone network to use [vgg16 res101 res50 mobile mobile_v2]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [wider_face]',
                        choices=DATASETS.keys(), default='wider_face')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    backbone = args.backbone
    dataset = args.dataset
    tfmodel = os.path.join('/home/oeasy/PycharmProjects/SSH-TensorFlow/output', backbone, DATASETS[dataset][0], 'default',
                              NETS[backbone][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if backbone == 'vgg16':
        print('ssh backbone is vgg16')
        net = vgg16_ssh()
    elif backbone == 'res101':
        print('ssh backbone is resnet101')
        net = resnetv1_ssh(num_layers=101)
    elif backbone == 'res50':
        print('ssh backbone is resnet50')
        net = resnetv1_ssh(num_layers=50)
    elif backbone == 'mobile':
        print('ssh backbone is mobilenetnet_v1')
        net = mobilenetv1_ssh()
    elif args.backbone == 'mobile_v2':
        print('ssh backbone is mobilenetnet_v2')
        net = mobilenetv2_ssh()
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales={"M1": [1, 2], "M2": [4, 8], "M3": [16, 32]})
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    images_dir = os.path.join(cfg.DATA_DIR, 'demo')
    im_names = os.listdir(images_dir)
    print('>>>>', im_names)
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        img_path = os.path.join(images_dir, im_name)
        demo(sess, net, img_path)

    ################################################ video test demo #################################
    # videopath = "./video_test.avi"
    # video_capture = cv2.VideoCapture(videopath)
    # video_capture.set(3, 340)
    # video_capture.set(4, 480)
    # while True:
    #     # fps = video_capture.get(cv2.CAP_PROP_FPS)
    #     t1 = cv2.getTickCount()
    #     ret, frame = video_capture.read()
    #     # h, w, _ = frame.shape
    #     # print("video height: %s & width: %s" % (h, w))   # video height: 240 & width: 320
    #     if ret:
    #         image = np.array(frame)
    #         detetctions = video_demo(sess, net, image)
    #         t2 = cv2.getTickCount()
    #         t = (t2 - t1) / cv2.getTickFrequency()
    #         fps = 1.0 / t
    #         for i in range(detetctions.shape[0]):
    #             bbox = detetctions[i, :4]
    #             score = detetctions[i, 4]
    #             corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    #             # if score > thresh:
    #             cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
    #                           (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
    #             cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.5,
    #                         (0, 0, 255), 2)
    #         cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                     (255, 0, 255), 2)
    #         cv2.imshow("", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         print('device not find')
    #         break



