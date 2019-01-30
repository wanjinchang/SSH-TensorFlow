#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: RunOnFDDB.py
@time: 18-7-2 上午9:38
@desc:
'''
import sys
sys.path.append("..")
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from utils.timer import Timer

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2

from nets.vgg16 import vgg16_ssh
from nets.resnet_v1 import resnetv1_ssh
from nets.mobilenet_v1 import mobilenetv1_ssh
from nets.darknet53 import Darknet53_ssh
from nets.mobilenet_v2.mobilenet_v2 import mobilenetv2_ssh

CLASSES = ('__background__', 'face')

NETS = {'vgg16': ('vgg16_ssh_iter_300000.ckpt',), 'res101': ('res101_ssh_iter_110000.ckpt',)}

DATASETS= {'wider_face': ('wider_face_train',)}

data_dir = '/home/oeasy/Downloads/dataset/face_data/FDDB'
out_dir = '/home/oeasy/Downloads/dataset/face_data/FDDB/ssh_result'

def get_imdb_fddb(data_dir):
    imdb = []
    nfold = 10
    for n in range(nfold):
        file_name = 'FDDB-folds/FDDB-fold-%02d.txt' % (n + 1)
        file_name = os.path.join(data_dir, file_name)
        fid = open(file_name, 'r')
        image_names = []
        for im_name in fid.readlines():
            image_names.append(im_name.strip('\n'))
        imdb.append(image_names)
    return imdb

def run_on_fddb(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    inds = np.where(scores[:, 0] > CONF_THRESH)[0]
    scores = scores[inds, 0]
    boxes = boxes[inds, :]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return dets

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow SSH demo')
    parser.add_argument('--backbone', dest='backbone', help='Network to use [vgg16 res101]',
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
    tfmodel = os.path.join('/home/oeasy/PycharmProjects/tf-ssh_modify/output', backbone, DATASETS[dataset][0], 'default/20180628_hardmining',
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
    if args.backbone == 'vgg16':
        net = vgg16_ssh()
    elif args.backbone == 'res50':
        net = resnetv1_ssh(num_layers=50)
    elif args.backbone == 'res101':
        net = resnetv1_ssh(num_layers=101)
    elif args.backbone == 'res152':
        net = resnetv1_ssh(num_layers=152)
    elif args.backbone == 'darknet53':
        net = Darknet53_ssh('data/imagenet_weights/darknet53.conv.74.npz')
    elif args.backbone == 'mobile':
        net = mobilenetv1_ssh()
    elif args.backbone == 'mobile_v2':
        net = mobilenetv2_ssh()
    else:
        raise NotImplementedError

    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales={"M1": [1, 2], "M2": [4, 8], "M3": [16, 32]})
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    imdb = get_imdb_fddb(data_dir)
    nfold = len(imdb)
    for i in range(nfold):
        image_names = imdb[i]
        print(image_names)
        dets_file_name = os.path.join(out_dir, 'FDDB-det-fold-%02d.txt' % (i + 1))
        fid = open(dets_file_name, 'w')
        sys.stdout.write('%s ' % (i + 1))
        image_names_abs = [os.path.join(data_dir, 'originalPics', image_name + '.jpg') for image_name in image_names]
        for idx, im_name in enumerate(image_names):
            img_path = os.path.join(data_dir, 'originalPics', im_name + '.jpg')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Precessing for {}'.format(img_path))
            all_boxes = run_on_fddb(sess, net, img_path)
            if all_boxes.shape[0] == 0:
                fid.write(im_name + '\n')
                fid.write(str(1) + '\n')
                fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                continue
            print("boxes_row_clo:", all_boxes.shape[0], all_boxes.shape[1])
            fid.write(im_name + '\n')
            fid.write(str(all_boxes.shape[0]) + '\n')

            for box in all_boxes:
                fid.write('%f %f %f %f %f\n' % (float(box[0]), float(box[1]), float(box[2]-box[0]+1), float(box[3]-box[1]+1), box[4]))

        fid.close()


