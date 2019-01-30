#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: factory.py
@time: 18-6-26 上午9:38
@desc: modified version from Ross Girshick
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Factory method for easily getting imdbs by name."""

__sets = {}
import _init_paths
from datasets.wider_face import wider_face
import numpy as np

# Set up wider_face
for split in ['train', 'val']:
    name = 'wider_face_{}'.format(split)
    __sets[name] = (lambda split=split: wider_face(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
