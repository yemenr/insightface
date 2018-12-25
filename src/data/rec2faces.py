# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

#curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
#from builtins import range
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess
import face_image
import pdb
import numbers
import numpy as np

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--rec_path', type=str, help='the path for rec file.')
    parser.add_argument('--idx_path', type=str, help='the path for idx file.')
    parser.add_argument('--out_dir', type=str, default='the output directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.rec_path, 'r')
    anchor = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(anchor)
    if header.flag > 0:
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = range(1, int(header.label[0]))  #image index
        id2range = {}
        seq_identity = range(int(header.label[0]), int(header.label[1])) #identity sequence
        labelKey = 0
        for identity in seq_identity:
            labelKey += 1
            s = imgrec.read_idx(identity)
            header, _ = mx.recordio.unpack(s)
            a,b = int(header.label[0]), int(header.label[1])
            id2range[labelKey] = (a,b) #identity range
            count = b-a
            labelDir = os.path.join(args.out_dir, str(labelKey))
            if not os.path.exists(labelDir):
                os.mkdir(labelDir)
        print('id2range', len(id2range)) #num of identities
    else:
        imgidx = list(imgrec.keys)
        ## mkdirs

    imgCnts = []    
            
    for labelKey in id2range.keys():
        imgCnt = 0
        #pdb.set_trace()
        print('label: %s' % (str(labelKey)))
        for idx in range(id2range[labelKey][0], id2range[labelKey][1]):
            imgCnt += 1
            s = imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
            imgPath = os.path.join(args.out_dir, str(labelKey), str(imgCnt)+'.jpg')
            cv2.imwrite(imgPath, img)            
            #cv2.imshow("test", img)
            #cv2.waitKey()
        print('total images count: %d' % (imgCnt))
        imgCnts.append([imgCnt, labelKey])

    imgCnts = sorted(imgCnts)
    cntsVal = np.array(imgCnts)[:,0]
    print('avg cnt: %f' % np.mean(cntsVal))
    print('`````````````````````````````````````')
    print(imgCnts)
