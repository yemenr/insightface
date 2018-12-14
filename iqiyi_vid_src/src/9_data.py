import argparse
import cv2
import os
import sys
import datetime
import time
import pickle
import mxnet as mx
import random
import numpy as np
import sklearn


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
#parser.add_argument('--model', default='./models2/model-r100-sfza/model,0', help='path to load model.') #0.875
#parser.add_argument('--model', default='../iqiyi-competition/models2/model-r100-iqiyi/model,6', help='path to load model.') #0.875
parser.add_argument('--output', default='/data/sdb/iqiyi/scene_rec', help='')
parser.add_argument('--dataset', default='/gpu/data1/jiaguo/iqiyi', help='')
parser.add_argument('--sampling', default=20, type=int, help='')
parser.add_argument('--split', default='', type=str, help='')
args = parser.parse_args()


PARTS = [1]
MAX_LABEL = 574

PARTS = [1, 2, 3]
MAX_LABEL = 99999

print(args, MAX_LABEL, PARTS)

output_dir = args.output
writer = mx.recordio.MXIndexedRecordIO(os.path.join(output_dir, 'train.idx'), os.path.join(output_dir, 'train.rec'), 'w')
writer_val = mx.recordio.MXIndexedRecordIO(os.path.join(output_dir, 'val.idx'), os.path.join(output_dir, 'val.rec'), 'w')

SPLIT = [0, 1]
if len(args.split)>0:
  _v = args.split.split(',')
  SPLIT[0] = int(_v[0])
  SPLIT[1] = int(_v[1])

print('SPLIT:', SPLIT)

rec_idx = 0
rec_val_idx = 0

def get_faces(video, label):
  global rec_idx
  global rec_val_idx
  count = 0
  sampling = args.sampling
  while True:
    cap = cv2.VideoCapture(video)
    frame_num = 0
    while cap.isOpened(): 
      ret,frame = cap.read() 
      if frame is None:
        break
      frame_num+=1
      if frame_num%sampling!=0:
        continue
      frame = cv2.resize(frame, (473, 256))
      im = frame
      count+=1
      if random.random()<0.95:
        nheader = mx.recordio.IRHeader(0, label, rec_idx, 0)
        s = mx.recordio.pack_img(nheader, im, quality=95, img_fmt='.jpg')
        writer.write_idx(rec_idx, s)
        rec_idx+=1
      else:
        nheader = mx.recordio.IRHeader(0, label, rec_val_idx, 0)
        s = mx.recordio.pack_img(nheader, im, quality=95, img_fmt='.jpg')
        writer_val.write_idx(rec_val_idx, s)
        rec_val_idx+=1
    break
  return count


valid_name = {}
for line in open(os.path.join(args.dataset, 'gt_v2', 'IQIYI_VID.txt'), 'r'):
  name = line.strip()
  valid_name[name] = 1


name2path = {}
val_names = []
for part in PARTS:
  dataset = os.path.join(args.dataset, 'IQIYI_VID_DATA_Part%d'%part)
  for subdir in ['IQIYI_VID_TRAIN', 'IQIYI_VID_VAL']:
    _dir = os.path.join(dataset, subdir)
    _list = os.listdir(_dir)
    _list = sorted(_list)
    for video_file in _list:
      name = video_file
      if name not in valid_name:
        continue
      path = os.path.join(_dir, name)
      assert name not in name2path
      name2path[name] = path
      if subdir=='IQIYI_VID_VAL':
        val_names.append(name)
print(len(name2path), len(val_names))


gt_label = {}
gt_map = {}
ret_map = {}
for line in open(os.path.join(args.dataset, 'gt_v2', 'val_v2.txt'), 'r'):
  vec = line.strip().split()
  label = int(vec[0])
  if label>MAX_LABEL:
    continue
  if not label in gt_map:
    gt_map[label] = []
  if not label in ret_map:
    ret_map[label] = []
  for name in vec[1:]:
    assert name not in gt_label
    assert name in valid_name
    if name not in name2path:
      continue
    gt_label[name] = label
    gt_map[label].append(name)

vid = 0
for line in open(os.path.join(args.dataset, 'gt_v2', 'train_v2.txt'), 'r'):
  name, label = line.strip().split()
  label = int(label)
  if label>MAX_LABEL:
    continue
  #if name not in valid_name:
  #  continue
  if name not in name2path:
    continue
  vid+=1
  namehash = hash(name)
  mod = namehash%SPLIT[1]
  #print(namehash, mod)
  if mod!=SPLIT[0]:
    continue
  video_file = name2path[name]
  #assert os.path.exists(video_file)
  if not os.path.exists(video_file):
    print('XXXX not exists', video_file)
    continue
  timea = datetime.datetime.now()
  c = get_faces(video_file, label)
  timeb = datetime.datetime.now()
  diff = timeb - timea
  print(video_file, vid, c, diff.total_seconds())

#compute val start
vid = 0
for name in val_names:
  label = 0
  if name in gt_label:
    label = gt_label[name]
  if label<=0: #ignore distractors
    continue
  vid+=1
  namehash = hash(name)
  if namehash%SPLIT[1]!=SPLIT[0]:
    continue
  video_file = name2path[name]
  #assert os.path.exists(video_file)
  if not os.path.exists(video_file):
    print('XXXX not exists', video_file)
    continue
  c = get_faces(video_file, label)
  print(video_file, vid, c)



