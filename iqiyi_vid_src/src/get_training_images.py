import argparse
import cv2
import os
import sys
import datetime
import time
import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pdb
import threading

def div_list(ls,n):
    if not isinstance(ls,list) or not isinstance(n,int):
        return []
    ls_len = len(ls)
    if n<=0 or 0==ls_len or n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len//n
        k = ls_len%n
        ls_return = []
        for i in range(0,(n-1)*j,j):
            ls_return.append(ls[i:i+j])
        ls_return.append(ls[(n-1)*j:])
        return ls_return

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--model', default='', help='path to load model.') #0.875
parser.add_argument('--input_dir', default='./feat1', help='')
parser.add_argument('--output_dir', default='./feat1', help='')
parser.add_argument('--sampling', default=3, type=int, help='')
args = parser.parse_args()

# valid video map
valid_name = {}
for line in open(os.path.join(args.input_dir, 'IQIYI_VID.txt'), 'r'):
    name = line.strip()
    valid_name[name] = 1
print('all valid video names num: %d' % len(valid_name))
    
# get all video paths    
name2path = {}
for k in range(1,4):
    partPath = os.path.join(args.input_dir, 'IQIYI_VID_DATA_Part%d' % k)
    trainDataPath = os.path.join(partPath, 'IQIYI_VID_TRAIN')
    valDataPath = os.path.join(partPath, 'IQIYI_VID_VAL')
    for dataDir in [trainDataPath, valDataPath]:
        _list = os.listdir(dataDir)
        _list = sorted(_list)
        for video_file in _list:
            name = video_file
            if name not in valid_name: # pass invalid video
                continue
            path = os.path.join(dataDir, name)
            assert name not in name2path
            name2path[name] = path
print('all video num: %d' % len(name2path))

# get all labeled info
gt_label = {}
gt_map = {}
for line in open(os.path.join(args.input_dir, 'val_v2.txt'), 'r'):
    vec = line.strip().split()
    label = int(vec[0])
    if not label in gt_map:
        gt_map[label] = []
    for name in vec[1:]:
        assert name not in gt_label # make sure no duplicate
        assert name in valid_name # clean data
        if name not in name2path: # video exists
            continue
        gt_label[name] = label
        gt_map[label].append(name)

for line in open(os.path.join(args.input_dir, 'train_v2.txt'), 'r'):
    name, label = line.strip().split()
    label = int(label)
    assert name not in gt_label
    assert name in valid_name
    if name not in name2path:
        continue
    gt_label[name] = label
    gt_map[label].append(name)
print('counts of labeled videos:%d, counts of labels:%d\n' % (len(gt_label), len(gt_map)))
print('`````````````````````````````````````')
sortedLabeledInfo = sorted([[len(items[1]), items[0]] for items in gt_map.items()])
for cnts, label in sortedLabeledInfo:
    print(cnts, label)
print('`````````````````````````````````````')

#numList = [v[0] for v in sortedLabeledInfo]
#plt.bar(range(len(num_list)), num_list)
#plt.show()

# make all dirs
for label in gt_map.keys():
    labelDirName = os.path.join(args.output_dir, 'iqiyi_'+str(label))
    if not os.path.exists(labelDirName):
        os.mkdir(labelDirName)

proNum = 8
proNames = div_list(list(gt_label.keys()), proNum)

def th_process(vids):
    global gt_label, gt_map, name2path
    for name in vids:        
        outImgsDir = os.path.join(args.output_dir, 'iqiyi_'+str(gt_label[name]))
        print('thread %s is running...%s,%s' % (threading.current_thread().name, gt_label[name], name))
        video = name2path[name]
        labelCnt = len(gt_map[gt_label[name]])
        if labelCnt <= 20:
            sampling = 1
        elif labelCnt <= 40:
            sampling = 2
        else:
            sampling = 3
            
        cap = cv2.VideoCapture(video)
        frame_num = 0
        while cap.isOpened(): 
            ret, frame = cap.read() 
            if frame is None:
                break
            frame_num+=1
            if frame_num%sampling!=0:
                continue
            frame = cv2.resize(frame, (888, 480))
            #print('frame', frame.shape)
            cv2.imwrite(os.path.join(outImgsDir, str(frame_num//sampling)+'.jpg'), frame)            
        cap.release()

# multiprocess
oppo = 0
ths = []
for names in proNames:
    oppo += 1
    t = threading.Thread(target=th_process, args=(names,), name='th'+str(oppo))
    t.start()
    ths.append(t)

for t in ths:
    t.join()

print("processed end!")
