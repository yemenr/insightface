import argparse
import cv2
import mxnet as mx
import os
import sys
import datetime
import time
import pickle
import numpy as np
import sklearn


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/iqiyi-scene,7', help='path to load model.') #0.875
parser.add_argument('--output', default='./gpu/data2/jiaguo/iqiyi/scenepred', help='')
parser.add_argument('--dataset', default='/gpu/data1/jiaguo/iqiyi', help='')
parser.add_argument('--gpu', default=6, type=int, help='gpu id')
parser.add_argument('--sampling', default=10, type=int, help='')
parser.add_argument('--split', default='', type=str, help='')
args = parser.parse_args()


model = None

PARTS = [1]
MAX_LABEL = 574

PARTS = [1, 2, 3]
MAX_LABEL = 99999

print(args, MAX_LABEL, PARTS)


SPLIT = [0, 1]
if len(args.split)>0:
  _v = args.split.split(',')
  SPLIT[0] = int(_v[0])
  SPLIT[1] = int(_v[1])

print('SPLIT:', SPLIT)

ctx = mx.gpu(args.gpu)
_vec = args.model.split(',')
assert len(_vec)==2
prefix = _vec[0]
epoch = int(_vec[1])
print('loading',prefix, epoch)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['fc_output']
sym = mx.sym.SoftmaxActivation(sym)
model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
#model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
model.bind(data_shapes=[('data', (1, 3, 224, 224))])
model.set_params(arg_params, aux_params)

def get_faces(video):
  batch_size = 64
  R = []
  sampling = args.sampling
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
    img1 = frame[:,0:256,:]
    img2 = frame[:,108:364,:]
    img3 = frame[:,217:473,:]
    print(img1.shape, img2.shape, img3.shape)
    img1 = cv2.resize(img1, (224,224))
    img2 = cv2.resize(img2, (224,224))
    img3 = cv2.resize(img3, (224,224))
    for img in [img1, img2, img3]:
      img = cv2.resize(img, (224, 224))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = np.transpose(img, (2,0,1))
      R.append(img)
  if len(R)>batch_size:
    random.shuffle(R)
    R = R[0:batch_size]
  data = np.zeros( (len(R), 3, 224, 224), dtype=np.float32 )
  for i in range(len(R)):
    data[i] = R[i]
  data = mx.nd.array(data)
  db = mx.io.DataBatch(data=(data,))
  model.forward(db, is_train=False)
  ret = model.get_outputs()[0].asnumpy()
  xscore = np.mean(ret, axis=0).flatten()
  print(xscore.shape)
  return xscore



valid_name = {}
for line in open(os.path.join(args.dataset, 'gt_v2', 'IQIYI_VID.txt'), 'r'):
  name = line.strip()
  valid_name[name] = 1


name2path = {}
test_names = []
dataset = os.path.join(args.dataset, 'IQIYI_VID_TEST')
_list = os.listdir(dataset)
_list = sorted(_list)
for video_file in _list:
  name = video_file
  path = os.path.join(dataset, name)
  name2path[name] = path
  test_names.append(name)
print(len(name2path), len(test_names))



train_filename = args.output
assert not os.path.exists(train_filename)
f = open(train_filename, 'wb')

#compute test start
vid = 0
for name in test_names:
  label = -1
  vid+=1
  namehash = hash(name)
  if namehash%SPLIT[1]!=SPLIT[0]:
    continue
  video_file = name2path[name]
  #assert os.path.exists(video_file)
  if not os.path.exists(video_file):
    print('XXXX not exists', video_file)
    continue
  xscore = get_faces(video_file, False)
  print(video_file, vid)
  pickle.dump((name, xscore), f, protocol=pickle.HIGHEST_PROTOCOL)

f.close()

