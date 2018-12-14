import face_embedding
import argparse
import cv2
import os
import sys
import datetime
import time
import pickle
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from ssh_detector import SSHDetector
import face_preprocess

def IOU(Reframe,GTframe):
  x1 = Reframe[0];
  y1 = Reframe[1];
  width1 = Reframe[2]-Reframe[0];
  height1 = Reframe[3]-Reframe[1];

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.') #0.875
#parser.add_argument('--model', default='./models2/model-r100-sfza/model,0', help='path to load model.') #0.875
#parser.add_argument('--model', default='../iqiyi-competition/models2/model-r100-iqiyi/model,6', help='path to load model.') #0.875
parser.add_argument('--output', default='./feat1', help='')
parser.add_argument('--dataset', default='/gpu/data1/jiaguo/iqiyi', help='')
parser.add_argument('--gpu', default=6, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--sampling', default=3, type=int, help='')
parser.add_argument('--split', default='', type=str, help='')
parser.add_argument('--threshold', default=0.9, type=float, help='clustering dist threshold')
parser.add_argument('--quality-threshold', default=10.0, type=float, help='quality threshold')
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


detector = SSHDetector('../model/detc', 0, ctx_id=args.gpu, test_mode=False)

def get_faces(video, is_train=True):
  R = []
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
      frame = cv2.resize(frame, (888, 480))
      #print('frame', frame.shape)
      #faces = model.get_all(frame)
      faces = detector.detect(frame, 0.5, scales=[1.0])
      if faces is None or faces.shape[0]==0:
        continue
      det = faces
      #det = np.zeros( (len(faces), 4), dtype=np.float32)
      #for f in range(len(faces)):
      #  _face = faces[f]
      #  det[f] = _face[1]
      img_size = np.asarray(frame.shape)[0:2]
      bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
      img_center = img_size / 2
      offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
      offset_dist_squared = np.sum(np.power(offsets,2.0),0)
      bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
      face = faces[bindex]
      landmark = face[5:15].reshape((5,2))
      bbox = face[0:4]
      rimg = face_preprocess.preprocess(frame, None, landmark, image_size='112,112')
      R.append( (frame_num, rimg, bbox, landmark) )
    if is_train:
      break
    else:
      if len(R)>0 or sampling==1:
        break
    sampling = 1
  return R

def compact_faces(R):
  if len(R)==0:
    return []
  imgs = []
  for r in R:
    imgs.append(r[1])
  features = model.get_features(imgs)
  assert features.shape[0]==len(imgs)
  X = []
  R2 = []
  for i in xrange(features.shape[0]):
    f = features[i]
    norm = np.linalg.norm(f)
    if norm<args.quality_threshold:
      continue
    img = R[i][1]
    img_encode = cv2.imencode('.jpg', img)[1]
    #print(img_encode.__class__)

    R2.append( (R[i][0], img_encode, R[i][2], R[i][3]) )
  return R2


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
if model is None:
  model = face_embedding.FaceModel(model=args.model, gpu_id = args.gpu, feature_norm=True)

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
  R = get_faces(video_file, False)
  print(video_file, vid, len(R))
  R = compact_faces(R)
  if len(R)==0:
    continue
  flag = 3
  pickle.dump((name, R, label, flag), f, protocol=pickle.HIGHEST_PROTOCOL)

f.close()

