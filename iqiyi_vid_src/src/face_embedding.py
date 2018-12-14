from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
  def __init__(self, model, gpu_id = 0, feature_norm = True):

    self.det_minsize = 60
    self.det_threshold = [0.6,0.7,0.7]
    #self.det_factor = 0.8
    image_size = (112, 112)
    self.image_size = image_size
    ctx = mx.gpu(gpu_id)
    self.model = None
    if model is not None:
      _vec = model.split(',')
      assert len(_vec)==2
      prefix = _vec[0]
      epoch = int(_vec[1])
      print('loading',prefix, epoch)
      sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
      all_layers = sym.get_internals()
      sym = all_layers['fc1_output']
      model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
      #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
      model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
      model.set_params(arg_params, aux_params)
      self.model = model
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold, minsize=self.det_minsize)
    self.detector = detector
    self.feature_norm = feature_norm


  def get(self, face_img):
    #face_img is bgr image
    ret = self.detector.detect_face(face_img)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    rimg = nimg
    embedding = None
    norm = 999.
    if self.model is not None:
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      #print(nimg.shape)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      embedding = self.model.get_outputs()[0].asnumpy().flatten()
      norm = np.linalg.norm(embedding)
      if self.feature_norm:
        embedding /= norm
    return (embedding, rimg, norm)

  def get_all(self, face_img):
    #face_img is bgr image
    ret = self.detector.detect_face(face_img)
    if ret is None:
      return None
    bbox, points = ret
    #print(bbox.shape)
    if bbox.shape[0]==0:
      return None
    result = []
    for i in xrange(bbox.shape[0]):
      _bbox = bbox[i,0:4]
      _points = points[i,:].reshape((2,5)).T
      #print(bbox)
      #print(points)
      nimg = face_preprocess.preprocess(face_img, _bbox, _points, image_size='112,112')
      rimg = nimg
      result.append( (rimg, _bbox) )
      #embedding = None
      #norm = 999.
      #if self.model is not None:
      #  nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      #  aligned = np.transpose(nimg, (2,0,1))
      #  #print(nimg.shape)
      #  input_blob = np.expand_dims(aligned, axis=0)
      #  data = mx.nd.array(input_blob)
      #  db = mx.io.DataBatch(data=(data,))
      #  self.model.forward(db, is_train=False)
      #  embedding = self.model.get_outputs()[0].asnumpy().flatten()
      #  norm = np.linalg.norm(embedding)
      #  if self.feature_norm:
      #    embedding /= norm
      #result.append((embedding,rimg,norm,_bbox))
    return result

  def get_features(self, imgs):
    ret = np.zeros( (len(imgs), 512), dtype=np.float32)
    batch_size = 32
    a = 0
    while True:
      b = min(a+batch_size, len(imgs))
      if b==a:
        break
      count = b-a
      data = np.zeros( (count, 3, 112,112), dtype=np.float32)
      for idx in range(a,b):
        nimg = imgs[idx]
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        data[idx-a] = aligned
      data = mx.nd.array(data)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      embedding = self.model.get_outputs()[0].asnumpy()
      #print(embedding.shape)
      ret[a:b,:] = embedding.copy()
      a = b
    return ret




