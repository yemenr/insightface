from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms

logger = logging.getLogger()

class FaceImageDataset(Dataset):

    def __init__(self, batch_size, data_shape,
                 path_imgrec = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0, color_jittering = 0,
                 images_filter = 0,
                 data_name='data', label_name='softmax_label', **kwargs):
        assert path_imgrec
        logging.info('loading recordio %s...', path_imgrec)
        path_imgidx = path_imgrec[0:-4]+".idx"
        self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)
        print('header0 label', header.label)
        self.header0 = (int(header.label[0]), int(header.label[1]))
        self.imgidx = []
        self.id2range = {}
        self.seq_identity = range(int(header.label[0]), int(header.label[1]))
        for identity in self.seq_identity:
            s = self.imgrec.read_idx(identity)
            header, _ = recordio.unpack(s)
            a,b = int(header.label[0]), int(header.label[1])
            count = b-a
            if count<images_filter:
                continue
            self.id2range[identity] = (a,b)
            self.imgidx += range(a, b)
        print('len(id2range.keys()): ', len(self.id2range))
        print("len(self.imgidx): ", len(self.imgidx))

        self.mean = mean
        self.nd_mean = None
        if self.mean:
            self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
            self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.color_jittering = color_jittering
        self.CJA = mx.image.ColorJitterAug(0.125, 0.125, 0.125)
        self.provide_label = [(label_name, (batch_size,))]
        
        self._transform = transforms.Compose([
                                                #transforms.RandomBrightness(0.3),
                                                #transforms.RandomContrast(0.3),
                                                #transforms.RandomSaturation(0.3),
                                                transforms.RandomFlipLeftRight(),
                                                #transforms.RandomRotation([-10.0, 10.0], zoom_out=True, rotate_with_proba=1.0),
                                                #transforms.ToTensor()
                                            ])

    def __len__(self):
      return len(self.imgidx)

    def __getitem__(self, idx):
        """Helper function for reading in next sample."""
        imgIdx = self.imgidx[idx]
        s = self.imgrec.read_idx(imgIdx)
        header, imgBytes = recordio.unpack(s)
        label = header.label      
        if not isinstance(label, numbers.Number):
            label = label[0]
        #_label = mx.nd.array([label])
        _data = self.imdecode(imgBytes)
        '''
        if label > 179720:
            xx = _data[:,:,::-1].asnumpy()
            print("label:%d, id:%d" % (label, header.id))
            cv2.imwrite("/home/ubun/camel/data/exp/"+str(header.id)+".png", xx)
            pdb.set_trace()
        '''
        if _data.shape[0] != self.data_shape[1]:
            _data = mx.image.resize_short(_data, self.data_shape[1])

        if self.cutoff>0:
            #_rd = random.randint(0,1)
            _rd = 1
            # p > 0.6
            if np.random.random() > 0.6:
                _rd = 0
            else:
                _rd = 1
            if _rd==1:
                #print('do cutoff aug', self.cutoff)
                #centerh = random.randint(0, _data.shape[0]-1)
                #centerw = random.randint(0, _data.shape[1]-1)
                #half = self.cutoff//2
                #starth = max(0, centerh-half)
                #endh = min(_data.shape[0], centerh+half)
                #startw = max(0, centerw-half)
                #endw = min(_data.shape[1], centerw+half)
                #print(starth, endh, startw, endw, _data.shape)
                h0 = random.randint(56,76)
                _data[h0:112, 10:102, :] = 128

        #x = _data.asnumpy()[...,::-1]
        #cv2.imwrite("1.png", x)

        if self._transform is not None:
            _data = self._transform(_data)
        #x = _data.asnumpy()[...,::-1]
        #cv2.imwrite("3.png", x)
        _data = self.postprocess_data(_data)

        # do rotating
        #rot_angle = random.randint(-15,15)
        #args = {'src': _data, 'rotation_degrees': rot_angle, 'zoom_in': False, 'zoom_out': False}
        #_data = mx.image.imrotate(**args)        
        #x = _data.transpose((1,2,0)).asnumpy()[...,::-1]
        #cv2.imwrite("2.png", x)
        #pdb.set_trace()
        return _data, label

    def brightness_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      src *= alpha
      return src

    def contrast_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = nd.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
      src *= alpha
      src += gray
      return src

    def saturation_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = nd.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = nd.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src

    def color_aug(self, img, x):
      #augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
      #random.shuffle(augs)
      #for aug in augs:
      #  #print(img.shape)
      #  img = aug(img, x)
      #  #print(img.shape)
      #return img
      return self.CJA(img)

    def mirror_aug(self, img):
      _rd = random.randint(0,1)
      if _rd==1:
        for c in range(img.shape[2]):
          img[:,:,c] = np.fliplr(img[:,:,c])
      return img

    def compress_aug(self, img):
      from PIL import Image
      from io import BytesIO
      buf = BytesIO()
      img = Image.fromarray(img.asnumpy(), 'RGB')
      q = random.randint(2, 20)
      img.save(buf, format='JPEG', quality=q)
      buf = buf.getvalue()
      img = Image.open(BytesIO(buf))
      return nd.array(np.asarray(img, 'float32'))

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s) #mx.ndarray
        img = img.astype('float32')
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))
