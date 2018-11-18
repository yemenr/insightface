from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
#import facenet
import detect_face
import random
from time import sleep
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image
from skimage import transform as trans
import cv2
import pdb

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


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


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = face_image.get_dataset(args.name, args.input_dir, args.dataset_type)
    print('dataset size', args.name, len(dataset))
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    #image_size = [112,96]
    image_size = [112,112]
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32 )

    if image_size[1]==112:
        src[:,0] += 8.0

    # Add a random key to the filename to allow alignment using multiple processes
    #random_key = np.random.randint(0, high=99999)
    #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    #output_filename = os.path.join(output_dir, 'faceinsight_align_%s.lst' % args.name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = os.path.join(args.output_dir, 'lst')
    
    
    with open(output_filename, "w") as text_file:
        nrof_images_total = 0
        nrof = np.zeros( (5,), dtype=np.int32)
        for fimage in dataset:
            if nrof_images_total%100==0:
                print("Processing %d, (%s)" % (nrof_images_total, nrof))
            nrof_images_total += 1
            #if nrof_images_total<950000:
            #  continue
            image_path = fimage.image_path
            if not os.path.exists(image_path):
                print('image not found (%s)'%image_path)
                continue
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            #print(image_path)
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2:
                    print('Unable to align "%s", img dim error' % image_path)
                    #text_file.write('%s\n' % (output_filename))
                    continue
                if img.ndim == 2:
                    img = to_rgb(img)
                img = img[:,:,0:3]
                _paths = fimage.image_path.split('/')
                a,b,c = _paths[-3], _paths[-2], _paths[-1]
                target_dir = os.path.join(args.output_dir, a, b)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                #target_file = os.path.join(target_dir, c)
                #warped = None                
                    
                _minsize = img.shape[0]//4
                bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                det = bounding_boxes[:,0:4]
                scores = bounding_boxes[:,4]
                aligned_imgs = []
                img_size = np.asarray(img.shape)[0:2]
                print(c)
                pdb.set_trace()
                if (nrof_faces>0): 
                    if nrof_faces > 1:                
                        if args.detect_multiple_faces:
                            for i in range(nrof_faces):
                                if scores[i] > 0:
                                    bb = np.squeeze(det[i])
                                    bb[0] = max(0,bb[0])
                                    bb[1] = max(0,bb[1])
                                    bb[2] = min(bb[2],img_size[1])
                                    bb[3] = min(bb[3],img_size[0])
                                    
                                    if ((bb[0] >= img_size[1]) or (bb[1] >= img_size[0]) or (bb[2] > img_size[1]) or (bb[3] > img_size[0])):
                                        continue
                                    
                                    h = bb[3]-bb[1]
                                    w = bb[2]-bb[0]
                                    x = bb[0]
                                    y = bb[1]
                                                                        
                                    _w = int((float(h)/image_size[0])*image_size[1] )
                                    x += (w-_w)//2
                                    #x = min( max(0,x), img.shape[1] )
                                    x = max(0,x)
                                    xw = x+_w
                                    xw = min(xw, img.shape[1])
                                    roi = np.array( (x, y, xw, y+h), dtype=np.int32)
                                    
                                    faceImg = img[roi[1]:roi[3],roi[0]:roi[2],:]
                                    dst = points[:, i].reshape( (2,5) ).T
                                    tform = trans.SimilarityTransform()
                                    tform.estimate(dst, src)
                                    M = tform.params[0:2,:]
                                    warped = cv2.warpAffine(faceImg,M,(image_size[1],image_size[0]), borderValue = 0.0)
                                    
                                    if (warped is None) or (np.sum(warped) == 0):
                                        warped = faceImg
                                        warped = cv2.resize(warped, (image_size[1], image_size[0]))
                                    
                                    aligned_imgs.append(warped)
                        else:
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            img_center = img_size / 2
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            
                            bb = np.squeeze(det[index])
                            
                            bb[0] = max(0,bb[0])
                            bb[1] = max(0,bb[1])
                            bb[2] = min(bb[2],img_size[1])
                            bb[3] = min(bb[3],img_size[0])
                            
                            if ((bb[0] >= img_size[1]) or (bb[1] >= img_size[0]) or (bb[2] > img_size[1]) or (bb[3] > img_size[0])):
                                continue
                            
                            h = bb[3]-bb[1]
                            w = bb[2]-bb[0]
                            x = bb[0]
                            y = bb[1] 
                            _w = int((float(h)/image_size[0])*image_size[1] )
                            x += (w-_w)//2
                            #x = min( max(0,x), img.shape[1] )
                            x = max(0,x)
                            xw = x+_w
                            xw = min(xw, img.shape[1])
                            roi = np.array( (x, y, xw, y+h), dtype=np.int32)
                            
                            faceImg = img[roi[1]:roi[3],roi[0]:roi[2],:]
                            dst = points[:, index].reshape( (2,5) ).T
                            tform = trans.SimilarityTransform()
                            tform.estimate(dst, src)
                            M = tform.params[0:2,:]
                            warped = cv2.warpAffine(faceImg,M,(image_size[1],image_size[0]), borderValue = 0.0)
                            
                            if (warped is None) or (np.sum(warped) == 0):
                                warped = faceImg
                                warped = cv2.resize(warped, (image_size[1], image_size[0]))
                            
                            aligned_imgs.append(warped)
                    else:
                        bb = np.squeeze(det[0])
                        
                        bb[0] = max(0,bb[0])
                        bb[1] = max(0,bb[1])
                        bb[2] = min(bb[2],img_size[1])
                        bb[3] = min(bb[3],img_size[0])
                        
                        if ((bb[0] >= img_size[1]) or (bb[1] >= img_size[0]) or (bb[2] > img_size[1]) or (bb[3] > img_size[0])):
                            continue
                        
                        h = bb[3]-bb[1]
                        w = bb[2]-bb[0]
                        x = bb[0]
                        y = bb[1] 
                        _w = int((float(h)/image_size[0])*image_size[1] )
                        x += (w-_w)//2
                        #x = min( max(0,x), img.shape[1] )
                        x = max(0,x)
                        xw = x+_w
                        xw = min(xw, img.shape[1])
                        roi = np.array( (x, y, xw, y+h), dtype=np.int32)
                        
                        faceImg = img[roi[1]:roi[3],roi[0]:roi[2],:]
                        dst = points[:, 0].reshape( (2,5) ).T
                        tform = trans.SimilarityTransform()
                        tform.estimate(dst, src)
                        M = tform.params[0:2,:]
                        warped = cv2.warpAffine(faceImg,M,(image_size[1],image_size[0]), borderValue = 0.0)
                        
                        if (warped is None) or (np.sum(warped) == 0):
                            warped = faceImg
                            warped = cv2.resize(warped, (image_size[1], image_size[0]))
                        
                        aligned_imgs.append(warped)  
                    
                    for i, warped in enumerate(aligned_imgs):
                        target_file = os.path.join(target_dir, c+str(i)+'.png')
                        bgr = warped[...,::-1]
                        cv2.imwrite(target_file, bgr)
                        oline = '%d\t%s\t%d\n' % (1,target_file, int(fimage.classname))
                        text_file.write(oline)
                else:
                    print('Unable to align "%s", face detection error' % image_path)
                    #text_file.write('%s\n' % (output_filename))
                    continue

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('--name', type=str, help='dataset name, can be facescrub, megaface, webface, celeb.')
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--dataset-type', type=str, default='id' ,help='dataset type: id | seq.')
    #parser.add_argument('--image_size', type=str, help='Image size (height, width) in pixels.', default='112,112')
    #parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

