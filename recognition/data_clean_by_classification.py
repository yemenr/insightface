import os
import sys
import argparse
import numpy as np
import shutil
import cv2
import tqdm
from image_iter_gluon import FaceImageDataset
import mxnet as mx
import pdb
sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
import noise_layer
import datetime

def main(args):    
    # load labelToNameFile
    labelToNameFile = open(args.labelToNameFile, "r")
    labelNameStr = [x.strip() for x in labelToNameFile.readlines()]
    labelNameMap = {}
    for tmpStr in labelNameStr:
        label, name = [x.strip() for x in tmpStr.split(",")]
        labelNameMap[label] = name

    # load data
    data_shape = (3,112,112)
    path_imgrec = os.path.join(args.dataDir, "train.rec")
    train_dataset = FaceImageDataset(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec
    )
    train_data = mx.gluon.data.DataLoader(train_dataset, args.batch_size, shuffle=False, last_batch="rollover", num_workers=2)
    #train_data = mx.gluon.data.DataLoader(train_dataset, args.batch_size, shuffle=True, last_batch="rollover", num_workers=2)
    train_dataiter = mx.contrib.io.DataLoaderIter(train_data)
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    
    # load model
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.mxnetModelPath.split(",")[0], int(args.mxnetModelPath.split(",")[1]))
    ctx = mx.gpu(0)
    internalSyms = sym.get_internals()
    sym = internalSyms["softmax_output"]
    mxModel = mx.mod.Module(symbol=sym, context=ctx)
    
    # inference
    mxModel.bind(data_shapes=[('data', (args.batch_size, 3, 112, 112))], label_shapes=[('softmax_label', (args.batch_size,))], for_training=False)
    mxModel.set_params(arg_params, aux_params)       
    print("===========", args.batch_size, "===========")
    end_of_batch = False
    data_iter = iter(train_dataiter)
    next_data_batch = next(data_iter)
    batchIdx = 0
    while not end_of_batch:
        data_batch = next_data_batch
        batchIdx += 1
        print("batchIdx: ", batchIdx)
        eData = data_batch.data[0].asnumpy().astype('int32')
        eLabel = data_batch.label[0].asnumpy().astype('int32').flatten()
        labelFlag = (eLabel > 179720)
        if labelFlag.sum() > 0: 
            begin = datetime.datetime.now()
            mxModel.forward(data_batch, is_train=False)
            pred_label = mxModel.get_outputs()[0]
            pred_label = mx.nd.argmax(pred_label, axis=1).asnumpy().astype('int32').flatten()        
            mx.nd.waitall()
            end = datetime.datetime.now()
            tmpCost = (end-begin).total_seconds()
            print("cost: ", tmpCost/args.batch_size)
            
            for k in range(args.batch_size):
                pLabel = pred_label[k]
                sLabel = eLabel[k]
                sData = eData[k].transpose((1,2,0))[:,:,::-1]
                if (sLabel > 179720) and (pLabel != sLabel):
                    outLabelDir = os.path.join(args.outDir, labelNameMap[str(sLabel)])
                    if not os.path.exists(outLabelDir):
                        os.mkdir(outLabelDir)
                        refLabelDir = os.path.join(args.refDir, labelNameMap[str(sLabel)])
                        if os.path.exists(refLabelDir):
                            anchorPaths = [n for n in os.listdir(refLabelDir) if "底库图片" in n]
                            if len(anchorPaths)==0:
                                print("warning! 底库不存在: %s" % labelNameMap[str(sLabel)])
                            for img in anchorPaths:
                                srcImg = os.path.join(refLabelDir, img) 
                                dstImg = os.path.join(outLabelDir, img) 
                                shutil.copyfile(srcImg, dstImg)
                        else:
                            print("warning! refLabelDir不存在: %s" % refLabelDir)
                        
                    mistakeFilePath = os.path.join(outLabelDir, "mistake.txt")
                    mistakeFile = open(mistakeFilePath, "a+")
                    mistakeTarget = str(pLabel)
                    if mistakeTarget not in labelNameMap.keys():
                        mistakeFile.write("%d, %s===>emoreglint\n" % (k, mistakeTarget))
                    else:                    
                        mistakeTarget = labelNameMap[mistakeTarget]
                        mistakeFile.write("%d, %s\n" % (k, mistakeTarget))
                    mistakeFile.close()
                    cv2.imwrite(os.path.join(outLabelDir,"%d_%d.png" % (batchIdx,k)), sData)
        try:
            # prefetch next batch
            next_data_batch = next(data_iter)
        except StopIteration:
            end_of_batch = True    
    print("done!!!!!!!!!!!!!")
    labelToNameFile.close()
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mxnetModelPath', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('labelToNameFile', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('dataDir', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('refDir', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('outDir', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in each context')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
'''
python statistic_cnt.py /home/cys/face_data/megaface/filtered_align_MegafaceIdentities_VGG
'''    
