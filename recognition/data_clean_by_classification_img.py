import os
import sys
import argparse
import numpy as np
import shutil
import cv2
import tqdm
from image_iter_gluon_ import FaceImageDataset
import mxnet as mx
import pdb
sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
import noise_layer
import datetime
#from queue import Queue
from multiprocessing import Queue

def main(args):    
    # load labelToNameFile
    labelToNameFile = open(args.labelToNameFile, "r")
    labelNameStr = [x.strip() for x in labelToNameFile.readlines()]
    labelNameMap = {}
    nameLabelMap = {}
    for tmpStr in labelNameStr:
        label, name = [x.strip() for x in tmpStr.split(",")]
        labelNameMap[label] = name
        nameLabelMap[name] = label

    # load idToPathFile
    idToPathFile = open(args.idToPathFile, "r")
    idPathStr = [x.strip() for x in idToPathFile.readlines()]
    idPathMap = {}
    for tmpStr in idPathStr:
        xid, xpath = [x.strip() for x in tmpStr.split(",")]
        idPathMap[int(xid)] = xpath

    # load newLabelToOldLabel
    lolMap = {}
    lolFile = open(args.nLabelToOLabelFile, "r")
    lolStr = [x.strip() for x in lolFile.readlines()]
    for tmpStr in lolStr:
        nLabel, oLabel = [x.strip() for x in tmpStr.split(',')]
        lolMap[int(nLabel)] = int(oLabel)

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
    
    im_tensor = np.zeros((args.batch_size, 3, 112, 112), dtype='float32')    
    label_tensor = np.zeros((args.batch_size, ), dtype='float32')    
    for dataLabelName in tqdm.tqdm(os.listdir(args.dataDir)):
        dataLabelDir = os.path.join(args.dataDir, dataLabelName)
        refLabelDir = os.path.join(args.refDir, dataLabelName)
        
        idxInBatch = 0
        facePathList = []
        srcPathList = []
        for imgName in tqdm.tqdm(os.listdir(dataLabelDir)):            
            if idxInBatch == args.batch_size:
                batchData = mx.nd.array(im_tensor, dtype='float32')
                labelData = mx.nd.array(label_tensor, dtype='float32')
                db = mx.io.DataBatch(data=(batchData,), label=(labelData, ), provide_data=[('data', batchData.shape)], provide_label=[('softmax_label', labelData.shape)])
                begin = datetime.datetime.now()
                mxModel.forward(db, is_train=False)
                pred_label = mxModel.get_outputs()[0]
                pred_label = mx.nd.argmax(pred_label, axis=1).asnumpy().astype('int32').flatten()        
                mx.nd.waitall()
                end = datetime.datetime.now()
                tmpCost = (end-begin).total_seconds()
                print("cost: ", tmpCost/args.batch_size)
                
                for k in range(idxInBatch):
                    pLabel = pred_label[k]
                    labelName = "other"
                    if pLabel in lolMap.keys():
                        pLabel = lolMap[pLabel]
                        labelName = labelNameMap[str(pLabel)]
                    faceName = facePathList[k].split("/")[-1]
                    srcName = srcPathList[k].split("/")[-1]
                    #if not os.path.exists(srcPath):
                    #    #pdb.set_trace()
                    #    print(srcPath, " does not exists!")
                    #    continue
                    
                    if (dataLabelName != labelName):
                        outLabelDir = os.path.join(args.outDir, dataLabelName)
                        if not os.path.exists(outLabelDir):
                            os.mkdir(outLabelDir)
                            if os.path.exists(refLabelDir):
                                anchorPaths = [n for n in os.listdir(refLabelDir) if "底库图片" in n]
                                if len(anchorPaths)==0:
                                    print("warning! 底库不存在: %s" % dataLabelName)
                                for img in anchorPaths:
                                    srcImg = os.path.join(refLabelDir, img) 
                                    dstImg = os.path.join(outLabelDir, img) 
                                    shutil.copyfile(srcImg, dstImg)
                            else:
                                print("warning! refLabelDir不存在: %s" % refLabelDir)
                            
                        mistakeFilePath = os.path.join(outLabelDir, "mistake.txt")
                        mistakeFile = open(mistakeFilePath, "a+")
                        mistakeFile.write("%s, %s\n" % (faceName, labelName))
                        mistakeFile.close()                        
                        shutil.copyfile(facePathList[k], os.path.join(outLabelDir, faceName))
                        if "底库图片" not in srcName:
                            shutil.copyfile(srcPathList[k], os.path.join(outLabelDir, srcName))
                facePathList.clear()
                srcPathList.clear()
                idxInBatch = 0
                    
            idxInBatch += 1
            facePath = os.path.join(dataLabelDir, imgName)
            srcPath = os.path.join(refLabelDir, imgName.split("0.png")[0])
            facePathList.append(facePath)
            srcPathList.append(srcPath)
        
            img = cv2.imread(facePath)
            for i in range(3):
                im_tensor[idxInBatch-1, 2-i, :, :] = img[:, :, i]
            label_tensor[idxInBatch-1] = int(nameLabelMap[dataLabelName])
            
        if idxInBatch > 0:
            batchData = mx.nd.array(im_tensor, dtype='float32')
            labelData = mx.nd.array(label_tensor, dtype='float32')
            db = mx.io.DataBatch(data=(batchData,), label=(labelData, ), provide_data=[('data', batchData.shape)], provide_label=[('softmax_label', labelData.shape)])
            begin = datetime.datetime.now()
            mxModel.forward(db, is_train=False)
            pred_label = mxModel.get_outputs()[0]
            pred_label = mx.nd.argmax(pred_label, axis=1).asnumpy().astype('int32').flatten()        
            mx.nd.waitall()
            end = datetime.datetime.now()
            tmpCost = (end-begin).total_seconds()
            print("cost: ", tmpCost/args.batch_size)
            
            for k in range(idxInBatch):
                pLabel = pred_label[k]
                labelName = "other"
                if pLabel in lolMap.keys():
                    pLabel = lolMap[pLabel]
                    labelName = labelNameMap[str(pLabel)]
                faceName = facePathList[k].split("/")[-1]
                srcName = srcPathList[k].split("/")[-1]
                #if not os.path.exists(srcPath):
                #    #pdb.set_trace()
                #    print(srcPath, " does not exists!")
                #    continue
                
                if (dataLabelName != labelName):
                    outLabelDir = os.path.join(args.outDir, dataLabelName)
                    if not os.path.exists(outLabelDir):
                        os.mkdir(outLabelDir)
                        if os.path.exists(refLabelDir):
                            anchorPaths = [n for n in os.listdir(refLabelDir) if "底库图片" in n]
                            if len(anchorPaths)==0:
                                print("warning! 底库不存在: %s" % dataLabelName)
                            for img in anchorPaths:
                                srcImg = os.path.join(refLabelDir, img) 
                                dstImg = os.path.join(outLabelDir, img) 
                                shutil.copyfile(srcImg, dstImg)
                        else:
                            print("warning! refLabelDir不存在: %s" % refLabelDir)
                        
                    mistakeFilePath = os.path.join(outLabelDir, "mistake.txt")
                    mistakeFile = open(mistakeFilePath, "a+")
                    mistakeFile.write("%s, %s\n" % (faceName, labelName))
                    mistakeFile.close()                        
                    shutil.copyfile(facePathList[k], os.path.join(outLabelDir, faceName))
                    if "底库图片" not in srcName:
                        shutil.copyfile(srcPathList[k], os.path.join(outLabelDir, srcName))
            facePathList.clear()
            srcPathList.clear()
            idxInBatch = 0
    print("done!!!!!!!!!!!!!")
    labelToNameFile.close()
    lolFile.close()
    idToPathFile.close()
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mxnetModelPath', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('labelToNameFile', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('idToPathFile', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('nLabelToOLabelFile', type=str,help='Path to the data directory containing aligned face batches.')
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
