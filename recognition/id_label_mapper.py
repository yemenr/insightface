import os
import pdb
import sys
import argparse
import numpy as np
import shutil
import cv2
import tqdm

def main(args):    
    srcLabelToNameFile = open(os.path.join(args.src_dir, "label_to_name.txt"), "r")
    dstLabelToNameFile = open(os.path.join(args.out_dir, "label_to_name.txt"), "w")
    srcIdToPathFile = open(os.path.join(args.src_dir, "id_to_path.txt"), "r")
    dstIdToPathFile = open(os.path.join(args.out_dir, "id_to_path.txt"), "w")
    newIdToOldIdFile = open(os.path.join(args.src_dir, "newId_to_oldId_1.txt"), "r")
    newLabelToOldLabelFile = open(os.path.join(args.src_dir, "newLabel_to_oldLabel_1.txt"), "r")
    
    srcLabelNamesStr = [xx.strip() for xx in srcLabelToNameFile.readlines()]
    srcIdPathsStr = [xx.strip() for xx in srcIdToPathFile.readlines()]
    idsMapperStr = [xx.strip() for xx in newIdToOldIdFile.readlines()]
    labelsMapperStr = [xx.strip() for xx in newLabelToOldLabelFile.readlines()]
    
    srcLabelNameMapper = {}
    srcIdPathMapper = {}    
    
    for tmpStr in srcLabelNamesStr:
        tmpKey, tmpValue = [x.strip() for x in tmpStr.split(",")]
        srcLabelNameMapper[tmpKey] = tmpValue
    for tmpStr in srcIdPathsStr:
        tmpKey, tmpValue = [x.strip() for x in tmpStr.split(",")]
        srcIdPathMapper[tmpKey] = tmpValue
    for tmpStr in idsMapperStr:
        newId, oldId = [x.strip() for x in tmpStr.split(",")]
        path = srcIdPathMapper[oldId]
        dstIdToPathFile.write("%s, %s\n" % (newId, path))
        
    for tmpStr in labelsMapperStr:
        newLabel, oldLabel = [x.strip() for x in tmpStr.split(",")]
        name = srcLabelNameMapper[oldLabel]
        dstLabelToNameFile.write("%s, %s\n" % (newLabel, name))
    srcLabelToNameFile.close()
    dstLabelToNameFile.close()
    srcIdToPathFile.close()
    dstIdToPathFile.close()
    newIdToOldIdFile.close()
    newLabelToOldLabelFile.close()
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('src_dir', type=str,help='Path to the data directory containing aligned face batches.')
    parser.add_argument('out_dir', type=str,help='Path to the data directory containing aligned face batches.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
'''
python statistic_cnt.py /home/cys/face_data/megaface/filtered_align_MegafaceIdentities_VGG
'''    
