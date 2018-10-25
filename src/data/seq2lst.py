
import sys
import os
import numpy as np

data_dir = sys.argv[1]
lmk_file_path = sys.argv[2]

if not os.path.exists(lmk_file_path):
    print("%s not exist" % lmk_file_path)
    return
    
idx = 0
cnt = 0
lmap = {}
for line in open(lmk_file_path, 'r'):
    idx+=1
    vec = line.strip().split(' ')
    assert len(vec)==12
    image_file = os.path.join(data_dir, vec[0])
    assert image_file.endswith('.jpg')
    if os.path.exists(image_file):
        cnt += 1
        label = int(vec[1])
        if label in lmap:
            vlabel = lmap[label]
        else:
            vlabel = -1-len(lmap)
            lmap[label] = vlabel
        lmk = np.array([float(x) for x in vec[2:]], dtype=np.float32)
        lmk = lmk.reshape( (5,2) ).T
        lmk_str = "\t".join( [str(x) for x in lmk.flatten()] )
        print("0\t%s\t%d\t0\t0\t0\t0\t%s"%(image_file, vlabel, lmk_str))
    if idx>10:
        break

print("classNum: %s , imgCnt: %s" % (len(lmap),cnt))