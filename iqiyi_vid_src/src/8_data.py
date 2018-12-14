import mxnet as mx
import os

input_dir = '/gpu/data1/jiaguo/iqiyi/scene_rec'
output_dir = '/gpu/data1/jiaguo/iqiyi/scene_rec2'
split = 'val'
path_imgrec = os.path.join(input_dir, '%s.rec'%split)
path_imgidx = os.path.join(input_dir, '%s.idx'%split)
imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
w_path_imgrec = os.path.join(output_dir, '%s.rec'%split)
w_path_imgidx = os.path.join(output_dir, '%s.idx'%split)
writer = mx.recordio.MXIndexedRecordIO(w_path_imgidx, w_path_imgrec, 'w')  # pylint: disable=redefined-variable-type
idx = 0
while True:
  if idx%10000==0:
    print('processing', idx)
  s = imgrec.read()
  if s is None:
    break
  header, c = mx.recordio.unpack(s)
  label = header.label
  #print(label, label.__class__)
  if label==0.0:
    #print('found 0, next')
    continue
  nheader = mx.recordio.IRHeader(0, label, idx, 0)
  s = mx.recordio.pack(nheader, c)
  writer.write_idx(idx, s)
  idx+=1
print('total', idx)
