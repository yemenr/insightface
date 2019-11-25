from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx
import pdb
sys.path.append(os.path.join(os.path.dirname(__file__), '../recognition/losses'))
import noise_layer

parser = argparse.ArgumentParser(description='face model slim')
# general
parser.add_argument('--model', default='../models/model-r34-amf/model,60', help='path to load model.')
args = parser.parse_args()

_vec = args.model.split(',')
assert len(_vec)==2
prefix = _vec[0]
epoch = int(_vec[1])
print('loading',prefix, epoch)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
#digraph = mx.viz.plot_network(sym, node_attrs={"shape":"oval", "fixedsize":"false"})
#digraph.view()

thres = 1e-15
# modify params
for k,v in arg_params.items():
  absV = v.abs()
  Idx = absV < thres
  newV = mx.nd.where(Idx, mx.nd.zeros_like(v), v)
  arg_params[k] = newV

for k,v in aux_params.items():
  absV = v.abs()
  Idx = absV < thres
  newV = mx.nd.where(Idx, mx.nd.zeros_like(v), v)
  aux_params[k] = newV

all_layers = sym.get_internals()
sym = all_layers['fc1_output']
dellist = []
for k,v in arg_params.items():
  if k.startswith('fc7'):
    dellist.append(k)
for d in dellist:
  del arg_params[d]
mx.model.save_checkpoint(prefix+"s", 0, sym, arg_params, aux_params)

