import sys
import os
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config, default
import numpy as np
from mxnet.gluon import nn

swish_index = 0

def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body

#def Act(data, act_type, name):
#    #ignore param act_type, set it in this function 
#    if act_type=='prelu':
#      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
#    else:
#      body = mx.sym.Activation(data=data, act_type=act_type, name=name)
#    return body

def gluon_act(act_type):
    if act_type=='prelu':
        return nn.PReLU()
    else:
        return nn.Activation(act_type)

def Act(data, act_type, name = 'act', lr_mult = 1.0):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name, lr_mult = lr_mult)
    elif act_type == 'swish':
      tmp_sigmoid = mx.symbol.Activation(data=data, act_type='sigmoid', name=name + '_sigmoid')
      global swish_index
      body = mx.symbol.elemwise_mul(data, tmp_sigmoid, name= 'swish_' + str(swish_index))
      swish_index += 1
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)

    return body

bn_mom = config.bn_mom
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)    
    return bn

def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
  body = last_conv
  if fc_type=='Z':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = body
  elif fc_type=='E':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=default.memonger)
  elif fc_type=='FC':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=default.memonger)
  elif fc_type=='SFC':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    body = Conv(data=body, num_filter=input_channel, kernel=(3,3), stride=(2,2), pad=(1,1),
                              no_bias=True, name="convf", num_group = input_channel)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bnf', cudnn_off=default.memonger)
    body = Act(data=body, act_type=config.net_act, name='reluf')
    body = Conv(data=body, num_filter=input_channel, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="convf2")
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bnf2', cudnn_off=default.memonger)
    body = Act(data=body, act_type=config.net_act, name='reluf2')
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=default.memonger)
  elif fc_type=='GAP':
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    relu1 = Act(data=bn1, act_type=config.net_act, name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=default.memonger)
  elif fc_type=='GNAP': #mobilefacenet++
    filters_in = 512 # param in mobilefacenet
    if num_classes>filters_in:
      body = mx.sym.Convolution(data=last_conv, num_filter=num_classes, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, name='convx')
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=0.9, name='convx_bn', cudnn_off=default.memonger)
      body = Act(data=body, act_type=config.net_act, name='convx_relu')
      filters_in = num_classes
    else:
      body = last_conv
    body = mx.sym.BatchNorm(data=body, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn6f', cudnn_off=default.memonger)  

    spatial_norm=body*body
    spatial_norm=mx.sym.sum(data=spatial_norm, axis=1, keepdims=True)
    spatial_sqrt=mx.sym.sqrt(spatial_norm)
    #spatial_mean=mx.sym.mean(spatial_sqrt, axis=(1,2,3), keepdims=True)
    spatial_mean=mx.sym.mean(spatial_sqrt)
    spatial_div_inverse=mx.sym.broadcast_div(spatial_mean, spatial_sqrt)

    spatial_attention_inverse=mx.symbol.tile(spatial_div_inverse, reps=(1,filters_in,1,1))   
    body=body*spatial_attention_inverse
    #body = mx.sym.broadcast_mul(body, spatial_div_inverse)

    fc1 = mx.sym.Pooling(body, kernel=(7, 7), global_pool=True, pool_type='avg')
    if num_classes<filters_in:
      fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn6w', cudnn_off=default.memonger)
      fc1 = mx.sym.FullyConnected(data=fc1, num_hidden=num_classes, name='pre_fc1')
    else:
      fc1 = mx.sym.Flatten(data=fc1)
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1', cudnn_off=default.memonger)
  elif fc_type=="GDC": #mobilefacenet_v1
    conv_6_dw = Linear(last_conv, num_filter=input_channel, num_group=input_channel, kernel=(7,7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")  
    #conv_6_dw = Linear(last_conv, num_filter=input_channel, num_group=input_channel, kernel=(4,7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")  
    conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=default.memonger)
  elif fc_type=='F':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
  elif fc_type=='G':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
  elif fc_type=='H':
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
  elif fc_type=='I':
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', cudnn_off=default.memonger)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=default.memonger)
  elif fc_type=='J':
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1', cudnn_off=default.memonger)
  return fc1

def residual_unit_v3(data, num_filter, stride, dim_match, name, **kwargs):
    
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit3')
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act1 = Act(data=bn2, act_type=config.net_act, name=name + '_relu1')
    conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return bn3 + shortcut

def residual_unit_v1l(data, num_filter, stride, dim_match, name, bottle_neck):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    workspace = config.workspace
    bn_mom = config.bn_mom
    memonger = default.memonger
    use_se = config.net_se
    act_type = config.net_act
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn2 = mx.symbol.broadcast_mul(bn2, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')

def antialiased_downsample(inputs, name, in_ch, fixed_param_names, pad_type='reflect', filt_size=3, stride=(2, 2), pad_off=0):
    pad_size = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
    pad_size = [x + pad_off for x in pad_size]

    def get_filter(filt_size, in_ch):
        if(filt_size==1):
            filter = np.array([1.,])
        elif(filt_size==2):
            filter = np.array([1., 1.])
        elif(filt_size==3):
            filter = np.array([1., 2., 1.])
        elif(filt_size==4):    
            filter = np.array([1., 3., 3., 1.])
        elif(filt_size==5):    
            filter = np.array([1., 4., 6., 4., 1.])
        elif(filt_size==6):    
            filter = np.array([1., 5., 10., 10., 5., 1.])
        elif(filt_size==7):    
            filter = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise NotImplementedError('invalid filter size %d' % filt_size)
        filter = filter[None,:]*filter[:,None]
        filter = filter/filter.sum()
        filter = filter[None,None,:,:]
        filter = np.tile(filter,(in_ch,1,1,1))
        
        filter = filter.astype(np.float32)
        return filter
    
    W_val = get_filter(filt_size, in_ch)
    # padding
    inputs = mx.sym.pad(inputs, mode=pad_type, pad_width=(0,)*4+tuple(pad_size), name=name+"_padding", constant_value=0)
    # downsample
    blurPoolW = mx.sym.Variable(name+"_BlurPool_weight", shape=W_val.shape, init=mx.init.Constant(W_val))
    mx.sym.BlockGrad(blurPoolW)
    fixed_param_names.append(name+"_BlurPool_weight")
    out = mx.sym.Convolution(data=inputs, weight=blurPoolW, bias=None, no_bias=True, kernel=(filt_size,filt_size), num_filter=in_ch, num_group=in_ch, stride=stride, name=name+"_BlurPool")
    
    return out

def get_loc(data, attr={'lr_mult': '0.01'}):
    """
    the localisation network in stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    ## 与gluon写法一致，只是调用的mx.symbol模块
    loc = mx.sym.Convolution(data=data, num_filter=24, kernel=(5, 5), stride=(1, 1), name="stn_loc_conv1", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn1", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max', name="stn_loc_pool1")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act1", lr_mult=config.stn_fc1_lr_mult)

    loc = mx.sym.Convolution(data=loc, num_filter=48, kernel=(3, 3), stride=(1, 1), name="stn_loc_conv2", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn2", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max', name="stn_loc_pool2")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act2", lr_mult=config.stn_fc1_lr_mult)

    loc = mx.sym.Convolution(data=loc, num_filter=96, kernel=(3, 3), stride=(1, 1), name="stn_loc_conv3", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn3", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max', name="stn_loc_pool3")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act3", lr_mult=config.stn_fc1_lr_mult)

    _weight1 = mx.symbol.Variable("stn_loc_fc1_weight", shape=(64, 12*12*96),
                                  lr_mult=config.stn_fc1_lr_mult, wd_mult=config.stn_fc1_wd_mult, init=mx.init.Normal(0.01))
    loc = mx.sym.FullyConnected(data=loc, weight=_weight1, no_bias=True, num_hidden=64, name="stn_loc_fc1")
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn4")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act4", lr_mult=config.stn_fc1_lr_mult)

    _weight2 = mx.symbol.Variable("stn_loc_fc2_weight", shape=(64, 64),
                                  lr_mult=config.stn_fc2_lr_mult, wd_mult=config.stn_fc2_wd_mult, init=mx.init.Normal(0.01))
    loc = mx.sym.FullyConnected(data=loc, weight=_weight2, no_bias=True, num_hidden=64, name="stn_loc_fc2")
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn5")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act5", lr_mult=config.stn_fc2_lr_mult)

    _weight3 = mx.symbol.Variable("stn_loc_fc3_weight", shape=(6, 64),
                                  lr_mult=config.stn_fc3_lr_mult, wd_mult=config.stn_fc3_wd_mult,
                                  init=mx.init.Normal(0.01))
    _bias3 = mx.symbol.Variable("stn_loc_fc3_bias", shape=(6, ),
                                lr_mult=config.stn_fc3_lr_mult, wd_mult=config.stn_fc3_wd_mult,
                                init=mx.init.Constant(mx.nd.array([1,0,0,0,1,0])))
    loc = mx.sym.FullyConnected(data=loc, weight=_weight3, bias=_bias3, num_hidden=6, name="stn_loc_fc3")
    return loc

def get_head(data, version_input, num_filter):
    bn_mom = config.bn_mom
    workspace = config.workspace
    kwargs = {'bn_mom': bn_mom, 'workspace' : workspace, 'memonger': default.memonger}
    data = data-127.5
    data = data*0.0078125
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if version_input==0:
      body = Conv(data=data, num_filter=num_filter, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type=config.net_act, name='relu0')
      body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
      body = data
      _num_filter = min(num_filter, 64)
      body = Conv(data=body, num_filter=_num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type=config.net_act, name='relu0')
      #body = residual_unit_v3(body, _num_filter, (2, 2), False, name='head', **kwargs)
      body = residual_unit_v1l(body, _num_filter, (2, 2), False, name='head', bottle_neck=False)
    return body


