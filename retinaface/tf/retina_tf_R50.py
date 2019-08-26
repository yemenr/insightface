import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    data            = tf.placeholder(tf.float32, shape = (None, None, None, 3), name = 'data')
    bn_data         = batch_normalization(data, variance_epsilon=1.9999999494757503e-05, name='bn_data')
    conv0_pad       = tf.pad(bn_data, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]])
    conv0           = convolution(conv0_pad, group=1, strides=[2, 2], padding='VALID', name='conv0')
    bn0             = batch_normalization(conv0, variance_epsilon=1.9999999494757503e-05, name='bn0')
    relu0           = tf.nn.relu(bn0, name = 'relu0')
    pooling0_pad    = tf.pad(relu0, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=float('-Inf'))
    pooling0        = tf.nn.max_pool(pooling0_pad, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pooling0')
    stage1_unit1_bn1 = batch_normalization(pooling0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn1')
    stage1_unit1_relu1 = tf.nn.relu(stage1_unit1_bn1, name = 'stage1_unit1_relu1')
    stage1_unit1_conv1 = convolution(stage1_unit1_relu1, group=1, strides=[1, 1], padding='VALID', name='stage1_unit1_conv1')
    stage1_unit1_sc = convolution(stage1_unit1_relu1, group=1, strides=[1, 1], padding='VALID', name='stage1_unit1_sc')
    stage1_unit1_bn2 = batch_normalization(stage1_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn2')
    stage1_unit1_relu2 = tf.nn.relu(stage1_unit1_bn2, name = 'stage1_unit1_relu2')
    stage1_unit1_conv2_pad = tf.pad(stage1_unit1_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit1_conv2 = convolution(stage1_unit1_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit1_conv2')
    stage1_unit1_bn3 = batch_normalization(stage1_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn3')
    stage1_unit1_relu3 = tf.nn.relu(stage1_unit1_bn3, name = 'stage1_unit1_relu3')
    stage1_unit1_conv3 = convolution(stage1_unit1_relu3, group=1, strides=[1, 1], padding='VALID', name='stage1_unit1_conv3')
    plus0           = stage1_unit1_conv3 + stage1_unit1_sc
    stage1_unit2_bn1 = batch_normalization(plus0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn1')
    stage1_unit2_relu1 = tf.nn.relu(stage1_unit2_bn1, name = 'stage1_unit2_relu1')
    stage1_unit2_conv1 = convolution(stage1_unit2_relu1, group=1, strides=[1, 1], padding='VALID', name='stage1_unit2_conv1')
    stage1_unit2_bn2 = batch_normalization(stage1_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn2')
    stage1_unit2_relu2 = tf.nn.relu(stage1_unit2_bn2, name = 'stage1_unit2_relu2')
    stage1_unit2_conv2_pad = tf.pad(stage1_unit2_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit2_conv2 = convolution(stage1_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit2_conv2')
    stage1_unit2_bn3 = batch_normalization(stage1_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn3')
    stage1_unit2_relu3 = tf.nn.relu(stage1_unit2_bn3, name = 'stage1_unit2_relu3')
    stage1_unit2_conv3 = convolution(stage1_unit2_relu3, group=1, strides=[1, 1], padding='VALID', name='stage1_unit2_conv3')
    plus1           = stage1_unit2_conv3 + plus0
    stage1_unit3_bn1 = batch_normalization(plus1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn1')
    stage1_unit3_relu1 = tf.nn.relu(stage1_unit3_bn1, name = 'stage1_unit3_relu1')
    stage1_unit3_conv1 = convolution(stage1_unit3_relu1, group=1, strides=[1, 1], padding='VALID', name='stage1_unit3_conv1')
    stage1_unit3_bn2 = batch_normalization(stage1_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn2')
    stage1_unit3_relu2 = tf.nn.relu(stage1_unit3_bn2, name = 'stage1_unit3_relu2')
    stage1_unit3_conv2_pad = tf.pad(stage1_unit3_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit3_conv2 = convolution(stage1_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit3_conv2')
    stage1_unit3_bn3 = batch_normalization(stage1_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn3')
    stage1_unit3_relu3 = tf.nn.relu(stage1_unit3_bn3, name = 'stage1_unit3_relu3')
    stage1_unit3_conv3 = convolution(stage1_unit3_relu3, group=1, strides=[1, 1], padding='VALID', name='stage1_unit3_conv3')
    plus2           = stage1_unit3_conv3 + plus1
    stage2_unit1_bn1 = batch_normalization(plus2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn1')
    stage2_unit1_relu1 = tf.nn.relu(stage2_unit1_bn1, name = 'stage2_unit1_relu1')
    stage2_unit1_conv1 = convolution(stage2_unit1_relu1, group=1, strides=[1, 1], padding='VALID', name='stage2_unit1_conv1')
    stage2_unit1_sc = convolution(stage2_unit1_relu1, group=1, strides=[2, 2], padding='VALID', name='stage2_unit1_sc')
    stage2_unit1_bn2 = batch_normalization(stage2_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn2')
    stage2_unit1_relu2 = tf.nn.relu(stage2_unit1_bn2, name = 'stage2_unit1_relu2')
    stage2_unit1_conv2_pad = tf.pad(stage2_unit1_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit1_conv2 = convolution(stage2_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage2_unit1_conv2')
    stage2_unit1_bn3 = batch_normalization(stage2_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn3')
    stage2_unit1_relu3 = tf.nn.relu(stage2_unit1_bn3, name = 'stage2_unit1_relu3')
    stage2_unit1_conv3 = convolution(stage2_unit1_relu3, group=1, strides=[1, 1], padding='VALID', name='stage2_unit1_conv3')
    plus3           = stage2_unit1_conv3 + stage2_unit1_sc
    stage2_unit2_bn1 = batch_normalization(plus3, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn1')
    stage2_unit2_relu1 = tf.nn.relu(stage2_unit2_bn1, name = 'stage2_unit2_relu1')
    stage2_unit2_conv1 = convolution(stage2_unit2_relu1, group=1, strides=[1, 1], padding='VALID', name='stage2_unit2_conv1')
    stage2_unit2_bn2 = batch_normalization(stage2_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn2')
    stage2_unit2_relu2 = tf.nn.relu(stage2_unit2_bn2, name = 'stage2_unit2_relu2')
    stage2_unit2_conv2_pad = tf.pad(stage2_unit2_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit2_conv2 = convolution(stage2_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit2_conv2')
    stage2_unit2_bn3 = batch_normalization(stage2_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn3')
    stage2_unit2_relu3 = tf.nn.relu(stage2_unit2_bn3, name = 'stage2_unit2_relu3')
    stage2_unit2_conv3 = convolution(stage2_unit2_relu3, group=1, strides=[1, 1], padding='VALID', name='stage2_unit2_conv3')
    plus4           = stage2_unit2_conv3 + plus3
    stage2_unit3_bn1 = batch_normalization(plus4, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn1')
    stage2_unit3_relu1 = tf.nn.relu(stage2_unit3_bn1, name = 'stage2_unit3_relu1')
    stage2_unit3_conv1 = convolution(stage2_unit3_relu1, group=1, strides=[1, 1], padding='VALID', name='stage2_unit3_conv1')
    stage2_unit3_bn2 = batch_normalization(stage2_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn2')
    stage2_unit3_relu2 = tf.nn.relu(stage2_unit3_bn2, name = 'stage2_unit3_relu2')
    stage2_unit3_conv2_pad = tf.pad(stage2_unit3_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit3_conv2 = convolution(stage2_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit3_conv2')
    stage2_unit3_bn3 = batch_normalization(stage2_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn3')
    stage2_unit3_relu3 = tf.nn.relu(stage2_unit3_bn3, name = 'stage2_unit3_relu3')
    stage2_unit3_conv3 = convolution(stage2_unit3_relu3, group=1, strides=[1, 1], padding='VALID', name='stage2_unit3_conv3')
    plus5           = stage2_unit3_conv3 + plus4
    stage2_unit4_bn1 = batch_normalization(plus5, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn1')
    stage2_unit4_relu1 = tf.nn.relu(stage2_unit4_bn1, name = 'stage2_unit4_relu1')
    stage2_unit4_conv1 = convolution(stage2_unit4_relu1, group=1, strides=[1, 1], padding='VALID', name='stage2_unit4_conv1')
    stage2_unit4_bn2 = batch_normalization(stage2_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn2')
    stage2_unit4_relu2 = tf.nn.relu(stage2_unit4_bn2, name = 'stage2_unit4_relu2')
    stage2_unit4_conv2_pad = tf.pad(stage2_unit4_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit4_conv2 = convolution(stage2_unit4_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit4_conv2')
    stage2_unit4_bn3 = batch_normalization(stage2_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn3')
    stage2_unit4_relu3 = tf.nn.relu(stage2_unit4_bn3, name = 'stage2_unit4_relu3')
    stage2_unit4_conv3 = convolution(stage2_unit4_relu3, group=1, strides=[1, 1], padding='VALID', name='stage2_unit4_conv3')
    plus6           = stage2_unit4_conv3 + plus5
    stage3_unit1_bn1 = batch_normalization(plus6, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn1')
    stage3_unit1_relu1 = tf.nn.relu(stage3_unit1_bn1, name = 'stage3_unit1_relu1')
    stage3_unit1_conv1 = convolution(stage3_unit1_relu1, group=1, strides=[1, 1], padding='VALID', name='stage3_unit1_conv1')
    stage3_unit1_sc = convolution(stage3_unit1_relu1, group=1, strides=[2, 2], padding='VALID', name='stage3_unit1_sc')
    stage3_unit1_bn2 = batch_normalization(stage3_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn2')
    stage3_unit1_relu2 = tf.nn.relu(stage3_unit1_bn2, name = 'stage3_unit1_relu2')
    stage3_unit1_conv2_pad = tf.pad(stage3_unit1_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit1_conv2 = convolution(stage3_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage3_unit1_conv2')
    ssh_m1_red_conv = convolution(stage3_unit1_relu2, group=1, strides=[1, 1], padding='VALID', name='ssh_m1_red_conv')
    stage3_unit1_bn3 = batch_normalization(stage3_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn3')
    ssh_m1_red_conv_bn = batch_normalization(ssh_m1_red_conv, variance_epsilon=1.9999999494757503e-05, name='ssh_m1_red_conv_bn')
    stage3_unit1_relu3 = tf.nn.relu(stage3_unit1_bn3, name = 'stage3_unit1_relu3')
    ssh_m1_red_conv_relu = tf.nn.relu(ssh_m1_red_conv_bn, name = 'ssh_m1_red_conv_relu')
    stage3_unit1_conv3 = convolution(stage3_unit1_relu3, group=1, strides=[1, 1], padding='VALID', name='stage3_unit1_conv3')
    plus7           = stage3_unit1_conv3 + stage3_unit1_sc
    stage3_unit2_bn1 = batch_normalization(plus7, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn1')
    stage3_unit2_relu1 = tf.nn.relu(stage3_unit2_bn1, name = 'stage3_unit2_relu1')
    stage3_unit2_conv1 = convolution(stage3_unit2_relu1, group=1, strides=[1, 1], padding='VALID', name='stage3_unit2_conv1')
    stage3_unit2_bn2 = batch_normalization(stage3_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn2')
    stage3_unit2_relu2 = tf.nn.relu(stage3_unit2_bn2, name = 'stage3_unit2_relu2')
    stage3_unit2_conv2_pad = tf.pad(stage3_unit2_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit2_conv2 = convolution(stage3_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit2_conv2')
    stage3_unit2_bn3 = batch_normalization(stage3_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn3')
    stage3_unit2_relu3 = tf.nn.relu(stage3_unit2_bn3, name = 'stage3_unit2_relu3')
    stage3_unit2_conv3 = convolution(stage3_unit2_relu3, group=1, strides=[1, 1], padding='VALID', name='stage3_unit2_conv3')
    plus8           = stage3_unit2_conv3 + plus7
    stage3_unit3_bn1 = batch_normalization(plus8, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn1')
    stage3_unit3_relu1 = tf.nn.relu(stage3_unit3_bn1, name = 'stage3_unit3_relu1')
    stage3_unit3_conv1 = convolution(stage3_unit3_relu1, group=1, strides=[1, 1], padding='VALID', name='stage3_unit3_conv1')
    stage3_unit3_bn2 = batch_normalization(stage3_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn2')
    stage3_unit3_relu2 = tf.nn.relu(stage3_unit3_bn2, name = 'stage3_unit3_relu2')
    stage3_unit3_conv2_pad = tf.pad(stage3_unit3_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit3_conv2 = convolution(stage3_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit3_conv2')
    stage3_unit3_bn3 = batch_normalization(stage3_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn3')
    stage3_unit3_relu3 = tf.nn.relu(stage3_unit3_bn3, name = 'stage3_unit3_relu3')
    stage3_unit3_conv3 = convolution(stage3_unit3_relu3, group=1, strides=[1, 1], padding='VALID', name='stage3_unit3_conv3')
    plus9           = stage3_unit3_conv3 + plus8
    stage3_unit4_bn1 = batch_normalization(plus9, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn1')
    stage3_unit4_relu1 = tf.nn.relu(stage3_unit4_bn1, name = 'stage3_unit4_relu1')
    stage3_unit4_conv1 = convolution(stage3_unit4_relu1, group=1, strides=[1, 1], padding='VALID', name='stage3_unit4_conv1')
    stage3_unit4_bn2 = batch_normalization(stage3_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn2')
    stage3_unit4_relu2 = tf.nn.relu(stage3_unit4_bn2, name = 'stage3_unit4_relu2')
    stage3_unit4_conv2_pad = tf.pad(stage3_unit4_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit4_conv2 = convolution(stage3_unit4_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit4_conv2')
    stage3_unit4_bn3 = batch_normalization(stage3_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn3')
    stage3_unit4_relu3 = tf.nn.relu(stage3_unit4_bn3, name = 'stage3_unit4_relu3')
    stage3_unit4_conv3 = convolution(stage3_unit4_relu3, group=1, strides=[1, 1], padding='VALID', name='stage3_unit4_conv3')
    plus10          = stage3_unit4_conv3 + plus9
    stage3_unit5_bn1 = batch_normalization(plus10, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn1')
    stage3_unit5_relu1 = tf.nn.relu(stage3_unit5_bn1, name = 'stage3_unit5_relu1')
    stage3_unit5_conv1 = convolution(stage3_unit5_relu1, group=1, strides=[1, 1], padding='VALID', name='stage3_unit5_conv1')
    stage3_unit5_bn2 = batch_normalization(stage3_unit5_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn2')
    stage3_unit5_relu2 = tf.nn.relu(stage3_unit5_bn2, name = 'stage3_unit5_relu2')
    stage3_unit5_conv2_pad = tf.pad(stage3_unit5_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit5_conv2 = convolution(stage3_unit5_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit5_conv2')
    stage3_unit5_bn3 = batch_normalization(stage3_unit5_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn3')
    stage3_unit5_relu3 = tf.nn.relu(stage3_unit5_bn3, name = 'stage3_unit5_relu3')
    stage3_unit5_conv3 = convolution(stage3_unit5_relu3, group=1, strides=[1, 1], padding='VALID', name='stage3_unit5_conv3')
    plus11          = stage3_unit5_conv3 + plus10
    stage3_unit6_bn1 = batch_normalization(plus11, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn1')
    stage3_unit6_relu1 = tf.nn.relu(stage3_unit6_bn1, name = 'stage3_unit6_relu1')
    stage3_unit6_conv1 = convolution(stage3_unit6_relu1, group=1, strides=[1, 1], padding='VALID', name='stage3_unit6_conv1')
    stage3_unit6_bn2 = batch_normalization(stage3_unit6_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn2')
    stage3_unit6_relu2 = tf.nn.relu(stage3_unit6_bn2, name = 'stage3_unit6_relu2')
    stage3_unit6_conv2_pad = tf.pad(stage3_unit6_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit6_conv2 = convolution(stage3_unit6_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit6_conv2')
    stage3_unit6_bn3 = batch_normalization(stage3_unit6_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn3')
    stage3_unit6_relu3 = tf.nn.relu(stage3_unit6_bn3, name = 'stage3_unit6_relu3')
    stage3_unit6_conv3 = convolution(stage3_unit6_relu3, group=1, strides=[1, 1], padding='VALID', name='stage3_unit6_conv3')
    plus12          = stage3_unit6_conv3 + plus11
    stage4_unit1_bn1 = batch_normalization(plus12, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn1')
    stage4_unit1_relu1 = tf.nn.relu(stage4_unit1_bn1, name = 'stage4_unit1_relu1')
    stage4_unit1_conv1 = convolution(stage4_unit1_relu1, group=1, strides=[1, 1], padding='VALID', name='stage4_unit1_conv1')
    stage4_unit1_sc = convolution(stage4_unit1_relu1, group=1, strides=[2, 2], padding='VALID', name='stage4_unit1_sc')
    stage4_unit1_bn2 = batch_normalization(stage4_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn2')
    stage4_unit1_relu2 = tf.nn.relu(stage4_unit1_bn2, name = 'stage4_unit1_relu2')
    stage4_unit1_conv2_pad = tf.pad(stage4_unit1_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit1_conv2 = convolution(stage4_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage4_unit1_conv2')
    ssh_c2_lateral  = convolution(stage4_unit1_relu2, group=1, strides=[1, 1], padding='VALID', name='ssh_c2_lateral')
    stage4_unit1_bn3 = batch_normalization(stage4_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn3')
    ssh_c2_lateral_bn = batch_normalization(ssh_c2_lateral, variance_epsilon=1.9999999494757503e-05, name='ssh_c2_lateral_bn')
    stage4_unit1_relu3 = tf.nn.relu(stage4_unit1_bn3, name = 'stage4_unit1_relu3')
    ssh_c2_lateral_relu = tf.nn.relu(ssh_c2_lateral_bn, name = 'ssh_c2_lateral_relu')
    stage4_unit1_conv3 = convolution(stage4_unit1_relu3, group=1, strides=[1, 1], padding='VALID', name='stage4_unit1_conv3')
    plus13          = stage4_unit1_conv3 + stage4_unit1_sc
    stage4_unit2_bn1 = batch_normalization(plus13, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn1')
    stage4_unit2_relu1 = tf.nn.relu(stage4_unit2_bn1, name = 'stage4_unit2_relu1')
    stage4_unit2_conv1 = convolution(stage4_unit2_relu1, group=1, strides=[1, 1], padding='VALID', name='stage4_unit2_conv1')
    stage4_unit2_bn2 = batch_normalization(stage4_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn2')
    stage4_unit2_relu2 = tf.nn.relu(stage4_unit2_bn2, name = 'stage4_unit2_relu2')
    stage4_unit2_conv2_pad = tf.pad(stage4_unit2_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit2_conv2 = convolution(stage4_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit2_conv2')
    stage4_unit2_bn3 = batch_normalization(stage4_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn3')
    stage4_unit2_relu3 = tf.nn.relu(stage4_unit2_bn3, name = 'stage4_unit2_relu3')
    stage4_unit2_conv3 = convolution(stage4_unit2_relu3, group=1, strides=[1, 1], padding='VALID', name='stage4_unit2_conv3')
    plus14          = stage4_unit2_conv3 + plus13
    stage4_unit3_bn1 = batch_normalization(plus14, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn1')
    stage4_unit3_relu1 = tf.nn.relu(stage4_unit3_bn1, name = 'stage4_unit3_relu1')
    stage4_unit3_conv1 = convolution(stage4_unit3_relu1, group=1, strides=[1, 1], padding='VALID', name='stage4_unit3_conv1')
    stage4_unit3_bn2 = batch_normalization(stage4_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn2')
    stage4_unit3_relu2 = tf.nn.relu(stage4_unit3_bn2, name = 'stage4_unit3_relu2')
    stage4_unit3_conv2_pad = tf.pad(stage4_unit3_relu2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit3_conv2 = convolution(stage4_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit3_conv2')
    stage4_unit3_bn3 = batch_normalization(stage4_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn3')
    stage4_unit3_relu3 = tf.nn.relu(stage4_unit3_bn3, name = 'stage4_unit3_relu3')
    stage4_unit3_conv3 = convolution(stage4_unit3_relu3, group=1, strides=[1, 1], padding='VALID', name='stage4_unit3_conv3')
    plus15          = stage4_unit3_conv3 + plus14
    bn1             = batch_normalization(plus15, variance_epsilon=1.9999999494757503e-05, name='bn1')
    relu1           = tf.nn.relu(bn1, name = 'relu1')
    ssh_c3_lateral  = convolution(relu1, group=1, strides=[1, 1], padding='VALID', name='ssh_c3_lateral')
    ssh_c3_lateral_bn = batch_normalization(ssh_c3_lateral, variance_epsilon=1.9999999494757503e-05, name='ssh_c3_lateral_bn')
    ssh_c3_lateral_relu = tf.nn.relu(ssh_c3_lateral_bn, name = 'ssh_c3_lateral_relu')
    ssh_m3_det_conv1_pad = tf.pad(ssh_c3_lateral_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m3_det_conv1 = convolution(ssh_m3_det_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m3_det_conv1')
    ssh_m3_det_context_conv1_pad = tf.pad(ssh_c3_lateral_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m3_det_context_conv1 = convolution(ssh_m3_det_context_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m3_det_context_conv1')
    ssh_c3_up       = tf.keras.layers.UpSampling2D(size=(2, 2))(ssh_c3_lateral_relu)
    ssh_m3_det_conv1_bn = batch_normalization(ssh_m3_det_conv1, variance_epsilon=1.9999999494757503e-05, name='ssh_m3_det_conv1_bn')
    ssh_m3_det_context_conv1_bn = batch_normalization(ssh_m3_det_context_conv1, variance_epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv1_bn')
    crop0           = tf.strided_slice(ssh_c3_up, [0, 0, 0, 0], tf.shape(ssh_c2_lateral_relu), [1, 1, 1, 1] , name='crop0')
    ssh_m3_det_context_conv1_relu = tf.nn.relu(ssh_m3_det_context_conv1_bn, name = 'ssh_m3_det_context_conv1_relu')
    plus0_1         = ssh_c2_lateral_relu + crop0
    ssh_m3_det_context_conv2_pad = tf.pad(ssh_m3_det_context_conv1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m3_det_context_conv2 = convolution(ssh_m3_det_context_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m3_det_context_conv2')
    ssh_m3_det_context_conv3_1_pad = tf.pad(ssh_m3_det_context_conv1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m3_det_context_conv3_1 = convolution(ssh_m3_det_context_conv3_1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m3_det_context_conv3_1')
    ssh_c2_aggr_pad = tf.pad(plus0_1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_c2_aggr     = convolution(ssh_c2_aggr_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_c2_aggr')
    ssh_m3_det_context_conv2_bn = batch_normalization(ssh_m3_det_context_conv2, variance_epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv2_bn')
    ssh_m3_det_context_conv3_1_bn = batch_normalization(ssh_m3_det_context_conv3_1, variance_epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv3_1_bn')
    ssh_c2_aggr_bn  = batch_normalization(ssh_c2_aggr, variance_epsilon=1.9999999494757503e-05, name='ssh_c2_aggr_bn')
    ssh_m3_det_context_conv3_1_relu = tf.nn.relu(ssh_m3_det_context_conv3_1_bn, name = 'ssh_m3_det_context_conv3_1_relu')
    ssh_c2_aggr_relu = tf.nn.relu(ssh_c2_aggr_bn, name = 'ssh_c2_aggr_relu')
    ssh_m3_det_context_conv3_2_pad = tf.pad(ssh_m3_det_context_conv3_1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m3_det_context_conv3_2 = convolution(ssh_m3_det_context_conv3_2_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m3_det_context_conv3_2')
    ssh_m2_det_conv1_pad = tf.pad(ssh_c2_aggr_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m2_det_conv1 = convolution(ssh_m2_det_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m2_det_conv1')
    ssh_m2_det_context_conv1_pad = tf.pad(ssh_c2_aggr_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m2_det_context_conv1 = convolution(ssh_m2_det_context_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m2_det_context_conv1')
    ssh_m2_red_up   = tf.keras.layers.UpSampling2D(size=(2, 2))(ssh_c2_aggr_relu)
    ssh_m3_det_context_conv3_2_bn = batch_normalization(ssh_m3_det_context_conv3_2, variance_epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv3_2_bn')
    ssh_m2_det_conv1_bn = batch_normalization(ssh_m2_det_conv1, variance_epsilon=1.9999999494757503e-05, name='ssh_m2_det_conv1_bn')
    ssh_m2_det_context_conv1_bn = batch_normalization(ssh_m2_det_context_conv1, variance_epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv1_bn')
    crop1           = tf.strided_slice(ssh_m2_red_up, [0, 0, 0, 0], tf.shape(ssh_m1_red_conv_relu), [1, 1, 1, 1] , name='crop1')
    ssh_m3_det_concat = tf.concat([ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn], 3, name = 'ssh_m3_det_concat')
    ssh_m2_det_context_conv1_relu = tf.nn.relu(ssh_m2_det_context_conv1_bn, name = 'ssh_m2_det_context_conv1_relu')
    plus1_1         = ssh_m1_red_conv_relu + crop1
    ssh_m3_det_concat_relu = tf.nn.relu(ssh_m3_det_concat, name = 'ssh_m3_det_concat_relu')
    ssh_m2_det_context_conv2_pad = tf.pad(ssh_m2_det_context_conv1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m2_det_context_conv2 = convolution(ssh_m2_det_context_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m2_det_context_conv2')
    ssh_m2_det_context_conv3_1_pad = tf.pad(ssh_m2_det_context_conv1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m2_det_context_conv3_1 = convolution(ssh_m2_det_context_conv3_1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m2_det_context_conv3_1')
    ssh_c1_aggr_pad = tf.pad(plus1_1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_c1_aggr     = convolution(ssh_c1_aggr_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_c1_aggr')
    face_rpn_cls_score_stride32 = convolution(ssh_m3_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_cls_score_stride32')
    face_rpn_bbox_pred_stride32 = convolution(ssh_m3_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_bbox_pred_stride32')
    face_rpn_landmark_pred_stride32 = convolution(ssh_m3_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_landmark_pred_stride32')
    ssh_m2_det_context_conv2_bn = batch_normalization(ssh_m2_det_context_conv2, variance_epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv2_bn')
    ssh_m2_det_context_conv3_1_bn = batch_normalization(ssh_m2_det_context_conv3_1, variance_epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv3_1_bn')
    ssh_c1_aggr_bn  = batch_normalization(ssh_c1_aggr, variance_epsilon=1.9999999494757503e-05, name='ssh_c1_aggr_bn')
    face_rpn_cls_score_reshape_stride32 = tf.transpose(face_rpn_cls_score_stride32, perm=[0,3,1,2])
    face_rpn_cls_score_reshape_stride32 = tf.reshape(face_rpn_cls_score_reshape_stride32, [tf.shape(face_rpn_cls_score_stride32)[0], 2, -1, tf.shape(face_rpn_cls_score_stride32)[2]])
    face_rpn_cls_score_reshape_stride32 = tf.transpose(face_rpn_cls_score_reshape_stride32, perm=[0,2,3,1], name='face_rpn_cls_score_reshape_stride32')
    ssh_m2_det_context_conv3_1_relu = tf.nn.relu(ssh_m2_det_context_conv3_1_bn, name = 'ssh_m2_det_context_conv3_1_relu')
    ssh_c1_aggr_relu = tf.nn.relu(ssh_c1_aggr_bn, name = 'ssh_c1_aggr_relu')
    face_rpn_cls_prob_stride32 = tf.nn.softmax(face_rpn_cls_score_reshape_stride32, name = 'face_rpn_cls_prob_stride32')
    ssh_m2_det_context_conv3_2_pad = tf.pad(ssh_m2_det_context_conv3_1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m2_det_context_conv3_2 = convolution(ssh_m2_det_context_conv3_2_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m2_det_context_conv3_2')
    ssh_m1_det_conv1_pad = tf.pad(ssh_c1_aggr_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m1_det_conv1 = convolution(ssh_m1_det_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m1_det_conv1')
    ssh_m1_det_context_conv1_pad = tf.pad(ssh_c1_aggr_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m1_det_context_conv1 = convolution(ssh_m1_det_context_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m1_det_context_conv1')
    face_rpn_cls_prob_reshape_stride32 = tf.transpose(face_rpn_cls_prob_stride32, perm=[0,3,1,2])
    face_rpn_cls_prob_reshape_stride32 = tf.reshape(face_rpn_cls_prob_reshape_stride32, [tf.shape(face_rpn_cls_prob_stride32)[0], 4, -1, tf.shape(face_rpn_cls_prob_stride32)[2]])
    face_rpn_cls_prob_reshape_stride32 = tf.transpose(face_rpn_cls_prob_reshape_stride32, perm=[0,2,3,1], name='face_rpn_cls_prob_reshape_stride32')
    ssh_m2_det_context_conv3_2_bn = batch_normalization(ssh_m2_det_context_conv3_2, variance_epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv3_2_bn')
    ssh_m1_det_conv1_bn = batch_normalization(ssh_m1_det_conv1, variance_epsilon=1.9999999494757503e-05, name='ssh_m1_det_conv1_bn')
    ssh_m1_det_context_conv1_bn = batch_normalization(ssh_m1_det_context_conv1, variance_epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv1_bn')
    ssh_m2_det_concat = tf.concat([ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn], 3, name = 'ssh_m2_det_concat')
    ssh_m1_det_context_conv1_relu = tf.nn.relu(ssh_m1_det_context_conv1_bn, name = 'ssh_m1_det_context_conv1_relu')
    ssh_m2_det_concat_relu = tf.nn.relu(ssh_m2_det_concat, name = 'ssh_m2_det_concat_relu')
    ssh_m1_det_context_conv2_pad = tf.pad(ssh_m1_det_context_conv1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m1_det_context_conv2 = convolution(ssh_m1_det_context_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m1_det_context_conv2')
    ssh_m1_det_context_conv3_1_pad = tf.pad(ssh_m1_det_context_conv1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m1_det_context_conv3_1 = convolution(ssh_m1_det_context_conv3_1_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m1_det_context_conv3_1')
    face_rpn_cls_score_stride16 = convolution(ssh_m2_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_cls_score_stride16')
    face_rpn_bbox_pred_stride16 = convolution(ssh_m2_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_bbox_pred_stride16')
    face_rpn_landmark_pred_stride16 = convolution(ssh_m2_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_landmark_pred_stride16')
    ssh_m1_det_context_conv2_bn = batch_normalization(ssh_m1_det_context_conv2, variance_epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv2_bn')
    ssh_m1_det_context_conv3_1_bn = batch_normalization(ssh_m1_det_context_conv3_1, variance_epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv3_1_bn')
    face_rpn_cls_score_reshape_stride16 = tf.transpose(face_rpn_cls_score_stride16, perm=[0,3,1,2])
    face_rpn_cls_score_reshape_stride16 = tf.reshape(face_rpn_cls_score_reshape_stride16, [tf.shape(face_rpn_cls_score_stride16)[0], 2, -1, tf.shape(face_rpn_cls_score_stride16)[2]])
    face_rpn_cls_score_reshape_stride16 = tf.transpose(face_rpn_cls_score_reshape_stride16, perm=[0,2,3,1], name='face_rpn_cls_score_reshape_stride16')
    ssh_m1_det_context_conv3_1_relu = tf.nn.relu(ssh_m1_det_context_conv3_1_bn, name = 'ssh_m1_det_context_conv3_1_relu')
    face_rpn_cls_prob_stride16 = tf.nn.softmax(face_rpn_cls_score_reshape_stride16, name = 'face_rpn_cls_prob_stride16')
    ssh_m1_det_context_conv3_2_pad = tf.pad(ssh_m1_det_context_conv3_1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    ssh_m1_det_context_conv3_2 = convolution(ssh_m1_det_context_conv3_2_pad, group=1, strides=[1, 1], padding='VALID', name='ssh_m1_det_context_conv3_2')
    face_rpn_cls_prob_reshape_stride16 = tf.transpose(face_rpn_cls_prob_stride16, perm=[0,3,1,2])
    face_rpn_cls_prob_reshape_stride16 = tf.reshape(face_rpn_cls_prob_reshape_stride16, [tf.shape(face_rpn_cls_prob_stride16)[0], 4, -1, tf.shape(face_rpn_cls_prob_stride16)[2]])
    face_rpn_cls_prob_reshape_stride16 = tf.transpose(face_rpn_cls_prob_reshape_stride16, perm=[0,2,3,1], name='face_rpn_cls_prob_reshape_stride16')
    ssh_m1_det_context_conv3_2_bn = batch_normalization(ssh_m1_det_context_conv3_2, variance_epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv3_2_bn')
    ssh_m1_det_concat = tf.concat([ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn], 3, name = 'ssh_m1_det_concat')
    ssh_m1_det_concat_relu = tf.nn.relu(ssh_m1_det_concat, name = 'ssh_m1_det_concat_relu')
    face_rpn_cls_score_stride8 = convolution(ssh_m1_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_cls_score_stride8')
    face_rpn_bbox_pred_stride8 = convolution(ssh_m1_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_bbox_pred_stride8')
    face_rpn_landmark_pred_stride8 = convolution(ssh_m1_det_concat_relu, group=1, strides=[1, 1], padding='VALID', name='face_rpn_landmark_pred_stride8')
    face_rpn_cls_score_reshape_stride8 = tf.transpose(face_rpn_cls_score_stride8, perm=[0,3,1,2])
    face_rpn_cls_score_reshape_stride8 = tf.reshape(face_rpn_cls_score_reshape_stride8, [tf.shape(face_rpn_cls_score_stride8)[0], 2, -1, tf.shape(face_rpn_cls_score_stride8)[2]])
    face_rpn_cls_score_reshape_stride8 = tf.transpose(face_rpn_cls_score_reshape_stride8, perm=[0,2,3,1], name='face_rpn_cls_score_reshape_stride8')
    face_rpn_cls_prob_stride8 = tf.nn.softmax(face_rpn_cls_score_reshape_stride8, name = 'face_rpn_cls_prob_stride8')
    face_rpn_cls_prob_reshape_stride8 = tf.transpose(face_rpn_cls_prob_stride8, perm=[0,3,1,2])
    face_rpn_cls_prob_reshape_stride8 = tf.reshape(face_rpn_cls_prob_reshape_stride8, [tf.shape(face_rpn_cls_prob_stride8)[0], 4, -1, tf.shape(face_rpn_cls_prob_stride8)[2]])
    face_rpn_cls_prob_reshape_stride8 = tf.transpose(face_rpn_cls_prob_reshape_stride8, perm=[0,2,3,1], name='face_rpn_cls_prob_reshape_stride8')
    return [data], [face_rpn_cls_prob_reshape_stride32, face_rpn_bbox_pred_stride32, face_rpn_landmark_pred_stride32, face_rpn_cls_prob_reshape_stride16, face_rpn_bbox_pred_stride16, face_rpn_landmark_pred_stride16, face_rpn_cls_prob_reshape_stride8, face_rpn_bbox_pred_stride8, face_rpn_landmark_pred_stride8]

def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    layer = tf.identity(layer, name=name)
    return layer

