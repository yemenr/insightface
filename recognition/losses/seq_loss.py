import os

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
#os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx

class GitLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, id_num_class, id_num_class, git_params, scale=1.0):
        if not len(shapes[0]) == 2:
            raise ValueError('dim for input_data shoudl be 2 for GitLoss')

        self.batch_size = shapes[0][0]
        self.scale = scale
        self.id_num_class = id_num_class
        self.seq_num_class = seq_num_class
        self.center_alpha = float(git_params[0])
        self.git_alpha = float(git_params[1])
        self.git_beta = float(git_params[2])
        self.git_p = float(git_params[3]) 

    def forward(self, is_train, req, in_data, out_data, aux):
        labels = in_data[1].asnumpy()
        diff = aux[0]
        center = aux[1]

        # calculate part of center loss 
        ## store derivative x_i - c_yi
        for i in range(self.batch_size):
            diff[i] = in_data[0][i] - center[int(labels[i])]
        center_loss_part = mx.nd.sum(mx.nd.square(diff)) / 2
        diff *= self.git_alpha
        
        # calculate part of git loss
        id_inter_cnt = int((self.id_num_class + self.seq_num_class) * self.git_p)
        seq_inter_cnt = int(self.id_num_class * self.git_p)
        id_label_selected = [x for x in range(self.id_num_class + self.seq_num_class)]
        random.shuffle(id_label_selected)
        id_label_selected = id_label_selected[:id_inter_cnt]
        seq_label_selected = [x for x in range(self.id_num_class)]
        random.shuffle(seq_label_selected)
        seq_label_selected = seq_label_selected[:seq_inter_cnt]        
        ## store derivative -2*(x_i - c_yj) / (1 + (x_i - c_yj)^2)^2
        inter_loss_part = 0        
        for i in range(self.batch_size):
            x_i = in_data[0][i]                
            inter_loss_part_i = 0
            inter_diff_part_i = 0
            inter_cnt = 0
            label_selected = None
            ### id part of inter diff
            if i < self.batch_size/2:                
                inter_cnt = id_inter_cnt
                label_selected = id_label_selected                    
            ### seq part of inter diff
            else:
                inter_cnt = seq_inter_cnt
                label_selected = seq_label_selected
            for k in range(inter_cnt):
                c_yj = center[label_selected[k]]
                inter_loss_part_i += mx.nd.sum(1 / (1 + mx.nd.square(x_i - c_yj)))
                inter_diff_part_i += -2 * (x_i - c_yj) / mx.nd.square(1 + mx.nd.square(x_i - c_yj))
            inter_loss_part += inter_loss_part_i / inter_cnt
            diff[i] += self.git_beta * inter_diff_part_i / inter_cnt
        
        loss = (self.git_alpha*center_loss_part + self.git_beta*inter_loss_part) / self.batch_size 
        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        diff = aux[0]
        center = aux[1]
        sum_ = aux[2]

        # back grad is just scale * ( x_i - c_yi)
        grad_scale = float(self.scale/self.batch_size)
        self.assign(in_grad[0], req[0], diff * grad_scale)

        # update the center
        labels = in_data[1].asnumpy()
        label_occur = dict()
        for i, label in enumerate(labels):
            label_occur.setdefault(int(label), []).append(i)

        for label, sample_index in label_occur.items():
            sum_[:] = 0
            for i in sample_index:
                sum_ = sum_ + diff[i]
            delta_c = sum_ / (1 + len(sample_index))
            center[label] += self.center_alpha * delta_c


@mx.operator.register("gitloss")
class GitLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, git_params, scale=1.0, batchsize=64):
        super(CenterLossProp, self).__init__(need_top_grad=False)

        # convert it to numbers        
        self.id_num_class = int(num_classes[0])
        self.seq_num_class = int(num_class[1])
        self.git_params = git_params    
        self.scale = float(scale)
        self.batchsize = int(batchsize)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return ['diff_bias', 'center_bias', 'sum_bias']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)

        # store diff , same shape as input batch
        diff_shape = [self.batchsize, data_shape[1]]

        # store the center of each class , should be ( num_class, d )
        center_shape = [self.id_num_class+self.seq_num_class, diff_shape[1]]

        # computation buf
        sum_shape = [diff_shape[1],]

        output_shape = [1, ]
        return [data_shape, label_shape], [output_shape], [diff_shape, center_shape, sum_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CenterLoss(ctx, shapes, dtypes, self.id_num_class, self.seq_num_class, self.git_params, self.scale)
