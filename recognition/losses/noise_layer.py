import os

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
#os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx
import random
import queue
import pdb
import math
import numpy as np


class NoiseLayer(mx.operator.CustomOp):
    #def __init__(self, ctx, shapes, dtypes, bins):
    def __init__(self, ctx, shapes, dtypes):
        self.bins = 200            # number of histogram bins
        self.iter = 0               # current batch
        self.batchSize = shapes[0][0]
        self.slideBatchNum = 128000 // self.batchSize  # approximate K for Histogram_all
        self.valueLow = -1          # lowest cos
        self.valueHigh = 1          # highest cos
        self.lBinId = -1             # leftmost bin id
        self.rBinId = -1             # rightmost bin id
        self.ltBinId = -1            # bin id of left peak (extreme)
        self.rtBinId = -1            # bin id of right peak (extreme)
        self.topS = 0.005           # top boundary for lmost & rmost
        self.hisFilterR = 2         # radius of mean filtering for the distribution
        
        self.samplesWeight = mx.nd.ones(shape=(self.batchSize, 1), ctx=ctx)
        self.pdf = np.array([0.0]*(self.bins+1), dtype='float32')
        self.filterPdf = np.array([0.0]*(self.bins+1), dtype='float32')
        self.pcf = np.array([0.0]*(self.bins+1), dtype='float32')
        self.binIdQueue = queue.Queue()
        self.queueSize = 0
        self.noiseRatio = mx.nd.ones(shape=(1, ), dtype='float32', ctx=ctx)
        self.ctx = ctx
        self.debug_file = open("logs/"+ctx.__str__(), "w")
    
    def get_bin_id(self, cosData):
        binIds = self.bins * (cosData - self.valueLow) / (self.valueHigh - self.valueLow)
        binIds = np.clip(binIds, 0, self.bins)
        binIds = binIds.astype('int32')
        return binIds
        
    def delta(self, a, b):
        if b == -1:
            return (a - b)
        if (a > b):
            return 1
        elif (a < b):
            return -1
        else:
            return 0
        
    def get_cos(self, binIds):
        cosV = self.valueLow + (self.valueHigh-self.valueLow) * binIds / self.bins
        cosV = np.clip(cosV, self.valueLow, self.valueHigh)
        return cosV
        
    def cos2weight(self, binIds):
        
        # focus on all samples
        # weight1: normal state [0, 1]
        # constant value
        weight1 = np.ones(shape=(self.batchSize, ))

        # focus on simple&clean samples, we just offer some methods, you can try other method to archive the same purpose and get better effect.
        #someone interested in this method can try gradient-ascend-like methods, such as LRelu/tanh and so on, which we have simply tried and failed, but i still think it may make sense.
        # weight2: activative func [0, 1]
        # activation function, S or ReLu or SoftPlus
        # SoftPlus
        weight2 = np.log(1.0 + np.exp(10.0 * (binIds - self.ltBinId) / (self.rBinId - self.ltBinId))) / math.log(1.0 + math.exp(10.0))
        weight2 = np.clip(weight2, 0.0, 1.0)
        #weight2 = clamp<Dtype>(log(1.0 + exp(10.0 * (bin_id - lt_bin_id_) / (r_bin_id_ - lt_bin_id_))) / log(1.0 + exp(10.0)), 0.0, 1.0)
        
        # ReLu (linear)
        # weight2 = clamp<Dtype>(1.0 * (bin_id - lt_bin_id_) / (r_bin_id_ - lt_bin_id_), 0.0, 1.0)
        # Pow Fix
        # weight2 = pow(clamp<Dtype>(1.0 * (bin_id - l_bin_id_) / (r_bin_id_ - l_bin_id_), 0.0, 1.0), 3.0)
        # S
        # weight2 = pcf_[bin_id]

        # focus on semi-hard&clean samples, we just offer some methods, you can try other method to archive the same purpose and get better effect.
        # weight3: semi-hard aug [0, 1]
        # gauss distribution
        x = binIds
        u = self.rtBinId
        if self.rBinId == u:
            #print("======warning rBinId==u")
            weight3 = weight2
        else:
            onesX = np.ones_like(x)
            # a = (r_bin_id_ - u) / r// symmetric
            a = np.where(x > u, onesX*((self.rBinId - u) / 2.576), onesX*((u - self.lBinId) / 2.576)) # asymmetric
            # a = x > u ? ((r_bin_id_ - u) / r) : ((u - l_bin_id_) / r)// asymmetric
            weight3 = np.exp(-1.0 * (x - u) * (x - u) / (2 * a * a))
            #  linear
            # a = (r_bin_id_ - u)// symmetric
            # a = x > u ? (r_bin_id_ - u) : (u - l_bin_id_)// asymmetric
            # weight3 = clamp<Dtype>(1.0 - fabs(x - u) / a, 0.0, 1.0)
            # 
        #  without stage3
        # weight3 = weight2
        # 
        #  merge weight
        alpha = self.valueLow + (self.valueHigh-self.valueLow) * self.rBinId / self.bins
        if alpha < 0:
            alpha = 0
        if alpha > 1:
            alpha = 1
        beta = 2.0 - 1.0 / (1.0 + math.exp(5-20*alpha)) - 1.0 / (1.0 + math.exp(20*alpha-15)) # [0, 1]
        #  linear
        # beta = fabs(2.0 * alpha - 1.0)//[0, 1]
    
        # alpha = 0.0 => beta = 1.0, weight = weight1
        # alpha = 0.5 => beta = 0.0, weight = weight2
        # alpha = 1.0 => beta = 1.0, weight = weight3
        if alpha < 0.5:
            weight13 = weight1
        else:
            weight13 = weight3
        weight = beta*weight13 + (1-beta) * weight2 # [0, 1]
        # weight = 1.0// normal method
        return weight
      
    def forward(self, is_train, req, in_data, out_data, aux):
        marginData = in_data[0]
        cosData = in_data[1]
        label = in_data[2]
        cosData = mx.nd.pick(cosData, label, axis=1)
        
        logitsOutput = marginData
        noiseRatio = self.noiseRatio
        
        self.iter += 1
        
        npCosData = cosData.asnumpy()
        # update pdf
        binIds = self.get_bin_id(npCosData)
        #print(binIds)
        for k in range(len(binIds)):
            idx = binIds[k]
            self.pdf[idx] += 1
            self.binIdQueue.put(idx)
            self.queueSize += 1
        #print(self.pdf)
        # del redundancy
        while(self.queueSize > self.slideBatchNum*self.batchSize):
            topElem = self.binIdQueue.get()
            self.queueSize -= 1
            self.pdf[topElem] -= 1
            
        if (self.iter >= self.slideBatchNum):
            #mean filtering of the distribution
            self.filterPdf *= 0
            filterPdfSum = 0
            for k in range(self.hisFilterR, self.bins+1-self.hisFilterR):
                for o in range(k-self.hisFilterR, k+1+self.hisFilterR):
                    self.filterPdf[k] += self.pdf[o] / (self.hisFilterR*2 + 1)
                filterPdfSum += self.filterPdf[k]
            #update probability cumulative function
            self.pcf[0] = self.filterPdf[0] / filterPdfSum
            for k in range(1, self.bins+1):
                self.pcf[k] = self.pcf[k-1] + self.filterPdf[k] / filterPdfSum
            
            # Find endpoints and toppoints of pdf
            lBinId_ = 0
            rBinId_ = self.bins
            ltBinId_ = 0
            rtBinId_ = 0
            while (lBinId_ <= self.bins) and (self.pcf[lBinId_] < self.topS):
                lBinId_ += 1
            while (rBinId_ >= 0) and (self.pcf[rBinId_] > (1-self.topS)):
                rBinId_ -= 1
            if lBinId_ >= rBinId_:
                print("Error binId!!!")
            else:
                mBinId = (lBinId_+rBinId_) // 2
                # extreme points of the distribution
                tBinId = self.filterPdf.argmax()
                tBinIds = []
                
                for k in range(max(lBinId_, 5), min(rBinId_, self.bins-5)+1):
                    if ((self.filterPdf[k] >= self.filterPdf[k-1]) and (self.filterPdf[k] >= self.filterPdf[k+1]) 
                       and (self.filterPdf[k] > self.filterPdf[k-2]) and (self.filterPdf[k] > self.filterPdf[k+2]) 
                       and (self.filterPdf[k] > self.filterPdf[k-3]+1) and (self.filterPdf[k] > self.filterPdf[k+3]+1) 
                       and (self.filterPdf[k] > self.filterPdf[k-4]+2) and (self.filterPdf[k] > self.filterPdf[k+4]+2) 
                       and (self.filterPdf[k] > self.filterPdf[k-5]+3) and (self.filterPdf[k] > self.filterPdf[k+5]+3)):
                    #if ((self.filterPdf[k] >= self.filterPdf[k-1]) and (self.filterPdf[k] >= self.filterPdf[k+1]) 
                    #   and (self.filterPdf[k] > self.filterPdf[k-2]) and (self.filterPdf[k] > self.filterPdf[k+2]) 
                    #   and (self.filterPdf[k] > self.filterPdf[k-3]) and (self.filterPdf[k] > self.filterPdf[k+3]) 
                    #   and (self.filterPdf[k] > self.filterPdf[k-4]) and (self.filterPdf[k] > self.filterPdf[k+4]) 
                    #   and (self.filterPdf[k] > self.filterPdf[k-5]) and (self.filterPdf[k] > self.filterPdf[k+5])):
                        tBinIds.append(k)
                        k += 5
                if len(tBinIds)==0:
                    tBinIds.append(tBinId)
                
                # left/right extreme point of the distribution
                if (tBinId < mBinId):
                    ltBinId_ = tBinId
                    rtBinId_ = max(tBinIds[-1], mBinId) # fix
                    #rtBinId_ = tBinIds[-1] #not fix
                else:
                    rtBinId_ = tBinId
                    ltBinId_ = min(tBinIds[0], mBinId) # fix
                    #ltBinId_ = tBinIds[0] #not fix              
                
                self.lBinId = lBinId_
                self.rBinId = rBinId_
                self.ltBinId = ltBinId_
                self.rtBinId = rtBinId_

                #self.lBinId += self.delta(lBinId_, self.lBinId)
                #self.rBinId += self.delta(rBinId_, self.rBinId)
                #self.ltBinId += self.delta(ltBinId_, self.ltBinId)
                #self.rtBinId += self.delta(rtBinId_, self.rtBinId)

                # estimate the ratio of noise to clean
                # method1
                if (self.ltBinId < mBinId):
                    noiseRatio[0] = 2.0 * self.pcf[self.ltBinId]
                else:
                    noiseRatio[0] = 0.0
                                
                # Compute weight of each sample 
                self.samplesWeight = mx.nd.array(self.cos2weight(binIds).reshape((-1,1)), dtype='float32', ctx=self.ctx)
                if (self.iter % 1000 == 0):
                    print("======================", file=self.debug_file)
                    print("cosData: ", npCosData, file=self.debug_file)
                    print("label: ", label, file=self.debug_file)
                    print("self.samplesWeight: ", self.samplesWeight, file=self.debug_file)
                    print("self.ctx: ", self.ctx, file=self.debug_file)
                    print("lBinId_: ", lBinId_, file=self.debug_file)
                    print("rBinId_: ", rBinId_, file=self.debug_file)
                    print("ltBinId_: ", ltBinId_, file=self.debug_file)
                    print("rtBinId_: ", rtBinId_, file=self.debug_file)
                    print("noiseRatio[0]: ", noiseRatio[0], file=self.debug_file)
                    print("tBinId: ", tBinId, file=self.debug_file)
                    print("tBinIds: ", tBinIds, file=self.debug_file)
                    self.debug_file.flush()
                # Forward with weight
                logitsOutput = marginData*self.samplesWeight
                
        self.assign(out_data[0], req[0], logitsOutput)
        self.assign(out_data[1], req[0], noiseRatio)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], self.samplesWeight * out_grad[0])
        self.assign(in_grad[1], 'write', 0*in_grad[1])
        self.assign(in_grad[2], 'write', 0*in_grad[2])

@mx.operator.register("noiselayer")
class NoiseLayerProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NoiseLayerProp, self).__init__(need_top_grad=True)
        # convert it to numbers
        #self.bins = 200

    def list_arguments(self):
        return ['marginData', 'cosData', 'label']

    def list_outputs(self):
        return ['logits', 'ratio']

    #def list_auxiliary_states(self):
    #    # call them 'bias' for zero initialization
    #    return ['pdf_bias']

    def infer_shape(self, in_shape):
        # input
        marginDataShape = in_shape[0]
        cosDataShape = in_shape[1]
        labelShape = in_shape[2]
        #labelShape = (in_shape[0][0],)

        # output
        logitsShape = marginDataShape
        ratioShape = [1, ]

        # aux        
        #pdfShape = [self.bins,]
        #return [marginDataShape, cosDataShape, labelShape], [logitsShape, ratioShape], [pdfShape]
        return [marginDataShape, cosDataShape, labelShape], [logitsShape, ratioShape], []

    def create_operator(self, ctx, shapes, dtypes):
        #return NoiseLayer(ctx, shapes, dtypes, self.bins)
        return NoiseLayer(ctx, shapes, dtypes)
