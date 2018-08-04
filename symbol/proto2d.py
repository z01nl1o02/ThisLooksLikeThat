import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd
import cPickle
import numpy as np
from func import patch2col,patch2col_2

def im2col(img,ks):
    if 1:
        ctx = img.context
        chNum, height, width = img.shape
        #imgPadding = nd.pad(nd.expand_dims(img,axis=0),mode='constant',pad_width=(0,0,0,0,0,2,0,2),constant_value=0)[0]
        #imgPadding = nd.pad(nd.expand_dims(img, axis=0), mode='edge', pad_width=(0,0,0,0,0,2,0,2))[0]
        imgPadding = img
        patchNum = (width-2)*(height-2)
        output = nd.zeros((patchNum, ks*ks*chNum), ctx=ctx)
        for y in range(imgPadding.shape[1] - 2):
            for x in range(imgPadding.shape[2] - 2):
                output[y*(width-2) + x] = imgPadding[:, y:y+ks, x:x+ks].reshape(ks*ks*chNum)
    else:       
        output = patch2col(img)
    return output


class Proto2D(mx.operator.CustomOp):
    def __init__(self, channels, ks):
        self.kernelSize=ks
        assert ks==3
        self.strides = 1
        self.channels=channels
        self.verbose = False
        return

    
    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context
        data = in_data[0]
        proj = aux[0]
        origin_images = aux[1]
        minDist,projWeight,prototypes = aux[2:5]
        weight = in_data[1]

        #minDist[0] = 98
        if self.verbose:
            print 'forward input start'
            print 'data {} {} {}'.format(data.min(),data.max(),data.mean())
            print 'weight {} {} {}'.format(weight.min(), weight.max(), weight.mean())
            print 'forward input end'

        batchSize, inChNum, height, width = data.shape
        outChNum, inChNum, kernelSize, _ = weight.shape

        weightMat = nd.zeros((outChNum,(width-2)*(height-2),inChNum * kernelSize * kernelSize),ctx=ctx)
        for outchidx in range(outChNum):
            w = nd.reshape(weight[outchidx],(1,-1))
            w = nd.tile( w, ((width-2) * (height-2), 1) )
            weightMat[outchidx] = w

        output = nd.zeros((batchSize, outChNum, height-2, width-2), ctx = ctx)
        for batchidx in range(batchSize):
           dataCur=data[batchidx]
           dataCur=im2col(dataCur,kernelSize)
           for outchidx in range(outChNum):
               weightCur= weightMat[outchidx]
               df = ((dataCur - weightCur)**2).sum(axis=1) + 0.00001
               output[batchidx,outchidx] = nd.reshape(-1*nd.log(df),(height-2,width-2))
        
        if self.verbose:
            print 'forward output start'
            print 'output {} {} {}'.format(output.min(), output.max(), output.mean())
            print 'forward output end'

        #
        # if self.minDist is None:
        #     self.minDist = nd.zeros(outChNum,ctx = ctx) - 99999.0
        #     self.projWeight = nd.zeros(weight.shape, ctx=ctx)
        #     self.prototypes = nd.zeros((outChNum,3,kernelSize*4, kernelSize*4), ctx=ctx)

        if proj[0] == 1: #start
            ratio = origin_images.shape[2] // width
            #print 'ratio = {} batch size = {}'.format(ratio,batchSize)
            #dataPading = nd.pad(data,mode='constant',pad_width=[0,0,0,0,0,2,0,2],constant_value=0)
            dataPading = data
            for batchidx in range(batchSize):
                dataCur =  dataPading[batchidx]
                for outchidx in range(outChNum):
                    if 0:
                        locx,locy = 0, 0
                        for y in range(height):
                            for x in range(width):
                                if output[batchidx,outchidx][y,x] > output[batchidx,outchidx][locy,locx]:
                                    locy,locx = y, x
                    else:
                        tmp = nd.reshape( output[batchidx,outchidx], (1,-1) )
                        pos = nd.argmax(tmp, axis=1).asnumpy()[0]
                        pos = np.int32(pos)
                        locy = pos//output[batchidx,outchidx].shape[1]
                        locx = pos - locy * output[batchidx,outchidx].shape[1]
                        #print output[batchidx,outchidx].asnumpy()
                        #print 'locx locy max = {} {} {}'.format(locx,locy,output[batchidx,outchidx][locy,locx])
                    if output[batchidx,outchidx][locy,locx] >  minDist[outchidx]:
                        minDist[outchidx] = output[batchidx,outchidx][locy,locx]
                        projWeight[outchidx] = dataPading[batchidx][:,locy:locy+kernelSize, locx:locx+kernelSize]
                        x0,y0 = locx*ratio,locy * ratio
                        x1,y1 = (locx+kernelSize)*ratio, (locy+kernelSize)*ratio
                        #print 'ch,x0,x1,y0,y1,val = {},{},{},{},{},{}'.format(outchidx,x0,x1,y0,y1,minDist[outchidx].asnumpy()[0])
                        prototypes[outchidx] = origin_images[batchidx][:,y0:y1,x0:x1]
                        #self.prototypes[outchidx] = origin_images[batchidx][:,0:12,0:12]
                        #with open('proto.pkl','wb') as f:
                        #    cPickle.dump(self.prototypes, f)
                        #with open('proto.pkl','rb') as f:
                        #    self.prototypes = cPickle.load(f)
                        #print self.prototypes[outchidx].asnumpy().std()
        elif proj[0] == 2: #end
           for outchidx in range(outChNum):
               if minDist[outchidx] < -99:
                   continue
               weight[outchidx] = projWeight[outchidx]
           self.assign(in_data[1],"write",weight)
           #print 'dist min = {} max = {}'.format(minDist.asnumpy().min(), minDist.asnumpy().max())
           in_data[1].wait_to_read()
           with open('proto.pkl','wb') as f:
               cPickle.dump(prototypes,f)
               print 'proto.pkl updated'
        self.assign(out_data[0],req[0],output)

        return

    def norm_grad(self,out_data_ch, out_grad_ch, inChNum):
        out = nd.exp((-1)*out_data_ch)
        out = out_grad_ch / out
        out = nd.pad( nd.expand_dims(nd.expand_dims(out,axis = 0), axis=0), mode = 'constant', pad_width=[0,0,0,0,2,2,2,2], constant_value=0)[0]
        out = nd.tile(out, (inChNum, 1, 1))
        return out

    def calc_grad_z(self, norm_grad, in_data, rot_weight, ctx):
        inChNum, height, width = in_data.shape
        _, ks, _ = rot_weight.shape
        dataMat = nd.zeros((width * height, inChNum, ks * ks), ctx=ctx )
        if 0:
            for y in range(height):
                for x in range(width):
                    val = nd.tile( nd.reshape(in_data[:, y, x],(inChNum, 1, 1)), (1, ks, ks) )
                    dataMat[y * width + x, :, ] = nd.reshape(val, (inChNum, ks * ks ))
        else:
            dataMat0 = nd.transpose( nd.reshape(in_data,(inChNum,-1)),(1,0))
            dataMat1 = nd.expand_dims(dataMat0,axis=-1)
            dataMat = nd.tile(dataMat1,(1,1,ks * ks))
        weightMat = nd.tile( nd.reshape(rot_weight,  (1, inChNum, ks*ks)), (width*height, 1, 1) )
        if 0:
            gradMat = nd.zeros((width * height, inChNum, ks * ks),ctx=ctx)
            for y in range(height):
                for x in range(width):
                    gx, gy = x + 2, y + 2
                    gradMat[y * width + x,:,] =  nd.reshape(norm_grad[:,gy-2:gy+1, gx-2:gx+1],(inChNum,ks*ks))
        else:
            gradMat = patch2col_2(norm_grad)
        output = (gradMat * (dataMat - weightMat)).sum(axis = 2)
        output = nd.reshape(output, (height, width, inChNum))
        output = nd.transpose(output, (2, 0, 1))
        return output

    def calc_grad_w(self, norm_grad, in_data, weight, ctx):
        inChNum, height, width = in_data.shape
        _, ks, _ = weight.shape
        dataPad = nd.pad(nd.expand_dims(in_data,axis=0),mode='constant',pad_width=[0,0,0,0,0,2,0,2], constant_value=0)[0]
        output = nd.zeros((inChNum,ks,ks),ctx=ctx)
        for y in range(ks):
            for x in range(ks):
                weightCur = nd.tile( nd.reshape(weight[:,y,x],(inChNum,1,1)), (1,height,width) )
                val = 2 * norm_grad[:, 2:, 2:] * ( dataPad[:,y:y+height,x:x+width] - weightCur  )
                output[:,y,x] = nd.reshape(val, (inChNum, width*height)).sum(axis=1)
        return output
    def add_r2_to_grad(self,dataOut,grad, C = 1): #second part of cost function
        batchSize, chNum, height, width = dataOut.shape
        val = nd.exp( (-1) * (nd.reshape(dataOut,(batchSize,-1)).max(axis=1) ) ).mean()
        for batchidx in range(batchSize):
            idx = nd.argmax( nd.reshape(dataOut[batchidx],(1,-1)), axis=1)
            #print idx
            tmp = nd.reshape(grad[batchidx],(1,-1))
            tmp[0,idx] += C / batchSize
            grad[batchidx] = nd.reshape( tmp, grad[batchidx].shape)
        return grad
    def show_max_dataout(self,dataOut_):
        dataOut = dataOut_.as_in_context(mx.cpu()).asnumpy()
        batchSize, chNum, height, width = dataOut.shape
        maxV = 0
        for batchidx in range(batchSize):
            maxV += np.exp(-dataOut[batchidx].max())
        print maxV / batchSize
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dataIn = in_data[0]
        dataOut = out_data[0]
        weight = in_data[1]
        grad = self.add_r2_to_grad(dataOut,out_grad[0])
        #self.show_max_dataout(dataOut)  #to show result of R2
        if self.verbose:
            print 'grad max = {} R2 = {}'.format( out_grad[0].max(), costR2 )
            print 'backward input start'
            print 'len of out_grad {}, len of in_grad {}'.format(len(out_grad), len(in_grad))
            print 'shape out_grad[0]:{} in_grad[0]:{} in_grad[1]:{}'.format(out_grad[0].shape, in_grad[0].shape, in_grad[1].shape)
            print 'backward input end'

        batchidx, inChNum, height, width = dataIn.shape
        batchSize, outChNum, _, _ = grad.shape
        outChNum, _, kernelSize, _ = weight.shape

        ctx = dataIn.context

        weightRot = nd.flip( nd.flip(weight, axis=3), axis=2 )

        dz = nd.zeros((batchSize, inChNum, height, width))
        dw = nd.zeros((outChNum, inChNum, kernelSize, kernelSize))

        for batchidx in range(batchSize):
            inDataCur = dataIn[batchidx]
            outDataCur = dataOut[batchidx]
            for outchidx in range(outChNum):
                weightCur = weight[outchidx]
                weightRotCur = weightRot[outchidx]
                normGrad = self.norm_grad(outDataCur[outchidx], grad[batchidx,outchidx], inChNum)
                # should be (w - z)
                dz[batchidx] -= self.calc_grad_z(normGrad, inDataCur, weightRotCur, ctx)
                dw[outchidx] -= self.calc_grad_w(normGrad, inDataCur, weightCur, ctx)
        if self.verbose:
            print 'backward output start'
            print 'grad input: {} {} {}'.format(grad.min(), grad.max(), grad.mean())
            print 'grad output: {} {} {}'.format(dz.min(), dz.max(),dz.mean())
            print 'dw output: {} {} {}'.format(dw.min(), dw.max(), dw.mean())
            print 'backward output end'

        self.assign(in_grad[0],req[0],dz)
        self.assign(in_grad[1],req[1],dw)
        return


@mx.operator.register("proto2d")
class Proto2DProp(mx.operator.CustomOpProp):
    def __init__(self,channels,kernelSize,origin_image_size): #parameters during initialization
        super(Proto2DProp,self).__init__(need_top_grad=True)
        self.kernelSize = int(kernelSize)
        self.channels = int(channels)
        self.origin_image_size = int(origin_image_size)
    def list_arguments(self): #parameter during forward()
        return ["input","weight"]
    def list_outputs(self): #output of forward
        return ['output']
    def list_auxiliary_states(self): #aux parameter during forward()
        return ['project_action','origin_image_shape','minDistCh','projW','prototypes']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        weight_shape = in_shape[1]
        output_shape = (data_shape[0],weight_shape[0],data_shape[2]-2,data_shape[3]-2) # shrink is necessary because padding will cause last value is always maximum
        project_action_shape = (1,)
        origin_image_shape = (data_shape[0],3, self.origin_image_size , self.origin_image_size) #origin image shape
        min_dist_ch_shape = (weight_shape[0],)
        project_w_shape = weight_shape
        scale = np.int32(self.origin_image_size // data_shape[2])
        prototypes_shape = (weight_shape[0],3,weight_shape[2] * scale, weight_shape[3] * scale)
        return (data_shape,weight_shape),(output_shape,),(project_action_shape,origin_image_shape,min_dist_ch_shape,project_w_shape,prototypes_shape)
    def infer_type(self, in_type):
        dtype = in_type[0]
        return (dtype,dtype),(dtype,),(np.int32,dtype,dtype,dtype,dtype)
    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Proto2D(self.channels,self.kernelSize)

class Proto2DBlock(nn.Block):
    def __init__(self,in_channels, out_channels,
                 origin_image_size, #size of the network input
                 batchSize,
                 kernel_size=3, **kwargs):
        super(Proto2DBlock,self).__init__(**kwargs)
        #configure parameters
        self.kernelSize = kernel_size
        self.inChNum = in_channels
        self.outChNum = out_channels #outChNum
        self.batchSize = batchSize
        self.origin_image_size = origin_image_size
        self.project_action = None
        self.ctx = None
        self.origin_image = None

        self.minDist = None
        self.projWeight = None
        self.prototypes = None

        #learnable parameters
        self.weights = self.params.get("weight",shape = (out_channels, in_channels, kernel_size, kernel_size)) #define shape of kernel
        return
    @property
    def weight(self):
        return self.weights
    @weight.setter
    def weight(self,val):
        self.weights = val

    def set_project_action_code(self,code):
        if self.ctx is None:
            print 'error:run forward before projection'
            return
        if code == 1:
            self.reset_project()
        self.project_action = nd.ones((1,),ctx=self.ctx,dtype=np.int32) * code
        return
    def reset_project(self):
        self.minDist = nd.zeros(self.outChNum,ctx = self.ctx) - 99999.0
        self.projWeight = nd.zeros((self.outChNum,self.inChNum,self.kernelSize,self.kernelSize), ctx=self.ctx)
        self.prototypes = nd.zeros((self.outChNum,3,self.kernelSize*4, self.kernelSize*4), ctx=self.ctx) #assume downsample scale = 4
        return

    def set_origin_image_batch(self,origin_images):
        if origin_images is None:
            self.origin_image = nd.zeros((self.batchSize,3,self.origin_image_size,self.origin_image_size),ctx=self.ctx)
            return
        self.origin_image = origin_images
        return
    def forward(self,x, *args):
        ctx = x.context
        if self.ctx is None:
            self.ctx = ctx
            self.project_action = nd.ones((1,),ctx=ctx,dtype=np.int32) * (-1)
            self.origin_image = nd.zeros((self.batchSize,3,self.origin_image_size,self.origin_image_size),ctx=ctx)
            self.reset_project()
        y = mx.nd.Custom(x,self.weights.data(ctx), self.project_action, self.origin_image,
                         self.minDist, self.projWeight,self.prototypes,
                         channels=self.outChNum, kernelSize=self.kernelSize, origin_image_size = self.origin_image_size,
                         op_type="proto2d")
        return y


