import mxnet as mx
import gzip,logging
import pdb,pickle
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
import numpy as np



batchSize = 100
classNum = 10
ctx = mx.gpu()
verbose = False

trainIter = mx.io.MNISTIter(batch_size = batchSize,image = 'data/train-images-idx3-ubyte', label='data/train-labels-idx1-ubyte')
validIter = mx.io.MNISTIter(batch_size = batchSize,image = 'data/t10k-images-idx3-ubyte', label='data/t10k-labels-idx1-ubyte')


class DEMONET(nn.HybridBlock):
    def __init__(self,outputNum,verbose=False,**kwargs):
        super(DEMONET,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            self.flatten = nn.Flatten()
            self.fc1  = nn.Dense(512)
            self.fc2  = nn.Dense(256)

            #self.conv1 = nn.Conv2D(channels=10,kernel_size=3,strides=2,padding=1)
            #self.conv2 = nn.Conv2D(channels=20,kernel_size=3,strides=4,padding=1)
            #self.pool1 = nn.GlobalAvgPool2D()
            self.fc=nn.Dense(outputNum)
        return
    def hybrid_forward(self, F, x, *args, **kwargs):
        if(self.verbose):
            print 'input:',x.shape
        out=F.relu(self.fc1(x))
        if(self.verbose):
            print "level 1:",out.shape
        out=F.relu(self.fc2(out))
        if(self.verbose):
            print 'level 2;',out.shape
        #out=self.pool1(out)
        if(self.verbose):
            print 'level 3:',out.shape
        out = F.relu(self.fc(out) )
        if(self.verbose):
            print 'level 4:',out.shape
        return out

if verbose:
    net = DEMONET(classNum,verbose=True)
    net.initialize(ctx=mx.cpu())
    imgSize = 28
    X = np.random.rand(1 * batchSize * imgSize * imgSize).reshape((batchSize,1,imgSize,imgSize))
    net.forward(mx.nd.array(X))
    exit(0)
else:
    net = DEMONET(classNum,verbose=False)
    net.initialize(ctx=ctx)
    net.hybridize()




trainer=gluon.Trainer(net.collect_params(),"sgd",{"learning_rate":1.0,"wd":0.00000})

loss_se=gluon.loss.SoftmaxCrossEntropyLoss()


class LOSSREC(mx.metric.EvalMetric):
    def __int__(self, name):
        super(CLS_LOSS,self).__init__(name)
    def update(self,labels,preds = None):
        for loss in labels:
            if isinstance(loss,mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1

train_cls_loss = LOSSREC("train class-error")
test_cls_loss = LOSSREC("test class-error")

def calc_hr(Y,predY):
    predY,Y = predY.asnumpy(), Y.asnumpy()
    m1 = np.tile( np.reshape(predY.max(axis=1),(-1,1)),(1, predY.shape[1] ) )
    normY = predY == m1
    Y = Y.astype(np.int32).tolist()
    hr = np.mean([normY[row,y] for row,y in enumerate(Y)])
    return hr

from time import time
t0 = time()
for epoch in range(200):
    trainIter.reset()
    for batchidx,batch in enumerate(trainIter):
        X,Y = batch.data[0].as_in_context(ctx), batch.label[0].as_in_context(ctx)
        with autograd.record():
            predY = net.forward(X)
            loss = loss_se(predY,Y)
        loss.backward()
        trainer.step(batchSize)
        train_cls_loss.update(loss)
    timeCost = (time() - t0)/60.0
    validIter.reset()
    test_cls_loss.reset()
    hrs = []
    for batch in validIter:
        X,Y = batch.data[0].as_in_context(ctx),batch.label[0].as_in_context(ctx)
        #print Y.asnumpy().min(), Y.asnumpy().max()
        predY = net.forward(X)
        loss = loss_se(predY,Y)
        test_cls_loss.update(loss)
        hrs.append( calc_hr(Y,predY) )
    hr = np.asarray(hrs).mean()
    print('Epoch {} {:.2f} min {}:{:.2f} {}:{:.2f} hr:{:.2f}'.format(epoch,timeCost,train_cls_loss.get()[0], train_cls_loss.get()[1],
                                        test_cls_loss.get()[0], test_cls_loss.get()[1],hr))
    net.export("output/mnist")


