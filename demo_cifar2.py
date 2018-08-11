import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as autograd
import os,sys,pdb
from symbol.proto2d import Proto2DBlock

root='c:/dataset/cifar/split/'
outdir = 'output/'

if not os.path.exists(outdir):
    os.makedirs(outdir)
#round number
pretrain = 4800

lr0 = 0.1
wd = 0.005
batchSize=10 #for testIter, if set size not multipler of batch, error label value returned
imgSize=28 #after crop
channelNum=3
classNum = 10
dataShape=(batchSize,channelNum,imgSize, imgSize)
ctx = mx.gpu()


action_null = 0
action_start_projection = 1
action_end_projection = 2

trainAugList = mx.image.CreateAugmenter((channelNum,imgSize, imgSize),rand_crop=True,rand_mirror=True,mean=True,std=True)
trainIter = mx.image.ImageIter(batchSize,(channelNum,imgSize, imgSize),label_width=1,
                               path_imglist='train.lst',path_root=os.path.join(root,'train'),
                               shuffle=True,aug_list=trainAugList)

testAugList = mx.image.CreateAugmenter((channelNum,imgSize, imgSize),rand_crop=False,mean=True,std=True)
testIter = mx.image.ImageIter(batchSize,(channelNum,imgSize, imgSize),label_width=1,
                               path_imglist='test.lst',path_root=os.path.join(root,'test'),
                               shuffle=False,aug_list=testAugList)

class CIFARCONV(nn.Block):
    def __init__(self,ch,downsample=False,**kwargs):
        super(CIFARCONV, self).__init__(**kwargs)
        if downsample:
            stride = 2
        else:
            stride = 1
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=ch, kernel_size=3, strides=stride,padding=1)
            self.bn1=nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels=ch, kernel_size=3, strides=1,padding=1)
            self.bn2=nn.BatchNorm()
            if downsample:
                self.conv3=nn.Conv2D(channels=ch, kernel_size=3, strides=stride,padding=1 )
                self.bn3=nn.BatchNorm()
            else:
                self.conv3, self.bn3 = None, None
        return

    def forward(self, x,*args):
        out=mx.nd.relu(self.bn1(self.conv1(x)))
        out=mx.nd.relu(self.bn2(self.conv2(out)))
        if self.conv3 is not None:
            x = mx.nd.relu(self.bn3(self.conv3(x)))
        return mx.nd.relu(x + out)

class CIFARNET(nn.Block):
    def __init__(self,classNum,verbose=False,**kwargs):
        super(CIFARNET,self).__init__(**kwargs)
        with self.name_scope():
            self.convs,self.fcs = nn.Sequential(), nn.Sequential()
            self.convs.add( nn.Conv2D(channels=32,kernel_size=3,strides=1,padding=1)  )
            self.convs.add( CIFARCONV(ch=32) )
            self.convs.add( CIFARCONV(ch=64,downsample=True) )
            self.convs.add( CIFARCONV(ch=64,downsample=True) )
            self.convs.add( Proto2DBlock(64,20,imgSize,batchSize) )
            self.fcs.add( nn.GlobalMaxPool2D() )
            self.fcs.add(nn.Dense(classNum))
        return

    def forward(self, x,*args):
        out = x
        for net in self.convs:
            out = net(out)
        for fc in self.fcs:
            out = fc(out)
        return out
    def origin_images(self,X):
        self.convs[-1].set_origin_image_batch(X)
        return
    def project(self,action): #get filter of one layer
        if action == 'start':
            print 'start project'
            self.convs[-1].set_project_action_code( mx.ndarray.ones(1,ctx=ctx,dtype=np.int32) * action_start_projection)
        elif action == 'set':
            print 'set project'
            self.convs[-1].set_project_action_code(mx.ndarray.ones(1,ctx=ctx,dtype=np.int32) * action_end_projection)
        elif action == 'end':
            print 'end project'
            self.convs[-1].set_project_action_code(mx.ndarray.ones(1,ctx=ctx,dtype=np.int32) * action_null)
        else:
            print 'unk {}'.format(action)
        return

net = CIFARNET(classNum)
net.initialize(ctx = ctx)
#net.hybridize()

if pretrain >= 0:
    net.load_params(os.path.join(outdir,'cifar-%.4d.params'%pretrain),ctx=ctx)
    print 'load model ....'


trainer = gluon.Trainer(net.collect_params(), "sgd", {'learning_rate':lr0,"wd":wd})

loss_ce = gluon.loss.SoftmaxCrossEntropyLoss()

class LOSSREC(mx.metric.EvalMetric):
    def __init__(self,name):
        super(LOSSREC,self).__init__(name)
        return
    def update(self, labels, preds = 0):
        for loss in labels:
            if isinstance(loss, mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1
        return

train_loss = LOSSREC("train-error")
test_loss = LOSSREC("test-error")
from sklearn.metrics import accuracy_score

class HITRATE(object):
    def __init__(self,name):
        self.Y, self.Yhat = [],[]
        self.name = name
    def update(self,Y,predY):
        if isinstance(predY,mx.nd.NDArray):
            predY = predY.asnumpy()
        if isinstance(Y,mx.nd.NDArray):
            Y = Y.asnumpy()
        Yhat = np.argmax(predY,axis=1).reshape(Y.shape)
        Yhat = Yhat.reshape((1,-1)).tolist()[0]
        Y = Y.reshape((1,-1)).tolist()[0]
        self.Y.extend(Y)
        self.Yhat.extend(Yhat)
        return
    def __str__(self):
       return "({},{:.3f})".format(self.name,accuracy_score(self.Y,self.Yhat))
    def reset(self):
        self.Y, self.Yhat = [], []
        return

from matplotlib import pyplot as plt

class VISUAL_LOSS(object):
    def __init__(self):
        #plt.ion()
        self.trainloss = []
        self.testloss = []
        return
    def reduce(self,th = 100):
        if len(self.trainloss) > th:
            self.trainloss = self.trainloss[th//10:]
        if len(self.testloss) > th:
            self.testloss = self.testloss[th//10:]
        return
    def update_train(self, round, loss):
        if isinstance(loss, mx.nd.NDArray):
            loss = loss.asnumpy()[0]
        self.trainloss.append((round, loss))
        return
    def update_test(self,round,loss):
        if isinstance(loss, mx.nd.NDArray):
            loss = loss.asnumpy()[0]
        self.testloss.append((round,loss))
    def show(self):
        self.reduce()
        if len(self.trainloss) > 0:
            x = [d[0] for d in self.trainloss]
            y = [d[1] for d in self.trainloss]
            plt.plot(x,y,"r")
        if len(self.testloss) > 0:
            x = [d[0] for d in self.testloss]
            y = [d[1] for d in self.testloss]
            plt.plot(x,y,"b")
        #plt.pause(0.05)
        plt.savefig("train.png")
        return

def do_project(dataIter,net):
    dataIter.reset()
    net.project('start')
    for batch in dataIter:
        X, Y = batch.data[0].as_in_context(ctx), batch.label[0].as_in_context(ctx)
        net.origin_images(X)
        predY = net.forward(X)
        predY.wait_to_read()
    net.project('set')
    dataIter.reset()
    for batch in dataIter:
        X, Y = batch.data[0].as_in_context(ctx), batch.label[0].as_in_context(ctx)
        predY = net.forward(X)
        predY.wait_to_read()
        net.project('end')
        predY = net.forward(X)
        predY.wait_to_read()
        break #call forward()
    net.origin_images(None)
    return net

from time import time
t0 = time()

visualloss = VISUAL_LOSS()

lrch = mx.lr_scheduler.PolyScheduler(50000, base_lr=lr0,pwr=2)


round = 0
for epoch in range(5000):
    trainIter.reset()


    for batchidx, batch in enumerate(trainIter):
        round += 1
        trainer.set_learning_rate(lrch(round))
        X,Y = batch.data[0].as_in_context(ctx), batch.label[0].as_in_context(ctx)
        with autograd.record():
            predY = net.forward(X)
            loss = loss_ce(predY,Y)
        loss.backward()
        trainer.step(batchSize)
        train_loss.update(loss)
        #do_project(trainIter,net)
        if round % 5 == 0:
            print 'round {} {:.2f} {}'.format(round,(time() - t0)/60.0,train_loss.get())
            visualloss.update_train(round,train_loss.get()[1])
            visualloss.show()
    net.save_params(os.path.join(outdir,'cifar-%.4d.params'%round))
    hr = HITRATE("hit-rate")
    testIter.reset()
    for batch in testIter:
        X,Y = batch.data[0].as_in_context(ctx), batch.label[0].as_in_context(ctx)
        predY = net.forward(X)
        loss = loss_ce(predY,Y)
        test_loss.update(loss)
        hr.update(Y,predY)
    visualloss.update_test(round,test_loss.get()[1])
    visualloss.show()
    print 'epoch {} lr {:.3f} {:.2f} min {} {} {}'.format( epoch,trainer.learning_rate, (time()-t0)/60.0,
              train_loss.get(),test_loss.get(), hr)
    
    # projection
    if (1+epoch) % 1000 == 0:
        net = do_project(trainIter,net)


#plt.show()


