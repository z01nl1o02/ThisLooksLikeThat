import mxnet as mx
import cPickle
import numpy as np
from matplotlib import pyplot as plt
import cv2

mean = np.array([123.68,116.28,103.53])
std = np.array([58.395,57.12,57.375])

with open('proto.pkl','rb') as f:
    protos = cPickle.load(f)

protos = mx.nd.transpose(protos, (0,2,3,1))
protos = protos.asnumpy()

print protos.shape

for k in range(protos.shape[0]):
    img = protos[k]
    img = (protos[k] * std) + mean
    img = np.uint8(img)
    img = cv2.resize(img,(64,64))
    cv2.imwrite('tmp/%d.jpg'%k,img)





