import os,sys,pdb
import ctypes
from mxnet.base import _LIB
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
class CHLEPDLL:
    def __init__(self):
        libpath = os.path.join( os.path.dirname(__file__), '../build/release/chelp.dll')
        self.lib = ctypes.CDLL(libpath)
        return
theLIB = CHLEPDLL()

def get_pointer(v):
    if isinstance(v, mx.ndarray.NDArray):
        v.wait_to_read() #lazy computation of mxnet
    ptr = ctypes.c_void_p()
    _LIB.MXNDArrayGetData(v.handle, ctypes.byref(ptr))
    return ptr

class PATCH2COL:
    def __init__(self):
        self.name = 'patch2col'
        self.func = getattr(theLIB.lib, self.name)
        return
    def __call__(self, *args, **kwargs):
        in_mat_xpu = args[0]
        if in_mat_xpu.context == mx.cpu():
            in_mat = in_mat_xpu
        else:
            in_mat = in_mat_xpu.as_in_context(mx.cpu())
        channels, height, width = in_mat.shape
        in_ptr = get_pointer(in_mat)
        out_height = (height - 2) * (width - 2)
        out_width = 3*3*channels
        out_mat = nd.zeros((out_height, out_width), ctx = mx.cpu(), dtype=np.float32)
        out_ptr = get_pointer(out_mat)
        self.func(0, in_ptr, channels, height, width, out_ptr)
        if in_mat_xpu.context == mx.cpu():
            out_mat_xpu = out_mat
        else:
            out_mat_xpu = out_mat.as_in_context(mx.gpu())
        return out_mat_xpu

        
class PATCH2COL2:
    def __init__(self):
        self.name = 'patch2col_2'
        self.func = getattr(theLIB.lib, self.name)
        return
    def __call__(self, *args, **kwargs):
        in_mat_xpu = args[0]
        if in_mat_xpu.context == mx.cpu():
            in_mat = in_mat_xpu
        else:
            in_mat = in_mat_xpu.as_in_context(mx.cpu())
        channels, height, width = in_mat.shape
        in_ptr = get_pointer(in_mat)
        out_height = (height - 2) * (width - 2)
        out_mat = nd.zeros((out_height, channels, 3 * 3), ctx = mx.cpu(), dtype=np.float32)
        out_ptr = get_pointer(out_mat)
        self.func(0, in_ptr, channels, height, width, out_ptr)
        if in_mat_xpu.context == mx.cpu():
            out_mat_xpu = out_mat
        else:
            out_mat_xpu = out_mat.as_in_context(mx.gpu())
        return out_mat_xpu
        
patch2col = PATCH2COL()
patch2col_2 = PATCH2COL2()

if 0: # testing
    import mxnet as mx
    from mxnet import ndarray as nd
    import numpy as np
    a = np.random.random((4,8,8))
    a = nd.array(a, ctx = mx.gpu())
    b = patch2col(a)

    print '====================a====================='
    print a.asnumpy()
    print '====================b====================='
    print b.asnumpy()

