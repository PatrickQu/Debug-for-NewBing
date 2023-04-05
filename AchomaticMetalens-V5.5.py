# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 15:11:57 2022
v5.5版本说明：做出较大改动，是v7.2的翻版，仅修改参数，不进行FOV的优化，程序主体不变。
v7.2版本说明：对v7.1的优化参数进行较大改动。基础优化思路不变，改为优化一维变量，然后将
    转变为二维矩阵再进行优化。
v5.4.4版本说明：和v5.4.3基本相同。给meta外加入mask，滤过样品外的光。同时极大地减小NN，加快计算速度。
v5.4.3版本说明：和v5.4.2基本相同，区别之处在于改变了setup，使用550um的玻璃基底，最后焦点在空气里。
v5.4.2版本说明：考虑到优化收敛性较差以及内存的限制，尝试保存梯度求平均值来进行优化。
v5.4版本说明：使用指定初始值的方法计算两层消色差透镜。所有计算在二维下进行。
v5.3.4版本说明：从v5.3.2而来，基本相同。不同之处在于前两层全部去掉相位优化的作用，仅保留光阑。
v5.3.2版本说明：和5.3基本类似。区别之处在于tfrecord还记录了输入光场，意在进一步提高gpu利用率
    除此之丸，改成四层
v5.3版本说明：和v5.2原理相同，有下述改动：
    1. 由于预处理太慢，使用tf.record.处理方式：tfrecord仅记录[波长，角度]和[输出]
        输入图像的建立加入到model中
    2. 将可优化参数改为一维相位，在计算时，先将其旋转成二维再做计算。
v5.2版本说明：v5.1写完了，但是显存放不下。只能进行大改。
    放弃一次性计算所有波长所有入射角。采用类神经网络的训练方法，产生波长和入射角度随机的数据集，
    然后分batch进行训练。
    为了简化，这一版先不使用tf.record，仅保存波长和入射角度作为数据集，在preprocess中生成
    入射光场和目标光场。注意矩阵运算的使用.
v5.1版本说明：整体v5目的是计算消色差斜入射透镜。考虑到斜入射的不对称性，需要改成二维模拟。
    这一版直接上三层。
v4.3.2版本说明：基于v4.3.改成三层。
v4.3版本说明：基于v4.2.由于4.2的循环运算太慢，改成矩阵运算。抛弃循环拥抱矩阵。
v4.2版本说明：基于v4.1.2。消色差斜入射透镜。先写两层。
v4.1.2版本说明：基本基于v4.1。第二片meta尺寸加大到第一片的两倍
v4.1版本说明：先写一个单色斜入射透镜。焦点移动满足f*sin(theta)的规律。
v4.0版本说明：基于v3改动而来，作为v4的先遣版本，用来验证斜入射
v3版本说明：大改动：改成一维模拟。基于第二版。
v2版本说明：第二版的消色差透镜优化程序。完全基于第一版，改成双层的。
v1版本说明：第一版的消色差透镜。同时优化相位和群延迟。

@author: qgy
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from tensorflow import keras
from tensorflow.keras import datasets, optimizers, Sequential
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import scipy.io as scio


radius = tf.constant(80e-6, dtype = tf.complex64)
n = tf.constant(1., dtype=tf.complex64)
n_sub = tf.constant(1.53, dtype = tf.complex64)
lamb0 = tf.linspace(0.45e-6, 0.65e-6, 3)
# lamb0 = [0.42e-6, 0.532e-6, 0.633e-6]
lamb0 = tf.cast(lamb0, dtype = tf.complex64)
lamb = lamb0 / n      # 介质中的波长
ssratio = 3 # ss: super sampling

fmax = 2e-15    # upper bound of group delay
dl = tf.constant(0.28e-6, dtype=tf.complex64)   # pixel size
NA = tf.constant(0.3 , dtype = tf.complex64)     # N.A. of the metalens
gap = tf.constant(335e-6, dtype = tf.complex64)

N = int(abs(2*radius/dl))
NN = 681
f = radius/np.tan(np.arcsin(NA))    # focal length
wvlnum = len(lamb0)

pi = tf.convert_to_tensor(np.pi, dtype=tf.complex64)
c = tf.constant(2.99792458e8, dtype=tf.complex64)
w0 = 2*pi*(c/lamb0[0])

dlp = dl / ssratio
fxmp = 1 / dlp
fymp = 1 / dlp

batchsz = 3
batchratio = 120/batchsz
rpt = 5000

## 1D Initial value
InitialVal0 = np.load('AML-byint-v4.4-NA0.3-p0.28um-gap335um-r80um-2fs-DesignedInitial-meta0.npy')
InitialVal1 = np.load('AML-byint-v4.4-NA0.3-p0.28um-gap335um-r80um-2fs-DesignedInitial-meta1.npy')

gdtmp = (InitialVal0[1] + fmax/2) / fmax
InitialVal0[1] = -tf.math.log(1/(gdtmp) - 1)
gdtmp = (InitialVal1[1] + fmax/2) / fmax
InitialVal1[1] = -tf.math.log(1/(gdtmp) - 1)

phaseinit1 = tf.constant_initializer(InitialVal0[0])
gdinit1 = tf.constant_initializer(InitialVal0[1])
phaseinit2 = tf.constant_initializer(InitialVal1[0])
gdinit2 = tf.constant_initializer(InitialVal1[1])

def sd_lensphase(lad):      # generating a standard lens phase
    lad = tf.cast(lad, dtype = tf.complex64)
    x = tf.linspace(-radius, radius, N)
    x = tf.cast(x, dtype = tf.complex64)
    y = x
    xx,yy = tf.meshgrid(x,y)
    
    k = 2*pi/lad   
    phase = -k*(tf.math.sqrt((tf.math.square(xx)+tf.math.square(yy)+f*f))-f)    
    phase = tf.cast(tf.math.real(phase), dtype = tf.float32)
    phase = tf.math.mod(phase, 2*np.pi)
    phase = tf.cast(phase, dtype = tf.complex64)       
    return phase

def processor2(inp):     # arbitrary --> NN*ssratio without ssratio
    n1 = len(inp)
    a00 = tf.zeros([int(NN*ssratio / 2 - n1 / 2), n1], dtype=tf.complex64)
    a01 = a00
    a10 = tf.zeros([NN*ssratio, int(NN*ssratio / 2 - n1 / 2)], dtype=tf.complex64)    
    a11 = a10  
    arr = tf.concat([a00, inp, a01], axis=0)
    inp = tf.concat([a10, arr, a11], axis=1)    
    return inp

@tf.function()
def processor3(inp):     # arbitrary --> NN*ssratio with super sampling
    n1 = len(inp)
    a00 = tf.zeros([int(NN / 2 - n1 / 2), n1], dtype=tf.complex64)
    a01 = a00
    a10 = tf.zeros([NN, int(NN / 2 - n1 / 2)], dtype=tf.complex64)    
    a11 = a10  
    arr = tf.concat([a00, inp, a01], axis=0)
    inp = tf.concat([a10, arr, a11], axis=1)
    inp = tf.repeat(inp, ssratio, axis=0)
    inp = tf.repeat(inp, ssratio, axis=1)
    return inp

def DimensionTransMatrix(nn):
    # 计算维度变换的矩阵
    x2 = np.linspace(-int(nn/2), int(nn/2), nn)
    xx, yy = np.meshgrid(x2, x2)
    raa = np.sqrt(np.square(xx)+np.square(yy))
    raa[raa > int(nn/2)] = 0
    raa = nn/2-raa
    raa = np.reshape(raa,[nn*nn,1])
    TransMat = tf.one_hot(raa, nn)
    TransMat = np.reshape(TransMat,[nn*nn,nn])
    return TransMat
    
TransMatN = DimensionTransMatrix(N)

def OneD2TwoD(phase1d,transmat):
    # 将一维的相位转化为二维矩阵
    nlength = len(phase1d)
    phase1d = tf.reshape(phase1d,[nlength,1])
    phase2d = transmat @ phase1d
    phase2d = tf.reshape(phase2d,[nlength,nlength])
    return phase2d

def source(nn):      # generating the circle intensity
    x = np.linspace(-1, 1, nn)
    y = x
    xx, yy = np.meshgrid(x, y)
    raa = np.square(xx)+np.square(yy)
    Ein = np.ones([nn, nn])
    Ein[raa > 1] = 0
    Ein = tf.cast(Ein, dtype = tf.complex64)
    Ein = processor3(Ein)
    return Ein

Ein = source(N)
Ein = tf.reshape(Ein, [1, NN*ssratio, NN*ssratio])
Ein = tf.repeat(Ein, batchsz, axis = 0)

@tf.function()
def source2(ein, lad, inang):
    # Adding phase to the circular incident field
    ## 乘上倾斜相位
    kvector = 2*pi/lad
    xposition = tf.cast(range(0, NN*ssratio, 1), dtype = tf.complex64)   # 横向的位置坐标
    xposition = xposition*dlp
    phase_delay = tf.math.exp(1j*kvector*tf.math.sin(inang)*xposition)     # 不同x位置的相位延迟
    phase_delay = tf.reshape(phase_delay, [NN*ssratio, 1])
    phase_delay = tf.tile(phase_delay, [1, NN*ssratio])
    x_phase = ein*phase_delay
    ein = ein*x_phase   # 去掉强度为0的点的相位      
    return ein

@tf.function()
def source22(ein, lad, inang):
    # Adding phase to the circular incident field, considering batch size
    ## 乘上倾斜相位
    lad = tf.reshape(lad, [batchsz, 1])
    inang = tf.reshape(inang, [batchsz, 1])
    kvector = 2*pi/lad
    xposition = tf.cast(range(0, NN*ssratio, 1), dtype = tf.complex64)   # 横向的位置坐标
    xposition = xposition*dlp
    xposition = tf.reshape(xposition, [1,NN*ssratio])
    xposition = tf.repeat(xposition, batchsz, axis =0)
    phase_delay = tf.math.exp(1j*kvector*tf.math.sin(inang)*xposition)     # 不同x位置的相位延迟
    phase_delay = tf.reshape(phase_delay, [batchsz, NN*ssratio, 1])
    phase_delay = tf.tile(phase_delay, [1, 1, NN*ssratio])
    x_phase = ein*phase_delay
    ein = ein*x_phase   # 去掉强度为0的点的相位      
    return ein

@tf.function()
def prop(ein, dz, lad, nn):  # propagating calculation by angular spectrum method
    lad = lad / nn    
    kk = 2*pi / lad
    Np = NN * ssratio

    fxp = tf.linspace(-fxmp / 2, fxmp / 2, Np)
    fyp = tf.linspace(-fymp / 2, fymp / 2, Np)
    fxp = tf.cast(fxp, dtype=tf.complex64)  # 频率坐标
    fyp = tf.cast(fyp, dtype=tf.complex64)

    [fxp, fyp] = tf.meshgrid(fxp, fyp)

    fre_Ein = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(ein)))  # 要先验证下tf的傅里叶变换需要几次fftshift
    fre_Eout = fre_Ein * tf.exp(1j * kk * dz * tf.sqrt(1 - tf.square(fxp * lad) - tf.square(fyp * lad)))
    Eout = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(fre_Eout)))
    return Eout

@tf.function()
def prop2(inputlist, dz, nn):  # propagating calculation by angular spectrum method
    # Input size: list
    # Field size: [batchsz, NN*ssratio, NN*ssratio]
    lad0, ein = inputlist
    lad = lad0/nn
    kk = 2*pi / lad
    Np = NN * ssratio

    fxp = tf.linspace(-fxmp / 2, fxmp / 2, Np)
    fyp = tf.linspace(-fymp / 2, fymp / 2, Np)
    fxp = tf.cast(fxp, dtype=tf.complex64)  # 频率坐标
    fyp = tf.cast(fyp, dtype=tf.complex64)

    [fxp, fyp] = tf.meshgrid(fxp, fyp)
    fxp = tf.reshape(fxp, [1, NN*ssratio, NN*ssratio])
    fyp = tf.reshape(fyp, [1, NN*ssratio, NN*ssratio])

    fre_Ein = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(ein,[1,2])),[1,2])  # 要先验证下tf的傅里叶变换需要几次fftshift
    fre_Eout = fre_Ein * tf.exp(1j * kk * dz * tf.sqrt(1 - tf.square(fxp * lad) - tf.square(fyp * lad)))
    Eout = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(fre_Eout,[1,2])),[1,2])
    outputlist = [lad0, Eout]
    return outputlist

@tf.function()
def focuspoint(ein, lad, z = f, lensphase = None):     # generating focus point field
    if lensphase == None:
        lensphase = sd_lensphase(lad)
    lensphase = processor3(lensphase)
    lensphase = tf.math.exp(1j * lensphase)
    fp_t = prop(ein*lensphase, z, lad, n)
    return fp_t

@tf.function()
def preprocess(wvll, angg):
    # Generate the incident field and target image for the specific wavelength and incident angle
    # unit: m, rad
    ein = source2(Ein, wvll, angg)
    eintmp = source2(Ein, wvll, tf.constant(0, dtype = tf.complex64))    # The normal incident field
    eouttmp = focuspoint(eintmp, wvll) # The field at the focal plane with normal incident
    shift_d = (f)*tf.math.tan(angg)
    shift_p = int(tf.math.real(shift_d / dlp))
    a0 = tf.zeros([shift_p, NN*ssratio], dtype = tf.complex64)
    a1= eouttmp[0:NN*ssratio-shift_p, :]
    etar = tf.concat([a0, a1], axis = 0)
    wvll2 = wvll*tf.ones([NN*ssratio, NN*ssratio], dtype = tf.complex64)
    # angg2 = tf.reshape(angg, [1,1])
    # wvll2 = wvll
    # wvll2 = tf.reshape(wvll, [1])
    angg2 = angg
    einall = tf.stack([ein, wvll2], axis = 0)
    # einall = [ein, wvll2]
    return einall, etar, wvll2, angg2
    
# Loading the data
raw_dataset = tf.data.TFRecordDataset('metalens-fixedcondition.r80-NA0.3-cont-NN681-ssratio3-focusinair')
feature_description = {
    'condition': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'incident': tf.io.FixedLenFeature([], tf.string, default_value=''),    
    'patt': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function).batch(batchsz).repeat(rpt)
dataset_iter = iter(parsed_dataset)

# # Simple test of the dataset
# for step100 in range(5):
#     dataset_batch = next(dataset_iter)
#     condition = dataset_batch['condition']
#     condition = tf.io.decode_raw(condition, tf.float32)
#     condition = tf.cast(condition, dtype = tf.complex64)
#     ##
#     inputfield = dataset_batch['incident']
#     inputfield = tf.io.decode_raw(inputfield, tf.complex64)
#     inputfield = tf.reshape(inputfield, [-1,NN*ssratio, NN*ssratio])
#     inputfield = tf.cast(inputfield, dtype = tf.complex64)
#     ##
#     y1 = dataset_batch['patt']
#     y1 = tf.io.decode_raw(y1, tf.float32)
#     y1 = tf.reshape(y1, [-1,NN*ssratio, NN*ssratio])
#     y1 = tf.cast(y1, dtype = tf.float32)
    # plt.figure()
    # plt.imshow(abs(y1))
    # print(np.array(condition))

def focusprocess_2layer(ein, lad, slicestep, z, lensphase0, lensphase1):
    slicestep = tf.cast(slicestep, dtype = tf.complex64)
    lensphase0 = tf.cast(lensphase0, dtype = tf.complex64)
    lensphase1 = tf.cast(lensphase1, dtype = tf.complex64)
  
    lensphase0 = processor3(lensphase0)
    lensphase1 = processor3(lensphase1)
 
    lensphase0 = tf.math.exp(1j * lensphase0)
    lensphase1 = tf.math.exp(1j * lensphase1)
 
    outxz = []
    # layer1
    slicenum1 = int(gap/slicestep)
    for i in range(slicenum1):
        out = prop(ein*lensphase0, slicestep*i, lad, n_sub)
        outxz.append((out[:, int(NN*ssratio/2)]))
    
    # layer2
    out = prop(ein*lensphase0, gap, lad, n_sub)
    slicenum2 = int(z/slicestep)
    for i in range(slicenum2):      
        out2 = prop(out*lensphase1, slicestep*i, lad, n)
        outxz.append((out2[:, int(NN*ssratio/2)]))

    outxz = np.array(outxz)
    outxz = np.transpose(outxz)
    return outxz

def focusprocess_meta2only(ein, lad, slicestep, z, lensphase1):
    slicestep = tf.cast(slicestep, dtype = tf.complex64)
    lensphase1 = tf.cast(lensphase1, dtype = tf.complex64)
  
    lensphase1 = processor3(lensphase1)
 
    lensphase1 = tf.math.exp(1j * lensphase1)
 
    outxz = []
    # layer1
    slicenum1 = int(z/slicestep)
    for i in range(slicenum1):
        out = prop(ein*lensphase1, slicestep*i, lad, n_sub)
        outxz.append((out[:, int(NN*ssratio/2)]))
    outxz = np.array(outxz)
    outxz = np.transpose(outxz)        
    return outxz

class MyDense(tf.keras.layers.Layer):
    # input: unmodulated field at each wavelength: [NN*ssratio, NN*ssratio]
    # output: modulated field at each wavelength: [NN*ssratio, NN*ssratio]
    # phase & gd: 1D. Need to transfor to 2D
    def __init__(self, inp_dim):
        super(MyDense, self).__init__()
        self.phi = self.add_weight('phi', shape=[inp_dim], initializer=phaseinit1)
        self.gd = self.add_weight('gd', shape=[inp_dim], initializer=gdinit1)
        # self.phi = self.add_weight('phi', shape=[inp_dim], initializer=tf.keras.initializers.HeNormal())
        # self.gd = self.add_weight('gd', shape=[inp_dim], initializer=tf.keras.initializers.HeNormal())
    
    def call(self, inputlist, training=None):   
        # generating the complex field with phase modulation at each wavelength  
        wvll, inputs = inputlist
        phase0 = self.phi
        gd = fmax*tf.sigmoid(self.gd) - fmax/2
        phase0 = (phase0+tf.reverse(phase0, axis = [0]))/2
        gd = (gd+tf.reverse(gd, axis = [0]))/2
        
        phase0 = OneD2TwoD(phase0, TransMatN)   # Convert the 1D to 2D
        gd = OneD2TwoD(gd, TransMatN)
        
        phase0 = tf.cast(phase0, dtype=tf.complex64)
        gd = tf.cast(gd, dtype=tf.complex64)
        phase0 = processor3(phase0)
        gd = processor3(gd)
        phase0 = tf.reshape(phase0, [1, NN*ssratio,NN*ssratio])
        gd = tf.reshape(gd, [1, NN*ssratio, NN*ssratio])
        w = 2*pi*(c/wvll)-w0    # angular frenquency differential     
        phase = phase0 + w*gd
        
        phase = tf.math.exp(1j * phase)
        out0 = inputs*phase
        return [wvll, out0]
    
class MyDense2(tf.keras.layers.Layer):
    # input: unmodulated field at each wavelength: [NN*ssratio, NN*ssratio]
    # output: modulated field at each wavelength: [NN*ssratio, NN*ssratio]
    # phase & gd: 1D. Need to transfor to 2D
    def __init__(self, inp_dim):
        super(MyDense2, self).__init__()
        self.phi2 = self.add_weight('phi2', shape=[inp_dim], initializer=phaseinit2)
        self.gd2 = self.add_weight('gd2', shape=[inp_dim], initializer=gdinit2)
        # self.phi2 = self.add_weight('phi2', shape=[inp_dim], initializer=tf.keras.initializers.HeNormal())
        # self.gd2 = self.add_weight('gd2', shape=[inp_dim], initializer=tf.keras.initializers.HeNormal())        
    
    def call(self, inputlist, training=None):   
        # generating the complex field with phase modulation at each wavelength  
        wvll, inputs = inputlist
        phase0 = self.phi2
        gd = fmax*tf.sigmoid(self.gd2) - fmax/2
        phase0 = (phase0+tf.reverse(phase0, axis = [0]))/2
        gd = (gd+tf.reverse(gd, axis = [0]))/2

        phase0 = OneD2TwoD(phase0, TransMatN)   # Convert the 1D to 2D
        gd = OneD2TwoD(gd, TransMatN)        
        
        phase0 = tf.cast(phase0, dtype=tf.complex64)
        gd = tf.cast(gd, dtype=tf.complex64)
        phase0 = processor3(phase0)
        gd = processor3(gd)
        phase0 = tf.reshape(phase0, [1, NN*ssratio,NN*ssratio])
        gd = tf.reshape(gd, [1, NN*ssratio, NN*ssratio])
        w = 2*pi*(c/wvll)-w0    # angular frenquency differential     
        phase = phase0 + w*gd
        
        phase = tf.math.exp(1j * phase)
        out0 = inputs*phase
        return [wvll, out0]
    
class MyModel(keras.Model):  
    # output: complex field at the output plane at eache wavelength: [NN*ssratio, NN*ssratio]
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(N)
        self.fc2 = MyDense2(N)
    def call(self, inputlist, training=None):             
        out = self.fc1(inputlist)     # modulate the phase at 1st plane
        out = prop2(out, gap, n_sub)  # propagating    
        out = self.fc2(out)     # modulate the phase at 2nd plane        
        out = prop2(out, f, n)      # propagating        
        return out[1]

@tf.function()
def mydense(phi, gd, inputs, wvll):
    # input: unmodulated field at each wavelength: [NN*ssratio, NN*ssratio]
    # output: modulated field at each wavelength: [NN*ssratio, NN*ssratio]
    phi = OneD2TwoD(phi, TransMatN)   # Convert the 1D to 2D
    gd = OneD2TwoD(gd, TransMatN)  
        
    phi = tf.cast(phi, dtype = tf.complex64)
    gd = tf.cast(gd, dtype = tf.complex64)
    phase0 = phi
            
    w = 2*pi*(c/wvll)-w0     # angular frenquency differential
    phase = phase0 + w*gd  
    phase = processor3(phase)
    phase = tf.math.exp(1j * tf.cast(phase, dtype=tf.complex64))
    out = inputs*phase
    return out

def mymodel(phi0, phi1, inputs, wvll):   # 功能：输入总的phi和经过preprocess的输入场，给出输出   
    out = mydense(phi0[0], phi0[1], inputs, wvll)    # modulate the phase at the 1st plane
    out_mid = prop(out, gap, wvll, n_sub)        
    out = mydense(phi1[0], phi1[1], out_mid, wvll)    # modulate the phase at the 2nd plane
    out = prop(out, f, wvll, n)
    return out, out_mid

def int_mask(left, right):      # generating a mask for interation
    # span: 2*(rad-1)+1 pixels
    # x = np.linspace(-int(NN*ssratio/2), int(NN*ssratio/2), NN*ssratio)
    # raa = np.square(x)    

    # span: 2*(rad-1)+1 pixels
    # generating a circle at the center
    x = np.linspace(-int(NN*ssratio/2), int(NN*ssratio/2), NN*ssratio)
    y = x
    xx, yy = np.meshgrid(x, y)
    raa = np.square(xx)+np.square(yy)
    
    mask = np.zeros([NN*ssratio, NN*ssratio])
    rad = int((right-left)/2)
    mask[raa < np.square(rad)] = 1
    # move down the circle
    move = int((right+left)/2-NN*ssratio/2)
    a00 = mask[-move:,:]
    a10 = mask[0:-move,:]
    mask = tf.concat([a00, a10], axis = 0)
    return mask

def efficiency(out):
    outi = out
    outi = np.array(outi)
    lin = outi[:, int(NN*ssratio/2)]
    xx = np.linspace(1, NN*ssratio, NN*ssratio)
    spline = UnivariateSpline(xx, lin-np.max(lin)/2, s=0)
    r = spline.roots() # find the roots
    rad = dlp*(r[-1]-r[0])
    left = int(np.floor(2*r[0]-r[-1]))
    right = int(np.ceil(2*r[-1]-r[0]))
    mask = int_mask(left, right)
    effr = np.sum(mask*outi) / np.sum(outi)
    return effr, rad

def StrehlRatio(out, lad):
    # Calculation of the Strehl ratio of the calculated doublet
    # out: calculated output field
    # lad: calculated wavelength
    fp_theo = focuspoint(Ein[0], lad)
    fp_theo_int = np.square(abs(fp_theo))
    out_int = np.square(abs(out))
    out_int = out_int/np.sum(out_int)*np.sum(fp_theo_int)
    sr = np.max(out_int)/np.max(fp_theo_int)
    return sr

@tf.function()
def compute_loss(fp_t, out):
    # compute the loss by complex field
    # loss = tf.reduce_mean(tf.losses.mean_squared_error(fp_t, out))
    
    # compute the loss by intensity
    # fp_t = fp_t*mask
    # out = out*mask    
    # fp_t = tf.cast(fp_t * tf.math.conj(fp_t), dtype=tf.float32)
    fp_t = tf.cast(fp_t, dtype=tf.float32)
    out = tf.cast(out * tf.math.conj(out), dtype=tf.float32)
    
    # normalization
    # fp_t = fp_t/tf.reduce_max(fp_t)
    # out = out/tf.reduce_max(out)
    # out = out*1.2
    
    loss = tf.reduce_mean(tf.losses.mean_squared_error(fp_t, out))
    return loss

@tf.function()
def train_one_step(model, optimizer, inputlist, etar):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        
        out = model(inputlist)
        loss = compute_loss(etar, out)
        
        # compute gradient
        gradss = tape.gradient(loss, model.trainable_variables)

        # update to weights
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # loss and accuracy is scalar tensor
        return loss, gradss

def train(model, optimizer):
    loss = 0.0
    losses2plot = []
    grads0 = []
    grads1 = []
    grads2 = []
    grads3 = []  
    loss = []
    for step1 in range(int(120*rpt/batchsz)-100):
        dataset_batch = next(dataset_iter)
        condition = dataset_batch['condition']
        condition = tf.io.decode_raw(condition, tf.float32)
        condition = tf.cast(condition, dtype = tf.complex64)
        ##
        incidentfield = dataset_batch['incident']
        incidentfield = tf.io.decode_raw(incidentfield, tf.complex64)
        incidentfield = tf.reshape(incidentfield, [-1, NN*ssratio, NN*ssratio])
        incidentfield = tf.cast(incidentfield, dtype = tf.complex64)
        ##
        inttar = dataset_batch['patt']
        inttar = tf.io.decode_raw(inttar, tf.float32)
        inttar = tf.reshape(inttar, [-1, NN*ssratio, NN*ssratio])
        inttar = tf.cast(inttar, dtype = tf.float32)
        ##
        wvll = condition[:,0]
        wvll = tf.reshape(wvll, [batchsz, 1,1])
        inputlist = [wvll, incidentfield]
        losstmp, gradstmp = train_one_step(model, optimizer, inputlist, inttar)
        loss.append(losstmp)
        grads0.append(gradstmp[0])
        grads1.append(gradstmp[1])        
        grads2.append(gradstmp[2])
        grads3.append(gradstmp[3])        
        
        if (step1 % batchratio == 0) & (step1 >0):
            loss_ba = tf.reduce_mean(loss)
            
            grads_ba0 = tf.reduce_mean(grads0, axis = 0)
            grads_ba1 = tf.reduce_mean(grads1, axis = 0) 
            grads_ba2 = tf.reduce_mean(grads2, axis = 0)
            grads_ba3 = tf.reduce_mean(grads3, axis = 0)             

            grads_ba = []
            grads_ba.append(grads_ba0)
            grads_ba.append(grads_ba1)            
            grads_ba.append(grads_ba2)
            grads_ba.append(grads_ba3)            
            
            optimizer.apply_gradients(zip(grads_ba, model.trainable_variables))
            losses2plot.append(loss_ba)
            phi = model.trainable_variables

            result0 = []
            result1 = []
            result0.append(np.array((phi[0]+tf.reverse(phi[0],axis=[0]))/2))
            result0.append(np.array((phi[1]+tf.reverse(phi[1],axis=[0]))/2))
            result0[1] = fmax*tf.sigmoid(result0[1])-fmax/2
            result1.append(np.array((phi[2]+tf.reverse(phi[2],axis=[0]))/2))
            result1.append(np.array((phi[3]+tf.reverse(phi[3],axis=[0]))/2))
            result1[1] = fmax*tf.sigmoid(result1[1])-fmax/2
            print('RMSE: ', loss_ba.numpy(), '. Step: ', step1)
            np.save('AML-byint-v5.5-NA0.3-p0.28-gap335-r80-cont-focusinair--1-1fs-1DOptimizedIniVal-meta0.npy', result0)
            np.save('AML-byint-v5.5-NA0.3-p0.28-gap335-r80-cont-focusinair--1-1fs-1DOptimizedIniVal-meta1.npy', result1)
            # np.save('GradientTmp.npy', np.array(grads_ba))
            grads0 = []
            grads1 = []
            grads2 = []
            grads3 = []
            loss = []
    return losses2plot

def main():
    model = MyModel()
    optimizer = optimizers.Adam()

    losses2plot = train(model, optimizer)

    x1 = [i for i in range(len(losses2plot))]
    plt.figure()
    plt.plot(x1, losses2plot, color='C0')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.show()
    return losses2plot

if __name__ == '__main__':
    losses = main()
    phi0 = np.load('AML-byint-v5.5-NA0.3-p0.28-gap335-r80-cont-focusinair--1-1fs-1DOptimizedIniVal-meta0.npy')
    phi1 = np.load('AML-byint-v5.5-NA0.3-p0.28-gap335-r80-cont-focusinair--1-1fs-1DOptimizedIniVal-meta1.npy')

    # np.savetxt('NA0.6-p0.32-gap50air-r25.12-phase-meta0.txt',phi0[0])
    # np.savetxt('NA0.6-p0.32-gap50air-r25.12-gd-meta0.txt',phi0[1])
    # np.savetxt('NA0.6-p0.32-gap50air-r25.12-phase-meta1.txt',phi1[0])
    # np.savetxt('NA0.6-p0.32-gap50air-r25.12-gd-meta1.txt',phi1[1])

    # plot the phase at minimum wavelength and group delay
    phi0_2d = []
    phi1_2d = []
    phi0_2d.append(OneD2TwoD(phi0[0], TransMatN))
    phi0_2d.append(OneD2TwoD(phi0[1], TransMatN))
    phi1_2d.append(OneD2TwoD(phi1[0], TransMatN))
    phi1_2d.append(OneD2TwoD(phi1[1], TransMatN))
    
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(np.mod(phi0_2d[0], 2*np.pi), cmap = 'jet')
    plt.title('Phase of metasurface 1 at 450nm')
    plt.subplot(223)
    plt.imshow(phi0_2d[1], cmap = 'jet')
    plt.title('Group delay of metasurface 1')
    plt.subplot(222)
    plt.imshow(np.mod(phi1_2d[0], 2*np.pi), cmap = 'jet')
    plt.title('Phase of metasurface 2 at 450nm')
    plt.colorbar()    
    plt.subplot(224)
    plt.imshow(phi1_2d[1], cmap = 'jet')
    plt.title('Group delay of metasurface 2')
    plt.colorbar()
    
    
    # plot the focal point at a specific wavelength and incident angle
    wvlll = 0.65e-6     # uint: m
    anggg = 0  # uint: degree
    wvlll = tf.cast(wvlll, dtype = tf.complex64)
    anggg = tf.cast(anggg, dtype = tf.complex64)
    anggg = anggg/180*np.pi
    inputs = source2(Ein[0], wvlll, anggg)
    out, out_mid = mymodel(phi0, phi1, inputs, wvlll)
    out = np.array(out)
    outint = np.square(abs(out))
    plt.figure()
    # plt.imshow(outint[2378:5125, 2378:5125], cmap = 'jet')
    plt.imshow(outint, cmap = 'jet')
    plt.colorbar()
    plt.title('Focal spot at'+str(int(1e9*abs(np.array(wvlll))))+' nm, '+str(int(abs(np.array(anggg))/np.pi*180))+' degree.')
    efff, radd = efficiency(outint)
    radd = 1e6*abs(np.array(radd))
    print('Efficiency: ', efff)
    print('FWHM: ', radd, 'um')
    
    
    # Test the efficiency and FWHM at each wavelength
    lambda_test = np.linspace(450, 650, 51)
    # lambda_test = [450,488,535,570,610,650]
    anggg = 0  # uint: degree
    anggg = tf.cast(anggg, dtype = tf.complex64)
    anggg = anggg/180*np.pi
    inputs = source2(Ein[0], wvlll, anggg)    
    lambda_test = np.array(lambda_test)
    lambda_test = 1e-9*lambda_test
    lambda_test = tf.cast(lambda_test, dtype = tf.complex64)
    efff_all = []
    radd_all = []
    outmid_all = []
    radd_theo_all = []
    StrehlR_all = []
    for step in range(len(lambda_test)):
        out, outmidtmp = mymodel(phi0, phi1, inputs, lambda_test[step])
        out = np.array(out)
        StrehlR_all.append(StrehlRatio(out, lambda_test[step]))
        outint = np.square(abs(out))
        efff, radd = efficiency(outint)
        efff_all.append(efff)
        radd_all.append(radd)
        outmid_all.append(outmidtmp)
        # Calculating the diffraction limit
        radd_theo_all.append(abs(1e6*lambda_test[step]/2/NA))        
    refff_all = np.array(efff_all)
    radd_all = abs(np.array(radd_all))*1e6
    plt.figure()
    plt.plot(abs(lambda_test.numpy())*1e9, refff_all)
    plt.title('Efficiency')
    plt.figure()
    plt.plot(abs(lambda_test.numpy())*1e9, radd_all)
    plt.plot(abs(lambda_test.numpy())*1e9, radd_theo_all)
    plt.title('FWHM')
    plt.figure()
    plt.plot(abs(lambda_test.numpy())*1e9, StrehlR_all)    
    plt.title('Strehl ratio')

    # # plot the focusing process at single wavelength
    # wvlll = 0.45e-6     # uint: m
    # anggg = 1  # uint: degree
    # wvlll = tf.cast(wvlll, dtype = tf.complex64)
    # anggg = tf.cast(anggg, dtype = tf.complex64)
    # anggg = anggg/180*np.pi
    # phase0 = tf.cast(phi0[0], dtype = tf.complex64)
    # gd0 = tf.cast(phi0[1], dtype = tf.complex64)
    # phase1 = tf.cast(phi1[0], dtype = tf.complex64)
    # gd1 = tf.cast(phi1[1], dtype = tf.complex64)

    # ww = 2*pi*(c/wvlll)-w0     # angular frenquency differential
    # phase0 = phase0 + ww*gd0
    # phase1 = phase1 + ww*gd1
    # phase0 = OneD2TwoD(phase0, TransMatN)
    # phase1 = OneD2TwoD(phase1, TransMatN2)
    # inputs = source2(Ein[0], wvlll, anggg)
    
    
    # # ## Calculate the propagating process for metasurface 2 only
    # # outxzsw_m2 = focusprocess_meta2only(inputs, wvlll, dlp, f, phase1)
    # # outxzsw_m2 = np.array(outxzsw_m2)
    # # outxzint_m2 = np.square(abs(outxzsw_m2))
    # # plt.figure()
    # # plt.imshow(outxzint_m2[1416:2487, :], cmap = 'jet')    
    # # plt.title('Focussing process'+str(int(1e9*abs(np.array(wvlll))))+' nm, '+str(int(abs(np.array(anggg))/np.pi*180))+' degree.')
       
    # ## Calculate the propagating process
    # outxzsw = focusprocess_2layer(inputs, wvlll, dlp, 1.4*f, phase0, phase1)
    # outxz = np.array(outxzsw)
    # outxzint = np.square(abs(outxzsw))
    # plt.figure()
    # plt.imshow(outxzint, cmap = 'jet')
    # # plt.imshow(outxzint[1485:2418, :], cmap = 'gray')
    # # plt.imshow(outxzint[1416:2487, 0:1072], cmap = 'jet')    
    # plt.title('Focussing process'+str(int(1e9*abs(np.array(wvlll))))+' nm, '+str(int(abs(np.array(anggg))/np.pi*180))+' degree.')
    # plt.plot([int((gap+f)/dlp),int((gap+f)/dlp)],[0,NN*ssratio],'w--')
    
    
    
    ################ calculate the incident angle on the second plane #############
    # field_0_5 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_5_10 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_10_15 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_15_20 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_20_25 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_25_30 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_30_35 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_35_40 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_40_45 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # field_45_50 = np.zeros([NN*ssratio, NN*ssratio, len(lambda_test)], dtype = np.complex64)
    # angle_spectrum = np.zeros([NN*ssratio, NN*ssratio, 10, len(lambda_test)], dtype = np.complex64)
    # ang = np.zeros([NN*ssratio, len(lambda_test)])
    # region_len = int(5e-6/dlp)
    # angle_spectrumint_interp = np.zeros([181, 10, len(lambda_test)], dtype = np.float32)
    # x = np.linspace(-int(NN*ssratio/2), int(NN*ssratio/2), NN*ssratio)
    # xx, yy = np.meshgrid(x, x)
    # raa = np.sqrt(np.square(xx)+np.square(yy))*dlp
    # raa = np.array(abs(raa)*1e6)
    # mask_0_5 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_0_5[raa<5] = 1
    # mask_5_10 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_5_10[raa<10] = 1
    # mask_5_10[raa<=5] = 0
    # mask_10_15 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_10_15[raa<15] = 1
    # mask_10_15[raa<=10] = 0    
    # mask_15_20 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_15_20[raa<20] = 1
    # mask_15_20[raa<=15] = 0    
    # mask_20_25 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_20_25[raa<25] = 1
    # mask_20_25[raa<=20] = 0    
    # mask_25_30 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_25_30[raa<30] = 1
    # mask_25_30[raa<=25] = 0    
    # mask_30_35 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_30_35[raa<35] = 1
    # mask_30_35[raa<=30] = 0  
    # mask_35_40 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_35_40[raa<40] = 1
    # mask_35_40[raa<=35] = 0  
    # mask_40_45 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_40_45[raa<45] = 1
    # mask_40_45[raa<=40] = 0  
    # mask_45_50 = np.zeros([NN*ssratio, NN*ssratio])
    # mask_45_50[raa<50] = 1
    # mask_45_50[raa<=45] = 0
    
    # for i in range(len(lambda_test)):
    #     fxp = np.linspace(-fxmp / 2, fxmp / 2, NN*ssratio, dtype = np.complex64)
    #     ang[:,i] = np.real(np.arcsin(lambda_test[i]*fxp)/np.pi*180)
        
    #     field_0_5 = outmid_all[i]*mask_0_5
    #     angle_spectrum[:,:,0,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_0_5))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,0,i])))
    #     angle_spectrumint_interp[:,0,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_5_10 = outmid_all[i]*mask_5_10
    #     angle_spectrum[:,:,1,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_5_10))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,1,i])))
    #     angle_spectrumint_interp[:,1,i] = tmp(np.linspace(-90, 90,181))        
            
    #     field_10_15 = outmid_all[i]*mask_10_15
    #     angle_spectrum[:,:,2,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_10_15))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,2,i])))
    #     angle_spectrumint_interp[:,2,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_15_20 = outmid_all[i]*mask_15_20
    #     angle_spectrum[:,:,3,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_15_20))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,3,i])))
    #     angle_spectrumint_interp[:,3,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_20_25 = outmid_all[i]*mask_20_25
    #     angle_spectrum[:,:,4,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_20_25))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,4,i])))
    #     angle_spectrumint_interp[:,4,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_25_30 = outmid_all[i]*mask_25_30
    #     angle_spectrum[:,:,5,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_25_30))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,5,i])))
    #     angle_spectrumint_interp[:,5,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_30_35 = outmid_all[i]*mask_30_35
    #     angle_spectrum[:,:,6,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_30_35))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,6,i])))
    #     angle_spectrumint_interp[:,6,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_35_40 = outmid_all[i]*mask_35_40
    #     angle_spectrum[:,:,7,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_35_40))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,7,i])))
    #     angle_spectrumint_interp[:,7,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_40_45 = outmid_all[i]*mask_40_45
    #     angle_spectrum[:,:,8,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_40_45))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,8,i])))
    #     angle_spectrumint_interp[:,8,i] = tmp(np.linspace(-90, 90,181))
            
    #     field_45_50 = outmid_all[i]*mask_45_50
    #     angle_spectrum[:,:,9,i] =  \
    #         np.array(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(field_45_50))))
    #     tmp = interp1d(ang[:,i], np.square(abs(angle_spectrum[int(abs(NN*ssratio/2)),:,9,i])))
    #     angle_spectrumint_interp[:,9,i] = tmp(np.linspace(-90, 90,181))       
    
    # scio.savemat('angle_spectrum.mat', {'angle':angle_spectrumint_interp})
    
    
    