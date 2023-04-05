# -*- coding: utf-8 -*-
"""
Created on Thur July 28 16:00:25 2022
生成单色随机数据集，给透镜优化用
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
import time

radius = tf.constant(25.12e-6, dtype = tf.complex64)
n = tf.constant(1.0, dtype=tf.complex64)
lamb0 = tf.linspace(0.45e-6, 0.65e-6, 120)
# lamb0 = [0.55e-6]
# lamb0 = [0.42e-6, 0.532e-6, 0.633e-6]
lamb0 = tf.cast(lamb0, dtype = tf.complex64)
lamb = lamb0      # 介质中的波长
ssratio = 5 # ss: super sampling

dl = tf.constant(0.32e-6, dtype=tf.complex64)   # pixel size
NA = tf.constant(0.6 , dtype = tf.complex64)     # N.A. of the metalens
inc_ang = tf.constant(0)  # incident angle
inc_ang = tf.cast(inc_ang, dtype = tf.complex64)
inc_ang = inc_ang/180*np.pi

N = int(abs(2*radius/dl))
NN = 409
f = radius/np.tan(np.arcsin(NA))    # focal length
wvlnum = len(lamb0)
# angnum = len(inc_ang)
angnum = 1

pi = tf.convert_to_tensor(np.pi, dtype=tf.complex64)
c = tf.constant(2.99792458e8, dtype=tf.complex64)
w0 = 2*pi*(c/lamb0[0])

dlp = dl / ssratio
fxmp = 1 / dlp
fymp = 1 / dlp
datasetnum = 1000
@tf.function()
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


@tf.function()
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
def focuspoint(ein, lad, z = f, lensphase = None):     # generating focus point field
    if lensphase == None:
        lensphase = sd_lensphase(lad / n)
    lensphase = processor3(lensphase)
    lensphase = tf.math.exp(1j * lensphase)
    fp_t = prop(ein*lensphase, z, lad, n)
    return fp_t

@tf.function()
def preprocess(wvll, angg):
    # Generate the incident field and target image for the specific wavelength and incident angle
    # Output field: intensity
    # unit: m, rad
    
    # ein = source2(Ein, wvll, angg)
    eintmp = source2(Ein, wvll, tf.constant(0, dtype = tf.complex64))    # The normal incident field
    eouttmp = focuspoint(eintmp, wvll) # The field at the focal plane with normal incident
    eouttmp = tf.cast(eouttmp * tf.math.conj(eouttmp), dtype=tf.float32)
    shift_d = (f)*tf.math.tan(angg)
    shift_p = int(tf.math.real(shift_d / dlp))
    
    a0 = tf.zeros([shift_p, NN*ssratio], dtype = tf.float32)
    a1= eouttmp[0:NN*ssratio-shift_p, :]
    etar = tf.concat([a0, a1], axis = 0)
    return etar

@tf.function()
def preprocess2(wvll, angg):
    # Generate the incident field and target image for the specific wavelength and incident angle
    # Output field: amplitude
    # unit: m, rad
    
    # ein = source2(Ein, wvll, angg)
    eintmp = source2(Ein, wvll, tf.constant(0, dtype = tf.complex64))    # The normal incident field
    eouttmp = focuspoint(eintmp, wvll) # The field at the focal plane with normal incident
    shift_d = (f)*tf.math.tan(angg)
    shift_p = int(tf.math.real(shift_d / dlp))
    
    a0 = tf.zeros([shift_p, NN*ssratio], dtype = tf.complex64)
    a1= eouttmp[0:NN*ssratio-shift_p, :]
    etar = tf.concat([a0, a1], axis = 0)
    return etar


# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式。
def _make_example(condition, incident, patt):
    patt = patt.tobytes()
    condition = condition.tobytes()
    incident = incident.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'condition': _bytes_feature(condition),
        'incident': _bytes_feature(incident),        
        'patt': _bytes_feature(patt)
    }))
    return example

# Generating random dataset
# inc_condition_wvl = tf.repeat(lamb0, angnum)
# inc_condition_ang = tf.tile(inc_ang, [wvlnum])

inc_condition_ang = tf.repeat(inc_ang, wvlnum)
inc_condition_wvl = tf.tile(lamb0, [angnum])
db = tf.data.Dataset.from_tensor_slices((inc_condition_wvl, inc_condition_ang))

filename = 'metalens-fixedcondition.r25.12-p0.32-NA0.6-cont-NN409-ssratio5-focusinair'
with tf.io.TFRecordWriter(filename) as writer:
    for step, (wvll, angg) in enumerate(db):
        condition = tf.concat([wvll, angg], axis = 0)
        condition = tf.reshape(condition, [2])
        condition = tf.cast(condition, dtype = tf.float32)
        inputfield = source2(Ein, wvll, angg)
        patt = preprocess(wvll, angg)
        ## 储存数据
        condition = np.array(condition, dtype = np.float32)
        inputfield = np.array(inputfield, dtype = np.complex64)
        patt = np.array(patt, dtype=np.float32)
        example = _make_example(condition, inputfield, patt)
        writer.write(example.SerializeToString())
    print("TFRecord训练文件已保存。")

