# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:30:55 2021

@author: RuiGuo
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:14:17 2021

@author: RuiGuo
"""


import time
import os
import platform
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy.io as sio
from numpy.linalg import norm
from scipy.io import savemat, loadmat
from scipy.ndimage import filters

# from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, MaxPool1D, Dropout
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Conv1DTranspose, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import mnist
import multiprocessing
import functools
import json
# from pathos.multiprocessing import ProcessingPool
import random
from tensorflow.keras import backend as K

import helpers_jointencoding as h_je
import math
import MT2DFWD2 as MT
import Jacobians as jacos
from scipy.interpolate import RectBivariateSpline as rbs

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

TRAIN = 0   ### TRAIN=0: Inversion ### TRAIN=1: TRAIN THENET ### TRAIN=2: test the net
print('TRAIN = {}'.format(TRAIN))
SELECT_AS_INVERSIONMODE = 0 # 0 don't use multiprocerssing
SEL_ACTIFUNC = 1  # what activation function is used. 1 means swish
max_iter = 13
lamda = 0.07e-3   #
lambda_decay = 0.95
beta = 6E-3
beta2 = 0   # Vertical regularization
kl_weight = 10E-3  # 1e-2
ssim_w = 40E-2   # 1e-2
VAE_NO = 81
VAE_NO2 = 81
latent_dim = 16
latent_dim2 = 16
trmono = 0
PINGHUA = 1
N1 = 1  # For training use 30000, for test use 8000
TB = 1   # Two block, another block plus what
bar_1 = np.log10(1)
bar_2 = np.log10(320)
initial_res = np.log10(200)
Cost_threshold = 0.005    # for one block, 1E-3better! # Change convergence logic
Delta_cost_threshold = 0.0005
Use_ref = 1
Add_noise = 1
# dir_name = 'training_rho_middle_dataset_vae_'+str(VAE_NO)   # Network direction
dir_name = '../myHTEM_20220707_for_1D_VAE/HTEM_network' + '/Net_' + str(VAE_NO)
dir_name2 = '../myHTEM_20220707_for_1D_VAE/HTEM_network' + '/Net_' + str(VAE_NO2)
fre1 = 0.5
fre2 = 4
freqNumberMT = 14
UID = 'model20220811b_vae_{:d}&{:d}_lmbd_{:2f}_dcy_{:2f}_beta_{:2f}_ur_{}_AN_{}_noi0.5perc_fre{}-{}_{}_beta2_{}'.\
    format(VAE_NO,VAE_NO2, lamda, lambda_decay,beta,Use_ref,Add_noise,fre1,fre2,freqNumberMT,beta2)
# if TRAIN == 1:
#     from tensorflow.python.framework.ops import disable_eager_execution
#     disable_eager_execution()
if latent_dim == 128:
    plot_x = 8
    plot_y = 16
if latent_dim == 64:
    plot_x=8
    plot_y=8
if latent_dim == 16:
    plot_x=4
    plot_y=4
if latent_dim == 36:
    plot_x=6
    plot_y=6
sys = platform.system()
if sys == "Windows":
    print("OS is Windows.")
elif sys == "Linux":
    # plt.switch_backend('Agg')
    print("OS is Linux.")
    if TRAIN == 1 or TRAIN == 0:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    # else:
    #     cpu = tf.config.list_physical_devices(device_type='CPU')
    #     tf.config.set_visible_devices(devices=cpu[0], device_type='CPU')
        # tf.config.experimental.set_memory_growth(device=cpu[0], enable=False)

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# %% Simulation Setup

fieldXStart = 0
fieldXEnd = 10000
XNumberMT = 100
xEdgeLocationMT = np.linspace(fieldXStart, fieldXEnd, XNumberMT + 1)
xElementLocationMT = 0.5 * (xEdgeLocationMT[0:-1] + xEdgeLocationMT[1:])
ZNumberMT = 80
dh = np.zeros(ZNumberMT)
for i in range(ZNumberMT):
    # dh[i] = 5 * math.pow(1.051,i)  # 5 km
    dh[i] = 2.5 * math.pow(1.047, i)  # 2 km
zEdgeLocationMT = np.concatenate(([0],np.cumsum(dh)))
zElementLocationMT = 0.5 * (zEdgeLocationMT[0:-1] + zEdgeLocationMT[1:])
[xMT, yMT] = np.meshgrid(xElementLocationMT, -zElementLocationMT)
[xElementLengthMT, zElementLengthMT] = np.meshgrid(np.diff(xEdgeLocationMT), np.diff(zEdgeLocationMT))
elementSizeMT = xElementLengthMT * zElementLengthMT
gridNumberMT = ZNumberMT * XNumberMT
timestamp2 = time.time()
domainDepth = zEdgeLocationMT[-1]

XNumberMT1 = 50
xEdgeLocationMT1 = np.linspace(fieldXStart, fieldXEnd, XNumberMT1 + 1)
interpXLocations = 0.5 * (xEdgeLocationMT1[0:-1] + xEdgeLocationMT1[1:])
ZNumberMT1 = 64
dh1 = np.zeros(ZNumberMT1)
for i in range(ZNumberMT1):
    # dh1[i] = 12 * math.pow(1.050,i)  # 5 km
    # dh1[i] = 12 * math.pow(1.009, i)  # 1 km
    dh1[i] = 5 * math.pow(1.032, i)  # 1 km
zEdgeLocationMT1 = np.concatenate(([0],np.cumsum(dh1)))
interpDepths = 0.5 * (zEdgeLocationMT1[0:-1] + zEdgeLocationMT1[1:])
[interX, interZ] = np.meshgrid(interpXLocations, -interpDepths)

# val_rate=1
#
# name="/home/hy-zhou/matlab/rho_vel2sD_8000.mat"
# name = "/home/hy-zhou/matlab/rho_vel/rho_vel0s/rho_vel0s_7400.mat"
# name = "/home/hy-zhou/joint_encoding/simple_set_dataset/simple_set1.mat"
# name = "/home/hy-zhou/joint_encoding/middle_dataset/middle_set1.mat"
# name = "/home/hy-zhou/joint_encoding/middle_dataset/middle_set_outlier2.mat"
# '''是否对训练集平滑'''
# w1 = h_je.fspecial_gaussian(np.array([4, 4]), 4)
#
# rhovels = loadmat(name)
# rhoTruth = rhovels['logFieldRhoInv']
# rhoTruth1 = np.zeros((64, 128, N1))
# for ii in range(N1):
#     VLayeredMatTemp = h_je.interp2_nearest(xMT,yMT,np.reshape(rhoTruth[ii,:],(ZNumberMT, XNumberMT), order='f'), interX, interZ)
#     if PINGHUA:
#         rhoTruth1[:,:,ii] = filters.convolve(VLayeredMatTemp, w1)
#
# rhoTruth1 = rhoTruth1.transpose((2,0,1))
# rhoTruth1 = np.expand_dims(rhoTruth1, axis=3)
#
# indices = list(range(N1))
# # random.shuffle(indices)
# train_ind = indices[:int(N1*(1-val_rate))]
# val_ind = indices[int(N1*(1-val_rate)):]
# #
# # train_ind = np.load('train_ind.npy').tolist()
# # val_ind = np.load('val_ind.npy').tolist()
# rho_train = rhoTruth1[train_ind,:, :,:]
# rho_test = rhoTruth1[val_ind, :, :, :]
# # np.save('train_ind.npy', np.array(train_ind, dtype='uint16'))
# # np.save('val_ind.npy', np.array(val_ind, dtype='uint16'))
#
# del rhoTruth
# del rhoTruth1

# %%  Network1
H = 64
C = 1
## Latent space

inputs = Input(shape=(H,C), name="inputs")  # 64 1
if True:
    x = inputs

    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    # x = MaxPool1D(2)(x) # 32 16

    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = MaxPool1D(2)(x)  # 32 16

    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    # x = MaxPool1D(2)(x) # 16  32

    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = MaxPool1D(2)(x)  # 16  32

    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    # x = MaxPool1D(2)(x) # 8 64

    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = MaxPool1D(2)(x)  # 8 64

    x = Flatten()(x)
    units = x.shape[1]
    z_mean  = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    meanModel = Model(inputs,z_mean)   # 64 - 8
    # encoder_output = Sampling()((z_mean, z_log_var))
    encoder_output = Sampling()((meanModel(inputs), z_log_var))

    encoder = Model(inputs,encoder_output)
    # encoder.compile(optimizer=Adam(1e-3), loss='mse')
    encoder.summary()

    decoder_inputs = Input(shape = (latent_dim), name="decoder_inputs")
    x = Dense(units)(decoder_inputs)
    x = keras.activations.swish(x)
    x = Reshape((8, 64))(x)

    x = Conv1DTranspose(32, 3, strides=2, padding="same")(x) # 16 32
    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1DTranspose(16, 3, strides=2, padding="same")(x) # 32 16
    x = Conv1D(16, 3,  padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1DTranspose(1, 3, strides=2, padding="same")(x) # 64 1
    x = Conv1D(1, 3,  padding="same")(x)

outputs = x
decoder = Model(decoder_inputs, outputs)
decoder.summary()

outputs = decoder(encoder_output)
vae = Model(inputs, outputs)
kl_loss = kl_weight * -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)
# ssim_loss = 1-tf.reduce_mean(tf.image.ssim(inputs,outputs,max_val=1))
# vae.add_loss(ssim_loss)
def my_loss_fn(y_true, y_pred):
    loss1=tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    loss2= 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,max_val=1))
    # loss3 = 1e-2 * -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    return loss1  + ssim_w * loss2

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

def ssim_loss(y_true, y_pred):
    return 1e-2*(1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,max_val=1)))

lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.000001, verbose=1)
vae.compile(optimizer=Adam(learning_rate=5e-4), loss=my_loss_fn, metrics=[mse_loss])
vae.summary()

# %% Network 2
inputs = Input(shape=(H,C), name="inputs")  # 64 1
if True:
    x = inputs

    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    # x = MaxPool1D(2)(x) # 32 16

    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = MaxPool1D(2)(x)  # 32 16

    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    # x = MaxPool1D(2)(x) # 16  32

    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = MaxPool1D(2)(x)  # 16  32

    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    # x = MaxPool1D(2)(x) # 8 64

    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = MaxPool1D(2)(x)  # 8 64

    x = Flatten()(x)
    units = x.shape[1]
    z_mean  = Dense(latent_dim2, name="z_mean")(x)
    z_log_var = Dense(latent_dim2, name="z_log_var")(x)

    meanModel2 = Model(inputs,z_mean)   # 64 - 8
    # encoder_output = Sampling()((z_mean, z_log_var))
    encoder_output = Sampling()((meanModel2(inputs), z_log_var))

    encoder = Model(inputs,encoder_output)
    # encoder.compile(optimizer=Adam(1e-3), loss='mse')
    encoder.summary()

    decoder_inputs = Input(shape = (latent_dim2), name="decoder_inputs")
    x = Dense(units)(decoder_inputs)
    x = keras.activations.swish(x)
    x = Reshape((8, 64))(x)

    x = Conv1DTranspose(32, 3, strides=2, padding="same")(x) # 16 32
    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1DTranspose(16, 3, strides=2, padding="same")(x) # 32 16
    x = Conv1D(16, 3,  padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = keras.activations.swish(x)

    x = Conv1DTranspose(1, 3, strides=2, padding="same")(x) # 64 1
    x = Conv1D(1, 3,  padding="same")(x)

outputs = x
decoder2 = Model(decoder_inputs, outputs)
decoder2.summary()

outputs = decoder2(encoder_output)
vae2 = Model(inputs, outputs)

# %%
batch_size=int(64)

checkpoint_path = dir_name + '/cp.ckpt'
checkpoint_path2 = dir_name2 + '/cp.ckpt'
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1,
                                                 save_best_only=True, save_weights_only=True)

checkpoint_dir2 = os.path.dirname(checkpoint_path2)

if TRAIN == 1:
    vae.save_weights(checkpoint_path.format(epoch=0))
    history = vae.fit(
        rho_train,
        rho_train,
        epochs=200,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(rho_test, rho_test),callbacks=[cp_callback, lr_schedule]
        )
    # os.listdir(checkpoint_dir)
    import pandas as pd

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    hist_json_file = dir_name + '/history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    with open(dir_name + '/history.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)    #
    # or save to csv:
    # hist_csv_file = dir_name + '/history.csv'
    # with open(hist_csv_file, mode='w') as f:
    #     hist_df.to_csv(f)

latest = tf.train.latest_checkpoint(checkpoint_dir)
vae.load_weights(latest)

latest2 = tf.train.latest_checkpoint(checkpoint_dir2)
vae2.load_weights(latest2)
# with open('/trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)

# %% Generate row data
#time_start = time.time()
if TRAIN == 1 or TRAIN == 2:
    for jj in range(5):
        rho_test_no = int(jj)
        rho_test_no1 = rho_test[rho_test_no:rho_test_no+1,:,:,:]
        v_true = meanModel(rho_test_no1) # input can be tensor or numpy, output is numpy
        v1 = v_true
        # v1=tf.constant(v_true)
        test_pred_y = decoder(v1)
        test_pred_y = K.eval(test_pred_y)
        rho_test_no1_p = np.reshape(test_pred_y, [64, 128], order='F')
        # h_je.Plot2DImage(fieldXEnd-fieldXStart, domainDepth, interpXLocations, interpDepths, rho_test_no1, 'mt', [1, 2], 0,
        #                  1, dir_name+'/no', str(rho_test_no)+'.png', rangex=[-2.1,12.9], rangez=[-2,0])
        # h_je.Plot2DImage(fieldXEnd-fieldXStart, domainDepth, interpXLocations, interpDepths, rho_test_no1_p, 'mt', [1, 2], 0,
        #                 1, dir_name+'/no', str(rho_test_no)+'_p.png', rangex=[-2.1,12.9], rangez=[-2,0])

        sig_power = np.average(v_true**2)
        noi_power = 1 * sig_power
        noise = np.random.normal(0, np.sqrt(noi_power), latent_dim)
        # for ii in range(10):
        #     Ka = 0.1*ii+0.1
        #     noise1 = Ka*noise
        #     v1=v_true+noise1
        #     test_pred_y = decoder(v1)
        #     test_pred_y = K.eval(test_pred_y)
        #     rho_test_no1_rand = np.reshape(test_pred_y, [64, 128], order='F')
        #     h_je.Plot2DImage(fieldXEnd-fieldXStart, domainDepth, interpXLocations, interpDepths, rho_test_no1_rand, 'mt', [1, 2], 0,
        #                      1, dir_name+'/no', str(rho_test_no)+'_{:.1f}.png'.format(Ka), rangex=[-2.1,12.9], rangez=[-2,0])

        rho_test_no5 = int(jj)
        rho_test_no5 = rho_test[rho_test_no5:rho_test_no5 + 1, :, :, :]
        v_true5 = meanModel(rho_test_no5)  # input can be tensor or numpy, output is numpy
        rho_test_no6 = int(jj+1)
        rho_test_no6 = rho_test[rho_test_no6:rho_test_no6 + 1, :, :, :]
        v_true6 = meanModel(rho_test_no6)  # input can be tensor or numpy, output is numpy
        h_je.Plot2DImage(fieldXEnd - fieldXStart, domainDepth, interpXLocations, interpDepths, rho_test_no5, 'mt', [bar_1, bar_2],
                         0,
                         1, dir_name + '/no', str(jj) + '.png', rangex=[-2.1, 12.9], rangez=[-2, 0])
        h_je.Plot2DImage(fieldXEnd - fieldXStart, domainDepth, interpXLocations, interpDepths, rho_test_no6, 'mt', [bar_1, bar_2],
                         0,
                         1, dir_name + '/no', str(jj+1) + '.png', rangex=[-2.1, 12.9], rangez=[-2, 0])
        for kk in range(11):
            v_5_6 = kk * 0.1 * v_true6 + (10 - kk) * 0.1 * v_true5
            test_pred_y = decoder(v_5_6)
            test_pred_y = K.eval(test_pred_y)
            rho_test_no1_rand = np.reshape(test_pred_y, [64, 128], order='F')
            h_je.Plot2DImage(fieldXEnd - fieldXStart, domainDepth, interpXLocations, interpDepths, rho_test_no1_rand, 'mt',
                             [bar_1, bar_2], 0,
                             1, dir_name + '/no', str(jj) + '.{:d}.png'.format(kk), rangex=[-2.1, 12.9],
                             rangez=[-2, 0])

    odod=['df']+2

# %%
print('Inversion with Gauss-Newton')
if not os.path.exists('./inversion_results/'+UID):
    os.makedirs('./inversion_results/'+UID)
    print('./inversion_results/'+UID+' does not exist! create it.')

[vG, hG] = h_je.computeGradient(interpXLocations, interpDepths, XNumberMT1, ZNumberMT1)

# DT = np.loadtxt("../myHTEM_20220707_for_1D_VAE/model20220809_clean.modt")
# DT = np.loadtxt("../myHTEM_20220707_for_1D_VAE/model20220811b_clean.modt")
DT = np.loadmat("Model-A.mat")
# DT = np.loadtxt("../myHTEM_20220707_for_1D_VAE/model_2D_B.modt")

Data_set = np.reshape(DT, (ZNumberMT1, XNumberMT1), order='c')
# Data_set[Data_set==10]=100
Data_set_log = np.log10(Data_set)
# beta = np.linalg.norm(vG*np.reshape(Data_set_log, -1, order='f'))**2

v_true_array = np.zeros((1, latent_dim*XNumberMT1))
for jj in range(XNumberMT1):
    true_rho = Data_set_log[:, jj]
    v_true = meanModel(true_rho.reshape([1,ZNumberMT1,1],order='F'))
    v_true = tf.constant(v_true)
    v_true = K.eval(v_true)
    v_true2 = meanModel2(true_rho.reshape([1, ZNumberMT1, 1], order='F'))
    v_true2 = tf.constant(v_true2)
    v_true2 = K.eval(v_true2)
    if (jj <= 23 and jj >= 14):
        v_true_array[:, jj * latent_dim:(jj + 1) * latent_dim] = v_true2
    else:
        v_true_array[:,jj*latent_dim:(jj+1)*latent_dim] = v_true

xs = np.linspace(0,XNumberMT1,XNumberMT1+1,dtype='uint8')
zs = np.linspace(0,latent_dim,latent_dim+1,dtype='uint8')
[xss, zss] = np.meshgrid(xs, zs)

fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(1,1,1)
plt.pcolor(xss, zss, np.reshape(v_true_array, (latent_dim, XNumberMT1), order='f'), cmap=plt.get_cmap('jet'))
# plt.xlim(-1, plot_x-1)
# plt.ylim(-1, plot_y-1)
cbar = plt.colorbar()
plt.clim(np.min(v_true_array), np.max(v_true_array))
plt.tight_layout()
plt.savefig('./inversion_results/'+UID+'/code_true.png')
plt.close()

frequencyMT = np.logspace(fre1, fre2, freqNumberMT)  # 0 Corresponds to 5000m
# frequencyMT = np.logspace(1, 4, freqNumberMT)
# rxIndexMT = np.array(np.linspace(10, 118, 16),dtype='uint32')   # 1-128 total
rxIndexMT = np.array(np.linspace(5, 45, 16),dtype='uint32')   # 1-128 total
rxNumberMT = len(rxIndexMT)
RxMT = [0, 10000]

[ia_temp, ja, value, Ub, Area, index1, Z] = MT.MT2SparseEquationSetUp_zhhy(interpXLocations, interpDepths)

MT2DFWD2_packet = {'freq':frequencyMT,'Field_rho':Data_set_log,'Rx':RxMT,'Field_grid_x':interpXLocations,
                   'Field_grid_z':interpDepths, 'X_number':XNumberMT1,'Z_number':ZNumberMT1,'Rx_index':rxIndexMT}

if SELECT_AS_INVERSIONMODE:
    pool = multiprocessing.Pool(8)
    # pool = ProcessingPool(8)
# MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2, MT2DFWD2_packet), range(len(frequencyMT)))
tmstp4a = time.time()
# print("before optimization MTFWD time = {} s".format(tmstp4a-timestamp4))
MT2DFWD2_packet['ia_temp'] = ia_temp
MT2DFWD2_packet['ja'] = ja
MT2DFWD2_packet['value'] = value
MT2DFWD2_packet['Ub'] = Ub
MT2DFWD2_packet['Area'] = Area
MT2DFWD2_packet['index1'] = index1
MT2DFWD2_packet['Z'] = Z

# %%
if SELECT_AS_INVERSIONMODE:
    MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_packet), range(len(frequencyMT)))
else:
    MT2DFWD2_back = []
    for ii in range(len(frequencyMT)):
        bci = MT.MT2DFWD2_zhhy(MT2DFWD2_packet, ii)
        MT2DFWD2_back.append(bci)
print('Successfully computed forward problem')
FieldData = np.zeros(len(rxIndexMT)*2*len(frequencyMT))
for i in range(len(frequencyMT)):
    FieldData[i*len(rxIndexMT):(i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][:len(rxIndexMT)]
    FieldData[len(rxIndexMT)*len(frequencyMT) + i*len(rxIndexMT):len(rxIndexMT)*len(frequencyMT) + (i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][len(rxIndexMT):]

obRhoAmpliAct = FieldData[0:int(len(FieldData)/2)]
obRhoPhaseAct = FieldData[int(len(FieldData) / 2):]

# Add Noise
FieldData_noise = 0.005*FieldData*np.random.normal(size=(np.shape(FieldData)))
FieldData = FieldData + FieldData_noise*Add_noise

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
[xss1, zss1] = np.meshgrid(interpXLocations[rxIndexMT], np.log10(frequencyMT))
plt.pcolor(xss1, zss1, np.reshape(FieldData[:rxNumberMT*freqNumberMT], (freqNumberMT,16), order='c'), cmap=plt.get_cmap('jet'))
cbar = plt.colorbar()
plt.clim(np.min(obRhoAmpliAct), np.max(obRhoAmpliAct))
plt.tight_layout()
plt.savefig('./inversion_results/'+UID+'/data_true.png')
plt.close()

h_je.Plot2DImage(fieldXEnd-fieldXStart, domainDepth, interpXLocations, interpDepths, Data_set_log, 'mt', [bar_1, bar_2], 0,
                 1, './inversion_results/'+UID+'/', 'model_true.png', rangex=[0, 10],
                             rangez=[-1, 0])

v_array = np.zeros((1, XNumberMT1*latent_dim))
# tensorData = tf.convert_to_tensor(numpyData, dtype=tf.float32)
rho_recon = initial_res * np.ones(ZNumberMT1, dtype=np.float32)
v = meanModel(rho_recon.reshape([1,ZNumberMT1,1],order='F'))
v2 = meanModel2(rho_recon.reshape([1,ZNumberMT1,1],order='F'))
v = tf.constant(v)
v_np = K.eval(v)
v2 = tf.constant(v2)
v_np2 = K.eval(v2)
for jj in range(XNumberMT1):
    if (jj <= 23 and jj >= 14):
        v_array[:, jj * latent_dim:(jj + 1) * latent_dim] = v_np2
    else:
        v_array[:,jj*latent_dim:(jj+1)*latent_dim] = v_np

# v_start_array = K.eval(v)
plt.ion()
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(1,1,1)
plt.pcolor(xss, zss, np.reshape(v_array, (latent_dim,XNumberMT1), order='f'), cmap=plt.get_cmap('jet'))
# plt.xlim(-1, plot_x-1)
# plt.ylim(-1, plot_y-1)
cbar = plt.colorbar()
plt.clim(np.min(v_true_array), np.max(v_true_array))
plt.tight_layout()
plt.savefig('./inversion_results/'+UID+'/code_start.png')
plt.ioff()
plt.close()

rho_recon_pred_ii = decoder(v) # tensor
rho_recon_pred_ii2 = decoder2(v2) # tensor
rho_recon_pred = np.zeros((ZNumberMT1, XNumberMT1))
for jj in range(XNumberMT1):
    if(jj <= 23 and jj >=14):
        rho_recon_pred[:, jj] = np.reshape(rho_recon_pred_ii2, -1)
    else:
        rho_recon_pred[:, jj] = np.reshape(rho_recon_pred_ii, -1)
# rho_recons_img = np.reshape(rho_recon_pred, [64, 128], order='F')
h_je.Plot2DImage(fieldXEnd-fieldXStart, domainDepth, interpXLocations, interpDepths, rho_recon_pred, 'mt', [bar_1, bar_2], 0,
                 1, './inversion_results/'+UID+'/', 'model_start.png', rangex=[0, 10],
                             rangez=[-1, 0])

Cost = [1]
MT2DFWD2_packet['Field_rho'] = rho_recon_pred
if SELECT_AS_INVERSIONMODE:
    MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_packet), range(len(frequencyMT)))
else:
    MT2DFWD2_back = []
    for ii in range(len(frequencyMT)):
        bci = MT.MT2DFWD2_zhhy(MT2DFWD2_packet, ii)
        MT2DFWD2_back.append(bci)
newData = np.zeros(len(rxIndexMT)*2*len(frequencyMT))
EFieldVectorf = np.zeros((XNumberMT1*ZNumberMT1,len(frequencyMT)),dtype='complex')
EobsVector = np.zeros(len(rxIndexMT)*len(frequencyMT), dtype='complex')
HobsVector = np.zeros(len(rxIndexMT)*len(frequencyMT),dtype='complex')
for i in range(len(frequencyMT)):
    newData[i*len(rxIndexMT):(i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][:len(rxIndexMT)]
    newData[len(rxIndexMT)*len(frequencyMT) + i*len(rxIndexMT):len(rxIndexMT)*len(frequencyMT) + (i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][len(rxIndexMT):]
    EobsVector[i*len(rxIndexMT):(i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['Eobs_in']
    HobsVector[i*len(rxIndexMT):(i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['Hobs_in']
    EFieldVectorf[:,i] = MT2DFWD2_back[i]['EFieldVector_in']

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
[xss1, zss1] = np.meshgrid(interpXLocations[rxIndexMT], np.log10(frequencyMT))
plt.pcolor(xss1, zss1, np.reshape(newData[0:int(len(newData)/2)], (freqNumberMT,16), order='c'), cmap=plt.get_cmap('jet'))
cbar = plt.colorbar()
plt.clim(np.min(obRhoAmpliAct), np.max(obRhoAmpliAct))
plt.tight_layout()
plt.savefig('./inversion_results/'+UID+'/data_start.png')
plt.close()

res =  FieldData - newData
Ck = norm(res) ** 1 / norm(FieldData) ** 1
# Cost.append(Ck1)
# print('Iter # {}, C = {}'.format(1, Ck1))

lamdavh = 0e-4
v_ref = np.zeros((1, latent_dim*XNumberMT1))
for jj in range(max_iter):
    [jacobianMTRho, jacobianMTPhi] = jacos.ComputeJacobianFunc_z(EobsVector, HobsVector, EFieldVectorf, frequencyMT,
                                                                 np.reshape(rho_recon_pred,(ZNumberMT1, XNumberMT1),order='f'), RxMT,
                                                                 interpXLocations, interpDepths, XNumberMT1, ZNumberMT1,
                                                                 rxIndexMT, ia_temp, ja, value, Area, index1)
    J = np.concatenate((jacobianMTRho, jacobianMTPhi), axis=0)  # [Ndata, Nmodel]

    lamda = lamda * lambda_decay

    gk = np.zeros(latent_dim*XNumberMT1)
    Hk = np.eye(latent_dim*XNumberMT1)*lamda*Ck
    JD = np.zeros((XNumberMT1*ZNumberMT1, XNumberMT1*latent_dim))
    for pp in range(XNumberMT1):
        print("Ite = {}, x_no = {}".format(jj, pp))
        J_slice = J[:, pp*ZNumberMT1:(pp+1)*ZNumberMT1]  # [Ndata, Nmodel]
        v = v_array[:, pp*latent_dim:(pp+1)*latent_dim]
        v = tf.convert_to_tensor(v, dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False,persistent=True) as g:
            g.watch(v)
            if (pp <=23 and pp >= 14):
                rho_recon_pred_ii = decoder2(v)
            else:
                rho_recon_pred_ii = decoder(v)
        
        jacb = tf.squeeze(g.jacobian(rho_recon_pred_ii,v)).numpy()
    # jacb = K.eval(tf.squeeze(g.jacobian(rho_recon_pred,v)))
        jacb1 = np.reshape(jacb,[ZNumberMT1, latent_dim],order='F') # [Nmodel, N_latent_z]

        JD[pp*ZNumberMT1:(pp+1)*ZNumberMT1, pp*latent_dim:(pp+1)*latent_dim] = jacb1

        J2 = np.matmul(J_slice,jacb1)
    # JLv = np.matmul(Lv,jacb1)
    # JLh = np.matmul(Lh,jacb1)
    # lamda = lamda * 0.8
        v_ref_ii = v_ref[:, pp*latent_dim:(pp+1)*latent_dim]*Use_ref
        Hk[pp*latent_dim:(pp+1)*latent_dim, pp*latent_dim:(pp+1)*latent_dim] = \
            Hk[pp*latent_dim:(pp+1)*latent_dim, pp*latent_dim:(pp+1)*latent_dim] + np.matrix(J2).H.dot(J2)\
            / norm(FieldData) ** 2
        gk[pp*latent_dim:(pp+1)*latent_dim] = -np.matrix(J2).H.dot(res) / norm(FieldData) ** 2  + lamda * Ck*(v-v_ref_ii)
        # gk = -2 * np.matrix(J2).H.dot(res) / norm(FieldData) ** 2  + lamda * Ck1*v
        # Hk = 2 * np.matrix(J2).H.dot(J2) / norm(FieldData) ** 2  + lamda * Ck1 * np.eye(latent_dim)

    rho_recon_pred_array = np.reshape(rho_recon_pred, -1, order='f')
    hG_JD = hG*JD
    hG_m = hG*rho_recon_pred_array
    vG_JD = vG*JD
    vG_m = vG*rho_recon_pred_array
    gk = gk + beta * Ck*np.dot(np.matrix(hG_JD).H, hG_m) + beta2 * Ck*np.dot(np.matrix(vG_JD).H, vG_m)
    Hk = Hk + beta * Ck*np.dot(np.matrix(hG_JD).H, hG_JD) + beta2 * Ck*np.dot(np.matrix(vG_JD).H, vG_JD)
    # print("To here.")
    time1=time.time()
    pk = -np.linalg.solve(Hk, gk.T)
    time2=time.time()
    print('time：' + str(time2 - time1) + 's')
    a = 1
    v_array_1 = v_array + a * np.transpose(np.real(pk))
        # print(v1)
    for hh in range(XNumberMT1):
        v1 = tf.convert_to_tensor(v_array_1[:, hh*latent_dim:(hh+1)*latent_dim])
        if(hh <= 23 and hh >=14):
            rho_recon_pred_ii = decoder2(v1)
        else:
            rho_recon_pred_ii = decoder(v1)
        rho_recon_pred[:, hh] = np.squeeze(rho_recon_pred_ii)
    # rho_recon_pred_img = np.reshape(rho_recon_pred,[64, 128],order='F')
    MT2DFWD2_packet['Field_rho'] = rho_recon_pred
    if SELECT_AS_INVERSIONMODE:
        MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_packet), range(len(frequencyMT)))
    else:
        MT2DFWD2_back = []
        for ii in range(len(frequencyMT)):
            bci = MT.MT2DFWD2_zhhy(MT2DFWD2_packet, ii)
            MT2DFWD2_back.append(bci)
    newData = np.zeros(len(rxIndexMT)*2*len(frequencyMT))
    EFieldVectorf = np.zeros((XNumberMT1*ZNumberMT1,len(frequencyMT)),dtype='complex')
    EobsVector = np.zeros(len(rxIndexMT)*len(frequencyMT), dtype='complex')
    HobsVector = np.zeros(len(rxIndexMT)*len(frequencyMT),dtype='complex')
    for i in range(len(frequencyMT)):
        newData[i*len(rxIndexMT):(i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][:len(rxIndexMT)]
        newData[len(rxIndexMT)*len(frequencyMT) + i*len(rxIndexMT):len(rxIndexMT)*len(frequencyMT) + (i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][len(rxIndexMT):]
        EobsVector[i*len(rxIndexMT):(i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['Eobs_in']
        HobsVector[i*len(rxIndexMT):(i+1)*len(rxIndexMT)] = MT2DFWD2_back[i]['Hobs_in']
        EFieldVectorf[:,i] = MT2DFWD2_back[i]['EFieldVector_in']
    res = FieldData - newData
    v_array = v_array_1
    Ck = norm(res) ** 1 / norm(FieldData) ** 1

    # 20220720
    ####
    v_ref = np.zeros((1, latent_dim*XNumberMT1))
    for gg in range(XNumberMT1):
        v_gg = meanModel(np.reshape(rho_recon_pred[:, gg], (1,ZNumberMT1, 1), order='f'))
        v_gg = tf.constant(v_gg)
        v_gg = K.eval(v_gg)
        v_ref[:, gg*latent_dim:(gg+1)*latent_dim] = v_gg
    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.pcolor(xss, zss, np.reshape(v_ref, (latent_dim, XNumberMT1), order='f'), cmap=plt.get_cmap('jet'))
    # plt.xlim(-1, plot_x-1)
    # plt.ylim(-1, plot_y-1)
    cbar = plt.colorbar()
    plt.clim(np.min(v_true_array), np.max(v_true_array))
    plt.tight_layout()
    plt.savefig('./inversion_results/' + UID + '/code_map_ite_{}.png'.format(jj + 1))
    plt.ioff()
    plt.close()
    # 20220720

    # 20220703: Try qiongsou step length?
    # Line Search######################################################
    ls_num = 1
    while (Ck > Cost[-1] and ls_num < 6):
        # cost_propose = 0.5 * norm(Wd * (d_true - d_propose)')^2+0.5*lambda*norm(Wm*m_propose') ^ 2;
        a1 = -0.5 * a ** 2 * (np.dot(gk,pk))/(Ck-Cost[-1]-a*np.dot(gk,pk))
        if (a1 < 0.01 * a):
            a1 = 0.01 * a
        a = a1
        print(['ite={}, line search, a={}'.format(jj, a)])
        v1_propose = v_array + a * np.transpose(np.real(pk))
        rho_recon_pred_propose = np.zeros((ZNumberMT1, XNumberMT1))
        for hh in range(XNumberMT1):
            v1 = tf.convert_to_tensor(v1_propose[:, hh * latent_dim:(hh + 1) * latent_dim])
            if(hh <=23 and hh >= 14):
                rho_recon_pred_ii = decoder2(v1)
            else:
                rho_recon_pred_ii = decoder(v1)
            rho_recon_pred_propose[:, hh] = np.squeeze(rho_recon_pred_ii)
        MT2DFWD2_packet['Field_rho'] = rho_recon_pred_propose
        if SELECT_AS_INVERSIONMODE:
            MT2DFWD2_back = pool.map(functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_packet), range(len(frequencyMT)))
        else:
            MT2DFWD2_back = []
            for ii in range(len(frequencyMT)):
                bci = MT.MT2DFWD2_zhhy(MT2DFWD2_packet, ii)
                MT2DFWD2_back.append(bci)
        newData_propose = np.zeros(len(rxIndexMT) * 2 * len(frequencyMT))
        EFieldVectorf = np.zeros((XNumberMT1 * ZNumberMT1, len(frequencyMT)), dtype='complex')
        EobsVector = np.zeros(len(rxIndexMT) * len(frequencyMT), dtype='complex')
        HobsVector = np.zeros(len(rxIndexMT) * len(frequencyMT), dtype='complex')
        for i in range(len(frequencyMT)):
            newData_propose[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['data_f'][:len(rxIndexMT)]
            newData_propose[
            len(rxIndexMT) * len(frequencyMT) + i * len(rxIndexMT):len(rxIndexMT) * len(frequencyMT) + (i + 1) * len(
                rxIndexMT)] = MT2DFWD2_back[i]['data_f'][len(rxIndexMT):]
            EobsVector[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['Eobs_in']
            HobsVector[i * len(rxIndexMT):(i + 1) * len(rxIndexMT)] = MT2DFWD2_back[i]['Hobs_in']
            EFieldVectorf[:, i] = MT2DFWD2_back[i]['EFieldVector_in']
        res = FieldData - newData_propose
        Ck_propose = norm(res) ** 1 / norm(FieldData) ** 1
        ls_num = ls_num + 1

        if (Ck_propose < Cost[-1] or ls_num == 6):
            newData = newData_propose
            newData_log = np.log10(newData)
            v = v1_propose
            Ck = Ck_propose
            rho_recon_pred = rho_recon_pred_propose
            rho_recon_pred_1D = np.reshape(rho_recon_pred_propose, -1)
            break

    #########################################3
    print('Iter # {}, C = {}'.format(jj + 1, Ck))
    Cost.append(Ck)
    print('Relative data misfit = ')
    print(Cost)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.pcolor(xss1, zss1, np.reshape(newData[0:int(len(newData) / 2)], (freqNumberMT, 16), order='c'),
               cmap=plt.get_cmap('jet'))
    cbar = plt.colorbar()
    plt.clim(np.min(obRhoAmpliAct), np.max(obRhoAmpliAct))
    plt.tight_layout()
    plt.savefig('./inversion_results/' + UID +'/data_ite_{}.png'.format(jj+1))
    plt.close()

    # v_recons_array = K.eval(v1)
    plt.ion()
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    plt.pcolor(xss, zss, np.reshape(np.array(v_array), (latent_dim, XNumberMT1), order='f'), cmap=plt.get_cmap('jet'))
    # plt.xlim(-1, plot_x-1)
    # plt.ylim(-1, plot_y-1)
    cbar = plt.colorbar()
    plt.clim(np.min(v_true_array), np.max(v_true_array))
    plt.tight_layout()
    plt.savefig('./inversion_results/'+UID+'/code_ite_{}.png'.format(jj+1))
    plt.ioff()
    plt.close()

    h_je.Plot2DImage(fieldXEnd-fieldXStart, domainDepth, interpXLocations, interpDepths, rho_recon_pred, 'mt', [bar_1, bar_2], 0,
                     1, './inversion_results/'+UID+'/', 'model_ite_{}.png'.format(jj+1), rangex=[0, 10],
                             rangez=[-1, 0])
    savemat('./inversion_results/' + UID + '/No.' + str(jj+1) + ' Resistivity Model.mat', {'model': rho_recon_pred})
    if Cost[-1] < Cost_threshold and Cost[-2]-Cost[-1]<Delta_cost_threshold:
        break

Cost_array = np.array(Cost, dtype='float32')
np.savetxt('./inversion_results/'+UID+'/cost.txt', Cost_array, '%.5f')
savemat('./inversion_results/'+UID+'/final_model.mat',{'xs':interpXLocations,'zs':interpDepths,'value':rho_recon_pred})
# def kl_loss(y_true, y_pred):
#     return 1e-2 * -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=8e-4,
#     decay_steps=100,
#     decay_rate=0.5)
