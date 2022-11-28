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
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.io import savemat, loadmat
from scipy.ndimage import filters
import platform
import multiprocessing
import functools
import json
# from pathos.multiprocessing import ProcessingPool
import random

import math
from scipy.interpolate import RectBivariateSpline as rbs
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, MaxPool1D, Dropout
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Conv1DTranspose, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import keras
from tensorflow.keras import backend as K
import VAE_1DTEM_help as v1h

# %% Froward parameter
ZNumber = 64
dN = 10
mu = 4*np.pi*1e-7
istp = 1
rad = 50
alt = 0
Tldmt = loadmat("receive_Time.mat")
T = Tldmt['tt1']*1E-3
T=np.squeeze(T)

#%% Inversion parameter
InversionNO = 100  # Inversion Case Number
lamda = 0.6e-3
lambda_decay = 0.9
Cost_threshold = 0.001
Delta_cost_threshold = 0.0005
initial_res = np.log10(100)
Nmax = 40

#%% Network & Training parameter
kl_weight = 10E-3  # 1e-2
ssim_w = 4E-2   # 1e-2
H = ZNumber
C = 1
latent_dim = 16
N1 = 30000
initial_rate = 2e-4
patience = 10
factor = 0.2
Max_step = 150
VAENO = 91

#%% Rander parameter
bar_1 = 0
bar_2 = 320

#%% Test Parameter
test_base_no = 0

#%% Main parameter
TRAIN = 1   # -1: Generate data set;  0: Inverse;  1: Train the Network;   2: Only test the network
UID = 'model2D_B_'+str(InversionNO)+"_"+str(lamda)+"_"+str(lambda_decay)+'_VAENO_'+str(VAENO)  # UID is for inversion

sys = platform.system()
if sys == "Windows":
    print("OS is Windows.")
elif sys == "Linux":
    # plt.switch_backend('Agg')
    print("OS is Linux.")
    if TRAIN != -1:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=gpus[3], device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpus[3], enable=True)

plot_x = 4
plot_y = 4
xs = np.linspace(0,plot_x,plot_x+1,dtype='uint8')
zs = np.linspace(0,plot_y,plot_y+1,dtype='uint8')
[xss, zss] = np.meshgrid(xs, zs)

# UID = str(InversionNO)+"_"+str(lamda)+"_"+str(lambda_decay)  # UID is for inversion

if not os.path.exists('./HTEM_VAE_inversion/'+UID):
    os.makedirs('./HTEM_VAE_inversion/'+UID)
    print('./HTEM_VAE_inversion/'+UID+' does not exist! create it.')

urange = np.logspace(-4, 1, 20)    
bar_1 =0
bar_2 =320
h = np.zeros((1, ZNumber))
for ii in range(ZNumber):
    h[0, ii] = 5*math.pow(1.032, ii)  # 64 grids  1000m depth
zEdgeLocation = np.concatenate(([0],np.cumsum(h)))
zElementLocation = 0.5 * (zEdgeLocation[0:-1] + zEdgeLocation[1:])
zLengths = zEdgeLocation[1:]-zEdgeLocation[0:-1]

fieldXStart = 0
fieldXEnd = 10000
XNumberMT = 50
xEdgeLocationMT = np.linspace(fieldXStart, fieldXEnd, XNumberMT + 1)
xElementLocationMT = 0.5 * (xEdgeLocationMT[0:-1] + xEdgeLocationMT[1:])

#%%
'''Training set generation'''############################################
if TRAIN == -1:
    '''Quasi2D_B'''
    ## 0-100
    ## 100-300
    ## 450-600
    Data_set = np.zeros((ZNumber, N1))
    w1 = v1h.fspecial_gaussian(np.array([1, 4]), 4)
    w1 = np.squeeze(w1)
    for pp in range(N1):
        model_ii = np.zeros(ZNumber) +np.random.uniform(50,110)
        u1 = np.random.uniform(0,1)
        u2 = np.random.uniform(0, 1)
        h1 = np.random.uniform(0.1,130)  #
        h2 = np.random.uniform(100, 400)
        h3 = np.random.uniform(450, 650)
        h4 = np.random.uniform(750, 950)
        h4b = np.random.uniform(210, 430, 2)
        h4b = np.sort(h4b)
        # P = h4b.tolist()
        # P.append(np.random.uniform(800, 900))
        # h4b = np.array(P)   #
        c1_ii = [0]
        cc = np.where(abs(zElementLocation - h1) == np.min(abs(zElementLocation - h1)))
        if(u1>0.5):
            cc1 = cc[0][0]
        else:
            cc1 = 0
        model_ii[0:cc1] = np.random.uniform(280, 320)

        cc = np.where(abs(zElementLocation - h2) == np.min(abs(zElementLocation - h2)))
        cc2 = cc[0][0]
        model_ii[cc1:cc2] = np.random.uniform(230, 270)

        cc = np.where(abs(zElementLocation - h3) == np.min(abs(zElementLocation - h3)))
        cc3 = cc[0][0]
        dg = np.random.uniform(180, 220)
        model_ii[cc2:cc3] = dg

        cc = np.where(abs(zElementLocation - h4) == np.min(abs(zElementLocation - h4)))
        cc4 = cc[0][0]
        dg = np.random.uniform(110, 150)
        model_ii[cc3:cc4] = dg

        if(u2 > 0.5 and h2 < 200 and h3 < 550):
            cc = np.where(abs(zElementLocation - h4b[0]) == np.min(abs(zElementLocation - h4b[0])))
            cc5 = cc[0][0]
            cc = np.where(abs(zElementLocation - h4b[1]) == np.min(abs(zElementLocation - h4b[1])))
            cc6 = cc[0][0]
            model_ii[cc5:cc6] = np.random.uniform(60, 100)  # (1,20)

        # if(u2>1):
        #     cc = np.where(abs(zElementLocation - h4) == np.min(abs(zElementLocation - h4)))
        #     cc4 = cc[0][0]
        #     model_ii[cc3:cc4] = np.random.uniform(70, 130)   #
        # else:
        #     cc = np.where(abs(zElementLocation - h4b[0]) == np.min(abs(zElementLocation - h4b[0])))
        #     cc5 = cc[0][0]
        #     pd = np.random.uniform(70, 130)
        #     model_ii[cc3:cc5] = pd
        #
        #     cc = np.where(abs(zElementLocation - h4b[1]) == np.min(abs(zElementLocation - h4b[1])))
        #     cc6 = cc[0][0]
        #     model_ii[cc5:cc6] = np.random.uniform(20, 40)    # (1,20)
        #
        #     cc = np.where(abs(zElementLocation - h4b[2]) == np.min(abs(zElementLocation - h4b[2])))
        #     cc7 = cc[0][0]
        #     model_ii[cc6:cc7] = pd

        model_ii1 = filters.convolve(model_ii, w1)
        Data_set[:, pp] = model_ii1
        print(pp)
    savemat("HTEM_dataset8/dataset.mat",{"Data_set":Data_set})

    Model_training_Ave = np.zeros((ZNumber, XNumberMT))
    for ii in range(XNumberMT):
        Model_training_Ave[:,ii] = Data_set[:, int(ii*N1/XNumberMT)]

    [xw, yw] = np.meshgrid(xEdgeLocationMT, -zEdgeLocation)
    a = np.reshape(Model_training_Ave, (ZNumber, XNumberMT), order='f')
    plt.ion()
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.pcolor(xw, yw, Model_training_Ave, cmap=plt.get_cmap('jet'))
    plt.xlim(0, 1e4)
    plt.ylim(-1000, 0)
    plt.xlabel('Distance (m)')
    plt.ylabel('Depth (m)')
    cbar = plt.colorbar()
    plt.clim(bar_1, bar_2)
    cbar.set_label('Resistivity')
    plt.tight_layout()
    plt.savefig('./HTEM_dataset8/' + 'Some_Training_Models.png')
    plt.ioff()
    plt.close()

# %%
'''Neural Network Set Up'''#####################################################3
if TRAIN != -1:
    class Sampling(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    ## Latent space
    inputs = Input(shape=(H,C), name="inputs")  # 64 1
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
    # x = Activation("tanh", name="outputs")(x)
    # x = LeakyReLU(alpha=0.2)(x)

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

    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                  patience=patience, min_lr=0.000001, verbose=1)
    vae.compile(optimizer=Adam(learning_rate=initial_rate), loss=my_loss_fn, metrics=[mse_loss])
    vae.summary()

#%%
'''Neural Network Training'''
if TRAIN == 1 or TRAIN == 2:
    val_rate = 0.2
    # Data = loadmat("HTEM_dataset8/dataset.mat")
    Data = loadmat("Net-A.mat")
    Data_set = Data['Data_set']    # (ZNumber, N1)
    Data_set = np.log10(Data_set)

    rhoTruth1 = Data_set.transpose((1,0))  #(N1, ZNumber)
    rhoTruth1 = np.expand_dims(rhoTruth1, axis=2)  # (N1, ZNumber, 1)

    indices = list(range(N1))
    # random.shuffle(indices)
    train_ind = indices[:int(N1*(1-val_rate))]
    val_ind = indices[int(N1*(1-val_rate)):]

    rho_train = rhoTruth1[train_ind,:, :]
    rho_test = rhoTruth1[val_ind, :, :]

    batch_size=int(64)
    checkpoint_path = 'HTEM_network' + '/Net_'+str(VAENO)+'/cp.ckpt'
    # checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
if TRAIN == 1:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1,
                                                     save_best_only=True, save_weights_only=True)

    vae.save_weights(checkpoint_path.format(epoch=0))
    history = vae.fit(
        rho_train,
        rho_train,
        epochs=Max_step,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(rho_test, rho_test),callbacks=[cp_callback, lr_schedule]
        )
    # os.listdir(checkpoint_dir)
    import pandas as pd

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    hist_json_file = 'HTEM_network' + '/Net_'+str(VAENO)+'/history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    with open('HTEM_network' + '/Net_'+str(VAENO)+'/history.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)    #
    # or save to csv:
    # hist_csv_file = dir_name + '/history.csv'
    # with open(hist_csv_file, mode='w') as f:
    #     hist_df.to_csv(f)
if TRAIN == 1 or TRAIN == 2:
    checkpoint_path = 'HTEM_network' + '/Net_' + str(VAENO) + '/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    vae.load_weights(latest)
    for jj in range(test_base_no, test_base_no+5):
        rho_test_no5 = int(jj)
        rho_test_no5 = rho_test[rho_test_no5:rho_test_no5 + 1, :, :]
        v_true5 = meanModel(rho_test_no5)  # input can be tensor or numpy, output is numpy
        rho_test_no6 = int(jj+1)
        rho_test_no6 = rho_test[rho_test_no6:rho_test_no6 + 1, :, :]
        v_true6 = meanModel(rho_test_no6)  # input can be tensor or numpy, output is numpy

        # fig = plt.figure(figsize=(5, 5))
        # ax1 = fig.add_subplot(1, 1, 1)
        # plt.plot(10**np.squeeze(rho_test_no5), zElementLocation, color='blue', linewidth=1.5)
        # plt.gca().invert_yaxis()
        # plt.tight_layout()
        #
        # plt.savefig('HTEM_network' + '/Net_'+str(VAENO) + '/model_true_'+str(jj)+'_.png')
        # plt.close()
        #
        # fig = plt.figure(figsize=(5, 5))
        # ax1 = fig.add_subplot(1, 1, 1)
        # plt.plot(10**np.squeeze(rho_test_no6), zElementLocation, color='blue', linewidth=1.5)
        # plt.gca().invert_yaxis()
        # plt.tight_layout()
        # plt.savefig('HTEM_network' + '/Net_'+str(VAENO)+ '/model_true_'+str(jj+1)+'_.png')
        # plt.close()

        for kk in range(11):
            v_5_6 = kk * 0.1 * v_true6 + (10 - kk) * 0.1 * v_true5
            test_pred_y = decoder(v_5_6)
            test_pred_y = K.eval(test_pred_y)
            fig = plt.figure(figsize=(5, 5))
            ax1 = fig.add_subplot(1, 1, 1)
            plt.plot(10**np.squeeze(test_pred_y), zElementLocation, color='blue', linewidth=1.5)
            plt.plot(10 ** np.squeeze(rho_test_no5), zElementLocation, color='red', linewidth=1.5)
            plt.plot(10 ** np.squeeze(rho_test_no6), zElementLocation, color='green', linewidth=1.5)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.xlim(bar_1, bar_2)
            plt.savefig('HTEM_network' + '/Net_'+str(VAENO)+'/No'+str(jj)+'_'+str(kk)+'.png')
            plt.close()

