"""
This code is originally from "https://github.com/studywolf/pydmps" by Studywolf, and modified for DOOSAN Project
More specific, it is for parent class of dynamic movement primitives(DMPs) by Stefan Schaal (2002)

Author: Bukun Son
Start date: 2019.Feb.21
Last date: 2019.Feb.21
"""

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from scipy import io
import matplotlib.pyplot as plt
from keras import layers
from keras import Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import keras
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def canonical(tau, size, dt):
    can_sys = np.zeros(size)

    can_sys[0] = 1
    for t in range(1, size):
        can_sys[t] = can_sys[t - 1] - tau * can_sys[t - 1] * dt

    #plt.plot(can_sys)
    #plt.show()
    return can_sys

def build_model():
    # build in functional api, not sequential model to merge models

    model_in = Input(shape=(1,), name='input')

    dense1 = layers.Dense(128, use_bias=True, bias_initializer='zeros')(model_in)
    #batch1 = layers.BatchNormalization()(dense1)
    act1 = layers.Activation(activation='relu')(dense1)
    dense2 = layers.Dense(128, use_bias=True, bias_initializer='zeros')(act1)
    act2 = layers.Activation(activation='relu')(dense2)
    answer = layers.Dense(2)(act2)

    final_model = Model(model_in, answer)

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    final_model.compile(optimizer = adam, loss='mae', metrics=['mae'])
    print (final_model.summary())

    return final_model

if __name__ == "__main__":

    Kp = 50.0
    Kv = (2*Kp)**0.5
    dt = 0.01
    resam_size = 600
    tau = 1

    demos = io.loadmat("./demos/sample0.mat")['sam0']

    #util = DMP(n_dmps=demos.shape[0], dt=.01, y0=0, goal=1)
    can_sys = canonical(tau, resam_size, dt)

    pos = demos
    dmp_num = pos.shape[0]
    demo_size = pos.shape[1]

    pos_sam = np.zeros((dmp_num,resam_size))
    for i in range(dmp_num):
        resam = scipy.interpolate.interp1d(np.linspace(0,demo_size-1,num=demo_size), pos[i], kind='cubic')
        pos_sam[i,:] = resam(np.linspace(0,demo_size-1,num=resam_size))

    vel_sam = np.gradient(pos_sam,axis=1)/dt
    acc_sam = np.gradient(vel_sam,axis=1)/dt

    xTar = pos_sam[:,-1].reshape(2,1)
    DataDMP = (acc_sam - (np.tile(xTar, resam_size) - pos_sam)*Kp + vel_sam*Kv) / np.tile(can_sys, (dmp_num, 1))

    model = build_model()
    early_stop = EarlyStopping(patience = 10)

    plt.subplot(211)
    plt.plot(DataDMP[0,:])
    plt.subplot(212)
    plt.plot(DataDMP[1, :])
    plt.show()

    history = model.fit({'input': can_sys[0:540]}, DataDMP.reshape(resam_size, dmp_num)[0:540,:],  epochs=100000, shuffle=False, callbacks=[early_stop])

    print("check")
