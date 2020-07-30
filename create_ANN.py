#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:00:08 2019

@author: aklimase
"""

import cPickle as pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import matplotlib as mpl
#from xgboost import XGBRegressor
import inversion as inv
import os
import sys
import seaborn as sns
from pyproj import Geod
from matplotlib.colors import ListedColormap
from plot_gmpe import plotting, plot_az,  plot_AIC, plotsiteterms, plot_dist_curves, plot_mag_curves
#tf.logging.set_verbosity(tf.compat.v1.logging.WARN)


def fitANN(act, hlayers, hunits, lr, epochs, x_train_scale, y_train, x_test_scale, y_test, foldername, workinghome):
	plt.style.use("classic")

	sns.set_context("poster")
	sns.set_style('whitegrid')
	g = Geod(ellps='clrk66') 

	mpl.rcParams['font.size'] = 22
  
    seed = 81
    np.random.seed(81)
#    tf.set_random_seed(81)
    bias = True

    
    def build_model(act, hlayers, hunits, lr, x_train_scale, y_train):#take args
        model = keras.Sequential()
#        model.add(keras.layers.Dense(x_train.shape[1], activation=act,input_shape=(x_train.shape[1],)))

        model.add(keras.layers.Dense(hunits[0], activation=act, input_shape=(x_train_scale.shape[1],), use_bias=bias))

        for l in range(1,hlayers):
            model.add(keras.layers.Dense(hunits[l], activation=act))
        model.add(keras.layers.Dense(1))
    
#        optimizer = tf.train.AdamOptimizer(lr)
        optimizer = tf.train.GradientDescentOptimizer(lr)#tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae'])
        return model
    model = build_model(act = act, hlayers = hlayers, hunits = hunits,lr = lr, x_train_scale = x_train_scale, y_train = y_train)
    
    orig_stdout = sys.stdout
    f = open(workinghome + '/' + foldername + '/' + 'history.txt', 'w')
    sys.stdout = f
    model.summary()        
    # train the model
    EPOCHS = epochs
    #validation_split check
    history = model.fit(x_train_scale, y_train, epochs=EPOCHS,batch_size=32,validation_data=(x_test_scale, y_test))
    sys.stdout = orig_stdout
    f.close()
    
    #serialize model to json and save
    #then can load, read and evaluate

    plt.figure()
    print history.history.keys()
    plt.plot(history.history['mean_absolute_error'], label='training')
    plt.plot(history.history['val_mean_absolute_error'], label='validation')
    plt.ylim([0.5,1])
    plt.xlabel('epochs')
    plt.ylabel('mean abs error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + 'mae.png')
    plt.close()

    return model