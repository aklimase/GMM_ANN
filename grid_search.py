#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:56:36 2019

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
from ANN_run_binnedstressdrop import runANN_k

fold = 5

searchdir = '/Users/aklimase/Documents/GMM_ML/ANN_withstressdrop'

param_grid = [{'hlayers': [1], 'hunits1': [1,2,3,4,6,8,10,12,14]}]
grid1 = ParameterGrid(param_grid)


param_grid = [{'hlayers': [2], 'hunits1': [2,3,4,6,8,10,12,14],'hunits2': [1,2,3,4,6,8,10,12,14]}]
grid2 = ParameterGrid(param_grid)


param_grid = [{'hlayers': [3], 'hunits1': [3,4,6,8,10,12,14],'hunits2': [4,6,8,10,12],'hunits3': [2,3,4,6,8,10,12]}]
grid3 = ParameterGrid(param_grid)

AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

def poolgrid(invar):
#    hlayers = g['hlayers']
#    hunits1 = g['hunits1']
    hlayers, hunits1 = invar['hlayers'], invar['hunits1']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_kappa',foldername = foldername, site = 'kappa', hlayers = hlayers, hunits_list = [hunits1], plot = False, epochs = 5, fold = fold)
#    print hunits1
#    print valid_std, AIC
#    AIClist.append(AIC)
#    validlist.append(valid_std)
#    testlist.append(test_std)
#    trainlist.append(train_std)
#    hlayerslist.append(hlayers)
#    hunitslist.append([hunits1])
    return [hlayers, str(hunits1), AIC, train_std, valid_std, test_std]


from multiprocessing import Pool
pool = Pool(4)
results = pool.map(poolgrid,[g for g in grid1])


#transpose objects

hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist = np.transpose(np.asarray(results, dtype = object))


#for g in grid1:
#    hlayers = g['hlayers']
#    hunits1 = g['hunits1']
#    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1)
#    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_kappa',foldername = foldername, site = 'kappa', hlayers = hlayers, hunits_list = [hunits1], plot = False, epochs = 200, fold = fold)
##    print hunits1
##    print valid_std, AIC
##    AIClist.append(AIC)
##    validlist.append(valid_std)
##    testlist.append(test_std)
##    trainlist.append(train_std)
##    hlayerslist.append(hlayers)
##    hunitslist.append([hunits1])
#    return AIC, valid_std, test_std, train_std, hlayers, hunits1


data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir + '/modelsearch_kappa'+ '/'+ 'AICsig_1layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])
   
    

#foldername = 'hiddenlayers_3_units_8_10_4'
#AIC, valid_std, test_std, train_std = runANN(workinghome = searchdir + '/modelsearch_kappa',foldername = foldername, site = 'kappa', hlayers = 3, hunits_list = [8,10,4], plot = True, epochs = 200)
  
    
AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

    
for g in grid2:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    hunits2 = g['hunits2']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1) + '_' + str(hunits2)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_kappa',foldername = foldername, site = 'kappa', hlayers = hlayers, hunits_list = [hunits1, hunits2], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1,hunits2])
    
data = np.transpose(np.asarray([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist], dtype = object))
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir +'/modelsearch_kappa'+ '/'+ 'AICsig_2layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])

    
    
AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

for g in grid3:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    hunits2 = g['hunits2']
    hunits3 = g['hunits3']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1) + '_' + str(hunits2)+ '_' + str(hunits3)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir +'/modelsearch_kappa',foldername = foldername, site = 'kappa', hlayers = hlayers, hunits_list = [hunits1, hunits2, hunits3], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1,hunits2, hunits3])
    
data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir +'/modelsearch_kappa'+ '/'+ 'AICsig_3layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])
        
###############
#vs30
param_grid = [{'hlayers': [1], 'hunits1': [1,2,3,4,6,8,10,12,14]}]
grid1 = ParameterGrid(param_grid)


param_grid = [{'hlayers': [2], 'hunits1': [2,3,4,6,8,10,12,14],'hunits2': [1,2,3,4,6,8,10,12,14]}]
grid2 = ParameterGrid(param_grid)


param_grid = [{'hlayers': [3], 'hunits1': [3,4,6,8,10,12,14],'hunits2': [4,6,8,10,12],'hunits3': [2,3,4,6,8,10,12]}]
grid3 = ParameterGrid(param_grid)

AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

for g in grid1:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir +'/modelsearch_vs30',foldername = foldername, site = 'vs30', hlayers = hlayers, hunits_list = [hunits1], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1])
    
data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir +'/modelsearch_vs30'+ '/'+ 'AICsig_1layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])
        

AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

    
for g in grid2:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    hunits2 = g['hunits2']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1) + '_' + str(hunits2)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_vs30',foldername = foldername, site = 'vs30', hlayers = hlayers, hunits_list = [hunits1, hunits2], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1,hunits2])
    
data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir + '/modelsearch_vs30'+ '/'+ 'AICsig_2layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])

    
    
AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

for g in grid3:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    hunits2 = g['hunits2']
    hunits3 = g['hunits3']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1) + '_' + str(hunits2)+ '_' + str(hunits3)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_vs30',foldername = foldername, site = 'vs30', hlayers = hlayers, hunits_list = [hunits1, hunits2, hunits3], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1,hunits2, hunits3])
    
data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir + '/modelsearch_vs30'+ '/'+ 'AICsig_3layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])
     
#%%
#no site
param_grid = [{'hlayers': [1], 'hunits1': [1,2,3,4,6,8,10,12,14]}]
grid1 = ParameterGrid(param_grid)

param_grid = [{'hlayers': [2], 'hunits1': [2,3,4,6,8,10,12,14],'hunits2': [1,2,3,4,6,8,10,12,14]}]
grid2 = ParameterGrid(param_grid)

param_grid = [{'hlayers': [3], 'hunits1': [3,4,6,8,10,12,14],'hunits2': [4,6,8,10,12],'hunits3': [2,3,4,6,8,10,12]}]
grid3 = ParameterGrid(param_grid)

AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

for g in grid1:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_nosite',foldername = foldername, site = 'none', hlayers = hlayers, hunits_list = [hunits1], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1])
    
data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir + '/modelsearch_nosite'+ '/'+ 'AICsig_1layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])
        
AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

for g in grid2:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    hunits2 = g['hunits2']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1) + '_' + str(hunits2)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_nosite',foldername = foldername, site = 'vs30', hlayers = hlayers, hunits_list = [hunits1, hunits2], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1,hunits2])
    
data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir + '/modelsearch_nosite'+ '/'+ 'AICsig_2layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])

AIClist = []
testlist = []
validlist = []
trainlist = []
hlayerslist = []
hunitslist = []

for g in grid3:
    hlayers = g['hlayers']
    hunits1 = g['hunits1']
    hunits2 = g['hunits2']
    hunits3 = g['hunits3']
    foldername = 'hiddenlayers_' + str(hlayers) + '_units_' + str(hunits1) + '_' + str(hunits2)+ '_' + str(hunits3)
    AIC, valid_std, test_std, train_std = runANN_k(workinghome = searchdir + '/modelsearch_nosite',foldername = foldername, site = 'vs30', hlayers = hlayers, hunits_list = [hunits1, hunits2, hunits3], plot = False, epochs = 200, fold = fold)
    AIClist.append(AIC)
    validlist.append(valid_std)
    testlist.append(test_std)
    trainlist.append(train_std)
    hlayerslist.append(hlayers)
    hunitslist.append([hunits1,hunits2, hunits3])
    
data = np.transpose([hlayerslist, hunitslist, AIClist, trainlist, validlist, testlist])  
header = 'hiddenlayers' + '\t' + 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma_train' '\t' + 'sigma_valid' '\t' + 'sigma_test'
np.savetxt(searchdir + '/modelsearch_nosite'+ '/'+ 'AICsig_3layer.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%s','%10.3f','%10.3f','%10.3f','%10.3f'])
     



