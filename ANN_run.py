#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:22:58 2019

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
from plot_gmpe import plotting, plot_az,  plot_AIC, plotsiteterms, plot_dist_curves, plot_mag_curves, residual_histo, setup_test_curves, setup_test_curves_scatter, setup_curves_compare
from create_ANN import fitANN
from sklearn.model_selection import KFold
import glob
import os
from keras.models import model_from_json
from tensorflow.keras.initializers import glorot_uniform
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)

plt.style.use("classic")

sns.set_context("poster")
g = Geod(ellps='clrk66') 

mpl.rcParams['font.size'] =28
sns.set(font_scale=3)
sns.set_style('whitegrid')



seed = 81
np.random.seed(seed)
#tf.set_random_seed(81)

t = '/Users/aklimase/Documents/GMM_ML/catalog/tstar_site.txt'
tstarcat = np.genfromtxt(t, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)
                         
v = '/Users/aklimase/Documents/GMM_ML/catalog/vs30_sta.txt'
vs30cat = np.genfromtxt(v, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)

sitename =  tstarcat['site']
k = tstarcat['tstars']
vs30 = vs30cat['Vs30']

siteparams = [sitename, k, vs30]

#workinghome = '/Users/aklimase/Documents/GMM_ML/model_AICtestvs30_1layer'
#workinghome = '/Users/aklimase/Documents/GMM_ML/mixed_effects_out'

#foldername_list = ['hiddenlayer_2_' + str(i) for i in range(1,4)]
#site_list = ['vs30']*3
#hlayers_list = [1]*3
#hunits_list = np.arange(1,4,1)

bias = True


def runANN(workinghome, foldername,hlayers,hunits_list, site, plot, epochs):
#    AIClist = []
#    sigmalist = []
#    for i in range(len(foldername_list)):
    foldername = foldername
    site = site
    act = 'tanh'
    hlayers = hlayers

    hunits = hunits_list
    
    lr = 0.01
    epochs = epochs
    
    if not os.path.exists(workinghome):
        os.mkdir(workinghome)
    
    if not os.path.exists(workinghome + '/' + foldername):
        os.mkdir(workinghome + '/' + foldername)
        os.mkdir(workinghome + '/' + foldername + '/testing')
        os.mkdir(workinghome + '/' + foldername + '/training')
        os.mkdir(workinghome + '/' + foldername + '/validation')
        os.mkdir(workinghome + '/' + foldername + '/curves')

    
    if site == 'vs30' or site == 'none':
        db = pickle.load(open('/Users/aklimase/Documents/GMM_ML/database_vs30.pckl', 'r'))
    else:
        db = pickle.load(open('/Users/aklimase/Documents/GMM_ML/database_kappa.pckl', 'r'))
    
    if site == 'none':
        d = {'mw': db.mw,'R': db.r,'sta': db.sta, 'pga': np.log(db.pga/9.81), 'elat': db.elat, 'elon': db.elon,'stlat': db.stlat,'stlon': db.stlon}
#        d = {'mw': db.mw,'R': db.r,'mw2': db.mw**2.,'logR': np.log(db.r),'sta': db.sta, 'pga': np.log(db.pga/9.81)}
    
    else:
        d = {'mw': db.mw,'R': db.r,'sta': db.sta,'vs30': db.vs30.flatten(), 'pga': np.log(db.pga/9.81), 'elat': db.elat, 'elon': db.elon,'stlat': db.stlat,'stlon': db.stlon}
    
    df = pd.DataFrame(data=d)
    
    train, test_valid = train_test_split(df, test_size=0.4, random_state = seed)
    valid, test = train_test_split(test_valid, test_size=1/2., random_state = seed)
    
    y_train = train['pga']
    x_train_sta = train['sta']
    x_train_coor = train[['elat', 'elon', 'stlat', 'stlon']]
    x_train =  train.drop(['pga','sta','elat', 'elon', 'stlat', 'stlon'], axis = 1)
    
    y_test = test['pga']
    x_test_sta = test['sta']
    x_test_coor = test[['elat', 'elon', 'stlat', 'stlon']]
    x_test =  test.drop(['pga','sta','elat', 'elon', 'stlat', 'stlon'], axis = 1)
    
    y_valid = valid['pga']
    x_valid_sta = valid['sta']
    x_valid_coor = valid[['elat', 'elon', 'stlat', 'stlon']]
    x_valid =  valid.drop(['pga','sta','elat', 'elon', 'stlat', 'stlon'], axis = 1)
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scale = scaler.transform(x_train)
    x_test_scale = scaler.transform(x_test)
    x_valid_scale = scaler.transform(x_valid)
    
    m = hunits[0]*(x_valid.shape[1] + 1)
    for i in range(len(hunits)-1):
            m += hunits[i+1]*(hunits[i]+1)
    m += 1*(hunits[-1]+1)
    
    ANN = fitANN(act,  hlayers, hunits, lr, epochs, x_train_scale,y_train, x_valid_scale, y_valid, foldername,workinghome)
    
    ANN_test_predicted = ANN.predict(x_test_scale)
    ANN_test_predicted = ANN_test_predicted.flatten()
    test_std = np.std(y_test - ANN_test_predicted)
    print 'ANN std test: ', test_std     
    ANN_valid_predicted = ANN.predict(x_valid_scale)
    ANN_valid_predicted = ANN_valid_predicted.flatten()
    valid_std = np.std(y_valid - ANN_valid_predicted)
    valid_mse = (1./len(y_valid))*np.sum((y_valid-ANN_valid_predicted)**2.)
    AIC = len(y_valid)*np.log(valid_mse)+2*m
    print valid_std
#        AIClist.append(AIC)
#        sigmalist.append(valid_std)
        
    ANN_train_predicted = ANN.predict(x_train_scale)
    ANN_train_predicted = ANN_train_predicted.flatten()
    train_std = np.std(y_train - ANN_train_predicted)
    print train_std

    if plot == True:
        plotting(pre = ANN_test_predicted, obs = np.asarray(y_test), mag = list(x_test['mw']), dist = list(x_test['R']), sta = x_test_sta, foldername = foldername, outname = 'testing', workinghome = workinghome)
        plot_az(pre = ANN_test_predicted, obs = np.asarray(y_test), mag = list(x_test['mw']), dist = list(x_test['R']), sta = x_test_sta, coor = x_test_coor, foldername = foldername, outname = 'testing', workinghome = workinghome)
        plotting(pre = ANN_valid_predicted, obs = np.asarray(y_valid), mag = list(x_valid['mw']), dist = list(x_valid['R']), sta = x_valid_sta, foldername = foldername, outname = 'validation', workinghome = workinghome)
        plot_az(pre = ANN_valid_predicted, obs = np.asarray(y_valid), mag = list(x_valid['mw']), dist = list(x_valid['R']), sta = x_valid_sta, coor = x_valid_coor, foldername = foldername, outname = 'validation', workinghome = workinghome)
        plotting(pre = ANN_train_predicted, obs = np.asarray(y_train), mag = list(x_train['mw']), dist = list(x_train['R']), sta = x_train_sta, foldername = foldername, outname = 'training', workinghome = workinghome)
        plot_az(pre = ANN_train_predicted, obs = np.asarray(y_train), mag = list(x_train['mw']), dist = list(x_train['R']), sta = x_train_sta, coor = x_train_coor, foldername = foldername, outname = 'training', workinghome = workinghome)
        plotsiteterms(obsln = np.asarray(y_test), preln = ANN_test_predicted, sta = x_test_sta, gmpename = 'database_5coeff', foldername = foldername, workinghome = workinghome)
        plotsiteterms(obsln = np.asarray(y_test), preln = ANN_test_predicted, sta = x_test_sta, gmpename = 'database_kappa', foldername = foldername, workinghome = workinghome)
        plotsiteterms(obsln = np.asarray(y_test), preln = ANN_test_predicted, sta = x_test_sta, gmpename = 'database_vs30', foldername = foldername, workinghome = workinghome)

    #write average site residuals to a file to plot ]
    #def a new funtincion

    f = open(workinghome  + '/' +  foldername + '/architecture.txt','w') 
    f.write('activation: ' + act + '\t' + 'layers: ' + str(hlayers)+ '\t' + 'units: ' + str(hunits) + '\t' + 'lr: ' + str(lr) + '\t' +'epochs: ' + str(epochs) + '\n')
    f.write('std of testing data: ' + str(test_std) + '\n')
    f.write('std of validation data: ' + str(valid_std) + '\n')
    f.write('std of training data: ' + str(train_std) + '\n')
    f.close()
    
#    #test,training, validation data
#    resid_train = np.asarray(y_train) - ANN_train_predicted
#    std_train = round(np.std(resid_train),4)
#    mean_train = round(np.mean(resid_train),4)
#    
#    resid_valid = np.asarray(y_valid) - ANN_valid_predicted
#    std_valid = round(np.std(resid_valid),4)
#    mean_valid = round(np.mean(resid_valid),4)
#    
#    resid_test = np.asarray(y_test) - ANN_test_predicted
#    std_test = round(np.std(resid_test),4)
#    mean_test = round(np.mean(resid_test),4)
#    
#    #histo of residuals
#    fig = plt.figure(figsize=(8,8))#tight_layout=True)
#    ax = fig.add_subplot(111)
#    # N is the count in each bin, bins is the lower-limit of the bin
#    labeltrain = r'$\mu_{train}:$ ' + str(mean_train) + '\n' + r'$\sigma_{train}:$ ' + str(std_train)
#    labelvalid = r'$\mu_{valid}:$ ' + str(mean_valid) + '\n' + r'$\sigma_{valid}:$ ' + str(std_valid)
#    labeltest = r'$\mu_{test}:$ ' + str(mean_test) + '\n' + r'$\sigma_{test}:$ ' + str(std_test)
#
#    label=[labeltrain, labelvalid, labeltest]
#    N, bins, patches = ax.hist([resid_train, resid_valid, resid_test],color=['blue', 'green', 'red'], label = label, bins=60)
#    ax.set_xlim(-4,4)
##    ax.text(0.7, 0.9,'std: ' + str(std) + '\n' + 'mean: ' + str(mean),  transform=ax.transAxes)
#    ax.set_xlabel('residuals')
#    ax.set_ylabel('counts')
#    plt.legend(fontsize = 16)
#    plt.title('Residuals')
##    plt.tight_layout()
#    plt.savefig(workinghome + '/' + foldername + '/histoln.png')
#    plt.close()
    
    residual_histo(y_train, y_valid, y_test, ANN_train_predicted, ANN_valid_predicted, ANN_test_predicted, workinghome, foldername, n = '')
    
    
    ##################################################################
    #####gmpe curves
    #Create an array of distances/magnitudes, presribe site effects (in log10 log10 space)
    
    setup_test_curves(site, scaler, workinghome, foldername, siteparams, ANN, n = '')
    
    
#    if plot == False:
#        data = np.transpose([hunits_list, AIClist, sigmalist])  
#        header = 'hiddenunits' + '\t' + 'AIC' + '\t' + 'sigma'
#        np.savetxt(workinghome + '/'+ 'AICsig.txt', data, header = header, delimiter = '\t', fmt = ['%3.0f','%10.3f','%10.3f'])
    
    return AIC, valid_std, test_std, train_std
#        
#foldername = 'kappa_hiddenlayers_3_units_8_10_4_rerun'
#AIC, valid_std, test_std, train_std = runANN(workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN',foldername = foldername, site = 'kappa', hlayers = 3, hunits_list = [8,10,4], plot = True, epochs = 400)



def runANN_k(workinghome, foldername, hlayers,hunits_list, site, plot, epochs, fold, train_model):
#    AIClist = []
#    sigmalist = []
#    for i in range(len(foldername_list)):
    foldername = foldername
    site = site
    act = 'tanh'
    hlayers = hlayers

    hunits = hunits_list
    
    if site == 'vs30':
        vref = 760.
#        vref = 519.
    elif site == 'kappa':
        vref = 0.06
    else:
        vref = 760.
    
    lr = 0.01
    epochs = epochs
    
    if not os.path.exists(workinghome):
        os.mkdir(workinghome)
    
    if not os.path.exists(workinghome + '/' + foldername):
        os.mkdir(workinghome + '/' + foldername)
        os.mkdir(workinghome + '/' + foldername + '/testing')
        os.mkdir(workinghome + '/' + foldername + '/training')
        os.mkdir(workinghome + '/' + foldername + '/validation')
        os.mkdir(workinghome + '/' + foldername + '/curves')
    if not os.path.exists(workinghome + '/' + foldername+ '/curves'):
        os.mkdir(workinghome + '/' + foldername + '/curves')

    
    if site == 'vs30' or site == 'none':
        db = pickle.load(open('/Users/aklimase/Documents/GMM_ML/database_vs30.pckl', 'r'))
    else:
        db = pickle.load(open('/Users/aklimase/Documents/GMM_ML/database_kappa.pckl', 'r'))
    
    if site == '5coeff':
        d = {'mw': db.mw,'R': db.r,'sta': db.sta, 'pga': np.log(db.pga/9.81), 'elat': db.elat, 'elon': db.elon,'stlat': db.stlat,'stlon': db.stlon}
#        d = {'mw': db.mw,'R': db.r,'mw2': db.mw**2.,'logR': np.log(db.r),'sta': db.sta, 'pga': np.log(db.pga/9.81)}
    
    else:
        d = {'mw': db.mw,'R': db.r,'sta': db.sta,'vs30': db.vs30.flatten(), 'pga': np.log(db.pga/9.81), 'elat': db.elat, 'elon': db.elon,'stlat': db.stlat,'stlon': db.stlon}
    
    df = pd.DataFrame(data=d)
    
    train, test_valid = train_test_split(df, test_size=0.4, random_state = seed)
    valid, test = train_test_split(test_valid, test_size=1/2., random_state = seed)
    
    y_train = train['pga']
    x_train_sta = train['sta']
    x_train_coor = train[['elat', 'elon', 'stlat', 'stlon']]
    x_train =  train.drop(['pga','sta','elat', 'elon', 'stlat', 'stlon'], axis = 1)
    
    y_test = test['pga']
    x_test_sta = test['sta']
    x_test_coor = test[['elat', 'elon', 'stlat', 'stlon']]
    x_test =  test.drop(['pga','sta','elat', 'elon', 'stlat', 'stlon'], axis = 1)
    
    y_valid = valid['pga']
    x_valid_sta = valid['sta']
    x_valid_coor = valid[['elat', 'elon', 'stlat', 'stlon']]
    x_valid =  valid.drop(['pga','sta','elat', 'elon', 'stlat', 'stlon'], axis = 1)
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scale = scaler.transform(x_train)
    x_test_scale = scaler.transform(x_test)
    x_valid_scale = scaler.transform(x_valid)
    
    kfold = KFold(n_splits=fold, shuffle=True)
    split = 0
    
    ann_predicted_total_test = []
    ann_predicted_total_valid = []
    ann_predicted_total_train = []
    test_std_kfold = []
    ANN_list = []
    

    
    for train_index, test_index in kfold.split(x_train_scale):
        n = split
        X_traink, X_validk = x_train_scale[train_index], x_train_scale[test_index]
        y_traink, y_validk = np.asarray(y_train)[train_index], np.asarray(y_train)[test_index]


        if train_model == True:
            ANN = fitANN(act, hlayers, hunits, lr, epochs, X_traink, y_traink, X_validk, y_validk, foldername, workinghome)
            model_json = ANN.to_json()
            with open(workinghome + '/' +  foldername  + '/'+ 'model_' + str(n) + '.json', "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
                ANN.save_weights(workinghome + '/' +  foldername  + '/'+ 'model_' + str(n) + '.h5')
                    
        #else the model is trained and saved already
        else:
            json_file = open(workinghome + '/' +  foldername   + '/' + 'model_' + str(n) + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            ANN = model_from_json(loaded_model_json, custom_objects={'GlorotUniform': glorot_uniform()})
            ANN.load_weights(workinghome + '/' +  foldername   + '/' + 'model_' + str(n) + '.h5')    
        
        ANN_list.append(ANN)

        ANN_test_predicted = ANN.predict(x_test_scale)
        ANN_test_predicted = ANN_test_predicted.flatten()
        test_std = np.std(y_test - ANN_test_predicted)
        print 'ANN  ' + str(split) + ' test_std: ' + str(test_std)
        test_std_kfold.append(test_std)
        ann_predicted_total_test.append(ANN_test_predicted)
        
        ANN_valid_predicted = ANN.predict(x_valid_scale)
        ANN_valid_predicted = ANN_valid_predicted.flatten()
        valid_std = np.std(y_valid - ANN_valid_predicted)
        print 'ANN  ' + str(split) + ' valid_std: ' + str(valid_std)
        ann_predicted_total_valid.append(ANN_valid_predicted)
    
#            ANN_train_predictedk = ANN.predict(X_traink)
#            ANN_train_predictedk = ANN_train_predictedk.flatten()
#            train_std = np.std(y_traink - ANN_train_predictedk)   
        ANN_train_predicted = ANN.predict(x_train_scale)
        ANN_train_predicted = ANN_train_predicted.flatten()
        train_std = np.std(y_train - ANN_train_predicted)
        print 'ANN  ' + str(split) + ' train_std: ' + str(train_std)
        ann_predicted_total_train.append(ANN_train_predicted)
        
#        setup_test_curves(site, scaler, workinghome, foldername, siteparams, [ANN], ndir = n)
#        residual_histo(y_traink, y_valid, y_test, ANN_train_predictedk, ANN_valid_predicted, ANN_test_predicted, workinghome, foldername, n=n)
        
        split+=1
        plt.close('all')

    #overall prediction is average of k folds  
    average_pre_test = np.average(ann_predicted_total_test, axis = 0)
    total_std = np.std(y_test - average_pre_test)
    average_pre_valid = np.average(ann_predicted_total_valid, axis = 0)
    valid_std = np.std(y_valid - average_pre_valid)
    average_pre_train = np.average(ann_predicted_total_train, axis = 0)
    train_std = np.std(y_train - average_pre_train)

    
    residual_histo(y_train, y_valid, y_test, average_pre_train, average_pre_valid, average_pre_test, workinghome, foldername, n='')

#    setup_test_curves(site, scaler, workinghome, foldername, siteparams, ANN_list, ndir = '')
#    setup_test_curves_scatter(site, scaler, workinghome, foldername, siteparams, ANN_list, pre_scatter = average_pre_test, obs_scatter = np.asarray(y_test), dist_scatter = np.asarray(x_test['R']), mag_scatter = np.asarray(x_test['mw']), sta = x_test_sta)

#    plot test curves for average site with ann and me both
#    for refsite in sitename:
#        refsite = str(refsite)
##        #vref same for the model
#        setup_curves_compare(site, scaler, workinghome, foldername, siteparams, ANN_list,vref, refsite, pre_scatter = np.asarray(average_pre_test), obs_scatter = np.asarray(y_test), dist_scatter = np.asarray(x_test['R']), mag_scatter = np.asarray(x_test['mw']),sta = x_test_sta)
#        setup_curves_compare(site, scaler, workinghome, foldername, siteparams, ANN_list,vref, refsite, pre_scatter = np.asarray([0]), obs_scatter = np.asarray([0]), dist_scatter = np.asarray([0]), mag_scatter = np.asarray([0]),sta = x_test_sta)

#    plotting(pre = average_pre_test, obs = np.asarray(y_test), mag = list(x_test['mw']), dist = list(x_test['R']), sta = x_test_sta, foldername = foldername, outname = 'testing', workinghome = workinghome)
#    plot_az(pre = average_pre_test, obs = np.asarray(y_test), mag = list(x_test['mw']), dist = list(x_test['R']), sta = x_test_sta, coor = x_test_coor, foldername = foldername, outname = 'testing', workinghome = workinghome)
#    plotting(pre = average_pre_valid, obs = np.asarray(y_valid), mag = list(x_valid['mw']), dist = list(x_valid['R']), sta = x_valid_sta, foldername = foldername, outname = 'validation', workinghome = workinghome)
#    plot_az(pre = average_pre_valid, obs = np.asarray(y_valid), mag = list(x_valid['mw']), dist = list(x_valid['R']), sta = x_valid_sta, coor = x_valid_coor, foldername = foldername, outname = 'validation', workinghome = workinghome)
#    plotting(pre = average_pre_train, obs = np.asarray(y_train), mag = list(x_train['mw']), dist = list(x_train['R']), sta = x_train_sta, foldername = foldername, outname = 'training', workinghome = workinghome)
#    plot_az(pre = average_pre_train, obs = np.asarray(y_train), mag = list(x_train['mw']), dist = list(x_train['R']), sta = x_train_sta, coor = x_train_coor, foldername = foldername, outname = 'training', workinghome = workinghome)
#    plotsiteterms(obsln = np.asarray(y_test), preln = average_pre_test, sta = x_test_sta, gmpename = 'database_5coeff', foldername = foldername, workinghome = workinghome)
#    plotsiteterms(obsln = np.asarray(y_test), preln = average_pre_test, sta = x_test_sta, gmpename = 'database_kappa', foldername = foldername, workinghome = workinghome)
#    plotsiteterms(obsln = np.asarray(y_test), preln = average_pre_test, sta = x_test_sta, gmpename = 'database_vs30', foldername = foldername, workinghome = workinghome)
        
#    f = open(workinghome  + '/' +  foldername + '/kfoldperformance.txt','w') 
#    f.write('std of testing data from average of folds: ' + str(total_std) + '\n')
#    f.write('std of validation data from average of folds: ' + str(valid_std) + '\n')
#    f.write('std of training data from average of folds: ' + str(train_std) + '\n')
#
#    for i in range(fold):
#        f.write('std of testing data fold k = ' + str(i) + ': ' + str(test_std_kfold[i]) + '\n')
#    f.close()
    

        
        
foldername = 'kappa_hiddenlayers_3_units_6_8_6_rerun'
workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN_kfold'
foldername = foldername
site = 'kappa'
hlayers = 3
hunits_list = [6,8,6]
plot = True
epochs = 200
#epochs = 10

    
runANN_k(workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN_kfold',foldername = foldername, site = site, hlayers = 3, hunits_list = hunits_list, plot = False, epochs = epochs, fold = 5, train_model = False)
#
#
##
foldername = 'nosite_hiddenlayers_3_units_4_6_2_rerun'
workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN_kfold'
foldername = foldername
site = '5coeff'
hlayers = 3
hunits_list = [4,6,2]
plot = True
epochs = 200
#    
##foldername = 'nosite_hiddenlayers_3_units_10_8_6_kfold5'
runANN_k(workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN_kfold',foldername = foldername, site = site, hlayers = hlayers, hunits_list = hunits_list, plot = False, epochs = 200, fold = 5, train_model =False)
#
#
#
foldername = 'vs30_hiddenlayers_3_units_8_8_6_rerun'
workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN_kfold'
foldername = foldername
site = 'vs30'
hlayers = 3
hunits_list = [8,8,6]
plot = True
epochs = 200
    
#foldername = 'nosite_hiddenlayers_3_units_10_8_6_kfold5'
runANN_k(workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN_kfold',foldername = foldername, site = site, hlayers = 3, hunits_list = hunits_list, plot = False, epochs = 200, fold = 5, train_model = False)
####


#workinghome = '/Users/aklimase/Documents/GMM_ML/Talapas_run/nostressdrop/top10nosite'
#folderlist = glob.glob(workinghome +  '/hidden*')[6:]
#site = '5coeff'
#
#for f in folderlist:
#    basename = os.path.basename(f)
#    hlayers = int(basename.split('_')[1])
#    hunits_list = basename.split('_')[3:]
#    hunits_list = [int(x) for x in hunits_list]
#    runANN_k(workinghome = workinghome, foldername = basename, site = site, hlayers = hlayers, hunits_list = hunits_list, plot = False, epochs = 200, fold = 5)


#loop through top 10 AICS
#in Talapas run model search




#workinghome = '/Users/aklimase/Documents/GMM_ML/Talapas_run/nostressdrop/top10nosite'
#filelist = glob.glob(workinghome +  '/hidden*/kfold*')
#stdev_test = []
#stdev_valid = []
#units_list = []
#layers_list = []
#
#print 'layers',  '\t','units',  '\t','std_test',  '\t','std_valid'
#for f in filelist:
#    data = np.genfromtxt(f)
#    basename = f.split('/')[-2]
#    hlayers = int(basename.split('_')[1])
#    hunits_list = basename.split('_')[3:]
#    hunits = [int(x) for x in hunits_list]
#    
#    stdev_test.append(data[0][-1])
#    stdev_valid.append(data[1][-1])
#    
#    units_list.append(hunits)
#    layers_list.append(hlayers)
#    
#    print hlayers, '\t', hunits, '\t','\t',round(data[0][-1], 4), '\t','\t', round(data[1][-1], 4)












