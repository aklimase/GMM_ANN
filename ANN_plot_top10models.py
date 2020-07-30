#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:30:08 2019

@author: aklimase
"""
import glob
from keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
import pandas as pd
import cPickle as pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import binned_statistic_dd
from numpy import ones,r_,c_


plt.style.use("classic")

sns.set_context("poster")
sns.set_style('whitegrid')
#g = Geod(ellps='clrk66') 

mpl.rcParams['font.size'] = 22

site = 'vs30'
fold = 5


if site == 'vs30':
    vref = 760.
elif site == 'kappa':
    vref = 0.06
else:
    vref = 760.


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

train, test_valid = train_test_split(df, test_size=0.4)
valid, test = train_test_split(test_valid, test_size=1/2.)

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


#plot model curves for given mag, distance on the same axis

#take in a list of models, read in the k fold model files
#make predictions per model and plot on axes
#workinghome = '/Users/aklimase/Documents/GMM_ML/Talapas_run/nostressdrop/top10' + site
#modellist = glob.glob(workinghome + '/hidden*')
#ANN_top10 = []
#cmap = plt.set_cmap('cividis')
#cNorm  = mpl.colors.Normalize(vmin=-13, vmax=-2)

workinghome = '/Users/aklimase/Documents/GMM_ML/best_ANN_kfold'
modellist = ['/Users/aklimase/Documents/GMM_ML/best_ANN_kfold/vs30_hiddenlayers_3_units_8_8_6_rerun']
ANN_top10 = []
cmap = plt.set_cmap('cividis')
cNorm  = mpl.colors.Normalize(vmin=-7, vmax=-1)


for m in modellist:
    #read in model files
    ANN_list = []

    foldername = m.split('/')[-1]
    for i in range(fold):
        json_file = open(m + '/' + 'model_' + str(i) + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ANN = model_from_json(loaded_model_json, custom_objects={'GlorotUniform': glorot_uniform()})
        ANN.load_weights(workinghome + '/' +  foldername   + '/' + 'model_' + str(i) + '.h5')
        ANN_test_predicted = ANN.predict(x_test_scale)
        ANN_list.append(ANN)    
#        setup_curves_compare(site, scaler, workinghome, foldername, siteparams, ANN_list, vref, pre_scatter = np.asarray([0]), obs_scatter = np.asarray([0]), dist_scatter = np.asarray([0]), mag_scatter = np.asarray([0]))
    ANN_top10.append(ANN_list)#the 5 fold models

    mlist = np.linspace(2.8,5.0,200)
    Rlist = np.linspace(10.,225.,200)
    X, Y = np.meshgrid(Rlist, mlist)
    Z = np.zeros((len(X), len(Y)))

    for i in range(len(X)):
        if site == '5coeff':
            d = {'mw': Y[i][0], 'R': X[i]}    
        else:
            d = {'mw': Y[i][0], 'R': X[i], 'vs30': vref}    

        df = pd.DataFrame(data=d)
        x_curve_scale = scaler.transform(df)    
        pre_list = [ANN_list[k].predict(x_curve_scale) for k in range(fold)]
        avg_pre = np.average(pre_list, axis = 0)
        
        #ln to log10
        avg_pre = avg_pre*np.log10(np.e)
        
        Z[i] = avg_pre.flatten()
    
    X = np.log10(X)
    fig = plt.figure(figsize=(12,10))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    plt.xlim([1,np.log10(225.)])
    plt.ylim([2.8,5.0])
 
    cp = plt.contourf(X, Y, Z, cmap = cmap, norm=cNorm, levels = np.arange(-6,0+1,0.5))

    cl = plt.contour(X, Y, Z, colors = 'k', levels = np.arange(-6,0+1,0.5))
    plt.clabel(cl, colors = 'k', fmt = '%2.1f',fontsize=18, inline=True)

    cbar = plt.colorbar(cp, norm=cNorm)
    cbar.ax.set_ylabel('predicted ln(PGA)')
    
    ax.set_title('Contour Plot')
    ax.set_xlabel('R (km)')
    ax.set_ylabel('mw')
    plt.show()
    plt.savefig(workinghome + '/' + foldername + '/' + 'contour.png')
    plt.close()


#create model curves for folds of all top 10 models
#y_curve = []
#for i in range(len(ANN_top10)):
#    ANN_list = ANN_top10[i]
#
#    mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
#    dist = np.arange(10.,225.,0.5)
#    
#    mag_curve = np.array([len(dist)*[x] for x in mag_test])
#    dist_curve = np.array([dist for i in range(len(mag_test))])
#
#    for i in range(len(mag_test)):
#        ###################    
#        if site == '5coeff':
#            d = {'mw': mag_curve[i], 'R': dist_curve[i]} 
#        else:
#            d = {'mw': mag_curve[i], 'R': dist_curve[i], 'vs30': vref*np.ones(len(mag_curve[i]))}    
#
#        df = pd.DataFrame(data=d)
#        x_curve_scale = scaler.transform(df)
#        
#        pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(fold)]
#        avg_pre = np.average(pre_list, axis = 0)
#        y_curve.append(avg_pre)
#
#plot_dist_curves_top10ANN(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, title = 'all')
#
#
#y_curve = []
#for i in range(len(ANN_top10)):
#    ANN_list = ANN_top10[i]
#    
#    mag_test = np.arange(2.7,4.5,0.1)
#    dist = np.arange(25.,225.,50.)
#    
#    mag_curve = np.array([mag_test for i in range(len(dist))])
#    dist_curve = np.array([len(mag_test)*[x] for x in dist])
#
#    for i in range(len(dist)):
#        ###################    
#        if site == '5coeff':
#            d = {'mw': mag_curve[i], 'R': dist_curve[i]} 
#        else:
#            d = {'mw': mag_curve[i], 'R': dist_curve[i], 'vs30': vref*np.ones(len(mag_curve[i]))}    
#
#        df = pd.DataFrame(data=d)
#        x_curve_scale = scaler.transform(df)
#        
#        pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(fold)]
#        avg_pre = np.average(pre_list, axis = 0)
#        y_curve.append(avg_pre)
#plot_mag_curves_top10ANN(preln = np.asarray(y_curve), mag= mag_curve, dist = dist_curve, workinghome=workinghome,title = 'all')
#


def plot_dist_curves_top10ANN(preln, mag, dist, workinghome, title):
    #go from log 10 to ln
#    pre = preln*np.log10(np.e)
#    obs = obs*np.log10(np.e)
    #pga vs distance for various magnitudes   
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=2.8, vmax=4)
    colors = [cmap(normalize(value)) for value in mag]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])

    fig = plt.figure(figsize=(12,10))
    plt.title('site = ' + title)
    
    for i in range(len(mag)):
        pre = (preln[i]*np.log10(np.e)).flatten()
        plt.plot(np.log10(dist[i]), pre, color = cmap(normalize(mag[i][0])), label = mag[i][0])

    for k in range(10):
        for i in range(len(mag)):
            pre = (preln[k*7+i]*np.log10(np.e)).flatten()
            plt.plot(np.log10(dist[i]), pre, color = cmap(normalize(mag[i][0])))#, label = mag[i][0])
    plt.legend(loc = 'lower left')
    plt.xlabel('log10 (distance)')
    plt.ylabel('log10 predicted pga')
    plt.ylim([-6, -1.5])
    plt.xlim([1,2.5])
    
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(ur"magnitude")#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    plt.savefig(workinghome + '/' + 'curves_dist_' + title + '.png')



def plot_mag_curves_top10ANN(preln, mag, dist, workinghome,title):
    cmap = mpl.cm.get_cmap('plasma')
    normalize = mpl.colors.Normalize(vmin=10., vmax=250.)
    colors = [cmap(normalize(value)) for value in dist]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])

    fig = plt.figure(figsize=(12,10))
    plt.title('site = ' + title)
    
    for i in range(len(dist)):
        pre = (preln[i]*np.log10(np.e)).flatten()
        plt.plot(mag[i], pre, color = cmap(normalize(dist[i][0])), label = dist[i][0])

    for k in range(10):
        for i in range(len(dist)):
            pre = (preln[k*4+i]*np.log10(np.e)).flatten()
            plt.plot(mag[i], pre, color = cmap(normalize(dist[i][0])))#, label = mag[i][0])
    plt.legend(loc = 'lower left')
    plt.xlabel('magnitude')
    plt.ylabel('log10 predicted pga')
    plt.ylim([-6, -1.5])
    plt.xlim([2.8, 4.5])
    
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(ur"distance (km)")#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()

    plt.savefig(workinghome + '/' + 'curves_mag_' + title + '.png')







###############'#### ME   
    
cmap = plt.set_cmap('cividis')
cNorm  = mpl.colors.Normalize(vmin=-7, vmax=-1)


site = 'vs30'
workinghome = '/Users/aklimase/Documents/GMM_ML/mixed_effects_out'
foldername = 'mixed_effects_'+ site
paramsfile = '/Users/aklimase/Documents/GMM_ML/models/pckl/database_' + site + '/r/results_fixed.csv'
params = np.genfromtxt(paramsfile,delimiter=",", names = True, usecols = [1,2,3], dtype = float)
n = len(params['Estimate'])
if site == '5coeff':
    a1,a2,a3,a4,a5 = params['Estimate']
else:
    a1,a2,a3,a4,a5,a6 = params['Estimate']
    
#read in results site
#use PFO for siteterm bc vs30 = 763
sitefile = '/Users/aklimase/Documents/GMM_ML/models/pckl/database_' + site + '/r/results_site.csv'
siteterms_me = np.genfromtxt(sitefile,delimiter=",", names = True, usecols = [0,1,2], dtype = None)
bias = siteterms_me[7][1]

mlist = np.linspace(2.8,5.0,200)
Rlist = np.linspace(10.,225.,200)
X, Y = np.meshgrid(Rlist, mlist)
if site == '5coeff':
    Z = a1 + a2*Y + a3*(8.5-Y)**2. + a4*np.log(X) + a5*X + bias
else:
    Z = a1 + a2*Y + a3*(8.5-Y)**2. + a4*np.log(X) + a5*X + a6*np.log(vref/vref)  + bias 

#ln to log10
Z= np.log10(np.e)*Z
X = np.log10(X)
fig = plt.figure(figsize=(12,10))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
plt.xlim([1,np.log10(225.)])
plt.ylim([2.8,5.0])
 
cp = plt.contourf(X, Y, Z, cmap = cmap, norm=cNorm, levels = np.arange(-6,0+1,0.5))
cl = plt.contour(X, Y, Z, colors='k',levels = np.arange(-6,0+1,0.5))#, levels = 6)
plt.clabel(cl, colors = 'k', fmt = '%2.1f',fontsize=20, inline=True)

    
cbar = plt.colorbar(cp, norm = cNorm)
cbar.ax.set_ylabel('predicted ln(PGA)')

ax.set_title('Contour Plot')
ax.set_xlabel('R (km)')
ax.set_ylabel('mw')
plt.show()
plt.savefig(workinghome + '/' + foldername + '/' + 'contour.png')
plt.close()

##
#
#

####################################### observation grid
#    
#ln to log10
#z = np.log(db.pga/9.81)*np.log10(np.e)
#cmap = mpl.cm.get_cmap('cividis')
#normalize = mpl.colors.Normalize(vmin=-7, vmax=-1)
#colors = [cmap(normalize(value)) for value in z]
#
#fig = plt.figure(figsize=(12,10))
#left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#ax = fig.add_axes([left, bottom, width, height]) 
#plt.xlim([1,np.log10(225.)])
#plt.ylim([2.8,5.0])
# 
#x = np.log10(db.r)
#y = db.mw
#
#plt.scatter(x,y,s=15,color = colors, alpha = 0.3)
##cl = plt.contour(X, Y, Z, colors='k',levels = np.arange(-13,-2+1,1))#, levels = 6)
##plt.clabel(cl, colors = 'k', fmt = '%2.1f',fontsize=20, inline=True)
#
#cbar = plt.colorbar(cp, norm = cNorm)
#cbar.ax.set_ylabel('predicted ln(PGA)')
#
##ax.set_title('Contour Plot')
#ax.set_xlabel('R (km)')
#ax.set_ylabel('mw')
#plt.show()
#plt.savefig(workinghome + '/observations.png')
#plt.close()
#





#grid obs pga values

#
#x = np.log10(db.r)
#y = db.mw
##ln to log10
#z = np.log(db.pga/9.81)*np.log10(np.e)
#cmap = mpl.cm.get_cmap('cividis')
#
#dR = np.log10(1.1)
#dm = 0.06
#R_edges = np.arange(1,np.log10(225.)+dR, dR)
#m_edges = np.arange(2.8,5.0+dm, dm)
#bindims = [R_edges, m_edges]
#
#
#sample = c_[x,y]
#statistic_s,bin_edges,binnumber = binned_statistic_dd(sample, z, statistic='median', bins=bindims)
#
#xedges, yedges = bin_edges
#X,Y = np.meshgrid(xedges, yedges)
#norm = mpl.colors.Normalize(vmin=-7, vmax=-1)
#
#
#fig = plt.figure(figsize=(12,10))
#
#cp=plt.pcolormesh(X,Y, statistic_s.T, cmap = cmap, norm = norm)
#
##ax.set_title('Contour Plot')
#plt.xlabel('R (km)')
#plt.ylabel('mw')
#plt.xlim([1,np.log10(225.)])
#plt.ylim([2.8,5.0])
#
#cbar = plt.colorbar(cp, norm = norm)
#cbar.ax.set_ylabel('predicted ln(PGA)')
#
#plt.show()
#plt.savefig(workinghome + '/observations_grid2.png')
##plt.close()
#
#














