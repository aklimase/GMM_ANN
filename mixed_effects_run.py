#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:58:21 2019

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
from plot_gmpe import plotting, plot_az,  plot_AIC, plotsiteterms, plot_dist_curves, plot_mag_curves, plot_dist_curves_scatter, plot_mag_curves_scatter, residual_histo

plt.style.use("classic")

sns.set_context("poster")
sns.set_style('whitegrid')
g = Geod(ellps='clrk66') 

mpl.rcParams['font.size'] = 22

seed = 81
np.random.seed(seed)
tf.set_random_seed(81)

t = '/Users/aklimase/Documents/GMM_ML/catalog/tstar_site.txt'
tstarcat = np.genfromtxt(t, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)
                         
v = '/Users/aklimase/Documents/GMM_ML/catalog/vs30_sta.txt'
vs30cat = np.genfromtxt(v, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)

site =  tstarcat['site']
k = tstarcat['tstars']
vs30 = vs30cat['Vs30']

siteparams = [site, k, vs30]

name = 'vs30'
n = 6
home = '/Users/aklimase/software/python/grmpy'
workinghome1 = '/Users/aklimase/Documents/GMM_ML'
workinghome = '/Users/aklimase/Documents/GMM_ML/mixed_effects_out'
codehome = '/Users/aklimase/software/python/grmpy'


if name == 'vs30':
    vref = 760.
elif name == 'kappa':
    vref = 0.06
else:
    vref = 760.

# 5 coeff
if name == 'kappa':
    dbname = 'database_kappa'
else:
    dbname = 'database_vs30'
    
foldername = 'mixed_effects_' + name

if not os.path.exists(workinghome + '/' + foldername):
    os.mkdir(workinghome + '/' + foldername)
    os.mkdir(workinghome + '/' + foldername + '/testing')
    os.mkdir(workinghome + '/' + foldername + '/training')
    os.mkdir(workinghome + '/' + foldername + '/validation')
    os.mkdir(workinghome + '/' + foldername + '/curves')
    
    
db = pickle.load(open(workinghome1  + '/' + dbname + '.pckl', 'r'))

if name == 'kappa':
#    d = {'mw': db.mw,'R': db.r, 'pga': db.pga/9.81,  'vs30': db.vs30.flatten(), 'evnum': db.evnum, 'sta':db.sta, 'elat': db.elat, 'elon': db.elon,'stlat': db.stlat,'stlon': db.stlon}
    d = {'mw': db.mw,'R': db.r, 'pga': db.pga/9.81,  'vs30': db.vs30.flatten(), 'evnum': db.evnum, 'sta':db.sta, 'elat': db.elat, 'elon': db.elon,'stlat': db.stlat,'stlon': db.stlon}
else: 
    d = {'mw': db.mw,'R': db.r, 'pga': db.pga/9.81,  'vs30': db.vs30.flatten(), 'evnum': db.evnum, 'sta':db.sta, 'elat': db.elat, 'elon': db.elon,'stlat': db.stlat,'stlon': db.stlon}

if name == '5coeff':
    dbname = 'database_5coeff'
    
df = pd.DataFrame(data=d)
train, test_valid = train_test_split(df, test_size=0.4, random_state = seed)
valid, test = train_test_split(test_valid, test_size=1/2., random_state = seed)

y_train = train['pga']
x_train_coor = train[['elat', 'elon', 'stlat', 'stlon']]
x_train =  train.drop(['pga','elat', 'elon', 'stlat', 'stlon'], axis = 1)  
y_test = test['pga']
x_test_coor = test[['elat', 'elon', 'stlat', 'stlon']]
x_test =  test.drop(['pga','elat', 'elon', 'stlat', 'stlon'], axis = 1)  
y_valid = valid['pga']
x_valid_coor = valid[['elat', 'elon', 'stlat', 'stlon']]
x_valid =  valid.drop(['pga','elat', 'elon', 'stlat', 'stlon'], axis = 1)

if name == 'kappa': #exp of kappa so that it scales linearly
    me_log, fixed_data, event_terms, site_terms, d_predicted, d_observed, event_mean, event_std, site_mean, site_std= inv.mixed_effects(codehome, workinghome1, dbname, np.asarray(y_train), np.asarray(x_train['mw']), np.asarray(x_train['R']), np.asarray(x_train['vs30']),np.asarray(x_train['evnum']),np.asarray(x_train['sta']), vref = vref, c = 4.5, Mc = 8.5, predictive_parameter = 'pga',  ncoeff =n,data_correct = 0)#, a4 = -1.2)
else:
    me_log, fixed_data, event_terms, site_terms, d_predicted, d_observed, event_mean, event_std, site_mean, site_std= inv.mixed_effects(codehome, workinghome1, dbname, np.asarray(y_train), np.asarray(x_train['mw']), np.asarray(x_train['R']), np.asarray(x_train['vs30']),np.asarray(x_train['evnum']),np.asarray(x_train['sta']), vref = vref, c = 4.5, Mc = 8.5, predictive_parameter = 'pga',  ncoeff =n,data_correct = 0)#, a4 = -1.2)

#read in parameters and calculate on test data
paramsfile = workinghome1 + '/models/pckl/' + dbname + '/r/results_fixed.csv'
params = np.genfromtxt(paramsfile,delimiter=",", names = True, usecols = [1,2,3], dtype = float)
if n == 5:
    a1,a2,a3,a4,a5 = params['Estimate']
#    a1,a2,a3,a5 = params['Estimate']

else:
    a1,a2,a3,a4,a5,a6 = params['Estimate']
    
print foldername

##testing data
#loop through each record and add in site term
siteterm = []
#read in results site
sitefile = workinghome1 + '/models/pckl/' + dbname + '/r/results_site.csv'
siteterms = np.genfromtxt(sitefile,delimiter=",", names = True, usecols = [0,1,2], dtype = None)
for i in range(len(x_test)):
    sta = x_test.iloc[i]['sta']
    for j in range(len(siteterms)):
        if siteterms[j][0] == sta:
            siteterm.append(siteterms[j][1])

#make predictions for test
if n == 6 and name == 'vs30':
    ME_test_predicted = a1 + a2*x_test['mw'] + a3*(8.5-x_test['mw'])**2. + a4*np.log(x_test['R']) + a5*x_test['R'] + a6*np.log(x_test['vs30']/vref)
elif n == 6 and name == 'kappa':
    ME_test_predicted = a1 + a2*x_test['mw'] + a3*(8.5-x_test['mw'])**2. + a4*np.log(x_test['R']) + a5*x_test['R'] + a6*np.log(x_test['vs30']/vref)
else:
    ME_test_predicted = a1 + a2*x_test['mw'] + a3*(8.5-x_test['mw'])**2. + a4*np.log(x_test['R']) + a5*x_test['R']

ME_test_predicted_noresid = np.asarray(ME_test_predicted)
ME_test_predicted = np.asarray(ME_test_predicted) + siteterm# in ln
print np.std(ME_test_predicted -np.log(y_test))

plotting(pre = ME_test_predicted, obs = np.asarray(np.log(y_test)), mag = list(x_test['mw']), dist = list(x_test['R']), sta = x_test['sta'], foldername = 'mixed_effects_' + name , outname = 'testing', workinghome = workinghome)
#plot_az(pre = ME_test_predicted, obs = np.asarray(np.log(y_test)), mag = list(x_test['mw']), dist = list(x_test['R']), sta = x_test['sta'], coor = x_test_coor, foldername = 'mixed_effects_' + name , outname = 'testing', workinghome = workinghome)

##testing data
#loop through each record and add in site term
siteterm = []
#read in results site
sitefile = workinghome1 + '/models/pckl/' + dbname + '/r/results_site.csv'
siteterms = np.genfromtxt(sitefile,delimiter=",", names = True, usecols = [0,1,2], dtype = None)
for i in range(len(x_train)):
    sta = x_train.iloc[i]['sta']
    for j in range(len(siteterms)):
        if siteterms[j][0] == sta:
            siteterm.append(siteterms[j][1])
            
#eventterm = []
##read in results site
#eventfile = workinghome1 + '/models/pckl/' + foldername + '/r/results_event.csv'
#eventterms = np.genfromtxt(eventfile,delimiter=",", names = True, usecols = [0,1,2], dtype = None)
#for i in range(len(x_train)):
#    sta = x_train.iloc[i]['sta']
#    for j in range(len(eventterms)):
#        if eventterms[j][0] == sta:
#            eventterm.append(eventterms[j][1])
  
#training data            
if n == 6 and name == 'vs30':
    ME_train_predicted = a1 + a2*x_train['mw'] + a3*(8.5-x_train['mw'])**2. + a4*np.log(x_train['R']) + a5*x_train['R'] + a6*np.log(x_train['vs30']/vref)
elif n == 6 and name == 'kappa':
    ME_train_predicted = a1 + a2*x_train['mw'] + a3*(8.5-x_train['mw'])**2. + a4*np.log(x_train['R']) + a5*x_train['R'] + a6*np.log(x_train['vs30']/vref)
else:
    ME_train_predicted = a1 + a2*x_train['mw'] + a3*(8.5-x_train['mw'])**2. + a4*np.log(x_train['R']) + a5*x_train['R']
ME_train_predicted_noresid = np.asarray(ME_train_predicted)
ME_train_predicted = np.asarray(ME_train_predicted) + event_terms[:,0] + siteterm


#plotting(pre = ME_train_predicted, obs = np.asarray(np.log(y_train)), mag = list(x_train['mw']), dist = list(x_train['R']), sta = x_train['sta'], foldername = 'mixed_effects_'+ name, outname = 'training', workinghome = workinghome)
#plot_az(pre = ME_train_predicted, obs = np.asarray(np.log(y_train)), mag = list(x_train['mw']), dist = list(x_train['R']), sta = x_train['sta'], coor = x_train_coor, foldername = 'mixed_effects_' + name , outname = 'training', workinghome = workinghome)


#validation data
siteterm = []
#read in results site
for i in range(len(x_valid)):
    sta = x_valid.iloc[i]['sta']
    for j in range(len(siteterms)):
        if siteterms[j][0] == sta:
            siteterm.append(siteterms[j][1])
            
#make predictions for validation
if n == 6 and name == 'vs30':
    ME_valid_predicted = a1 + a2*x_valid['mw'] + a3*(8.5-x_valid['mw'])**2. + a4*np.log(x_valid['R']) + a5*x_valid['R'] + a6*np.log(x_valid['vs30']/vref)
elif n == 6 and name == 'kappa':
    ME_valid_predicted = a1 + a2*x_valid['mw'] + a3*(8.5-x_valid['mw'])**2. + a4*np.log(x_valid['R']) + a5*x_valid['R'] + a6*np.log(x_valid['vs30']/vref)
else:
    ME_valid_predicted = a1 + a2*x_valid['mw'] + a3*(8.5-x_valid['mw'])**2. + a4*np.log(x_valid['R']) + a5*x_valid['R']
ME_valid_predicted_noresid = np.asarray(ME_valid_predicted)
ME_valid_predicted = np.asarray(ME_valid_predicted) + siteterm# in ln
print np.std(ME_valid_predicted -np.log(y_valid))
#plotting(pre = ME_valid_predicted, obs = np.asarray(np.log(y_valid)), mag = list(x_valid['mw']), dist = list(x_valid['R']), sta = x_valid['sta'], foldername = 'mixed_effects_' + name , outname = 'validation', workinghome = workinghome)
#plot_az(pre = ME_valid_predicted, obs = np.asarray(np.log(y_valid)), mag = list(x_valid['mw']), dist = list(x_valid['R']), sta = x_valid['sta'], coor = x_valid_coor, foldername = 'mixed_effects_' + name , outname = 'validation', workinghome = workinghome)


#test,training, validation data
resid_train = np.asarray(np.log(y_train)) - ME_train_predicted
std_train = round(np.std(resid_train),6)
mean_train = round(np.mean(resid_train),6)

resid_train_noresidterm = np.asarray(np.log(y_train)) - ME_train_predicted_noresid
std_train_noresidterm = round(np.std(resid_train_noresidterm),6)

resid_valid = np.asarray(np.log(y_valid)) - ME_valid_predicted
std_valid = round(np.std(resid_valid),6)
mean_valid = round(np.mean(resid_valid),6)

resid_valid_noresidterm = np.asarray(np.log(y_valid)) - ME_valid_predicted_noresid
std_valid_noresidterm = round(np.std(resid_valid_noresidterm),6)

resid_test = np.asarray(np.log(y_test)) - ME_test_predicted
std_test = round(np.std(resid_test),6)
mean_test = round(np.mean(resid_test),6)

resid_test_noresidterm = np.asarray(np.log(y_test)) - ME_test_predicted_noresid
std_test_noresidterm = round(np.std(resid_test_noresidterm),6)

f = open(workinghome  + '/' +  foldername + '/kfoldperformance.txt','w') 
f.write('std of training data + event + site: ' + str(std_train) + '\n')
f.write('std of training data no terms: ' + str(std_train_noresidterm) + '\n')
f.write('std of validation data + site: ' + str(std_valid) + '\n')
f.write('std of validation data no terms: ' + str(std_valid_noresidterm) + '\n')
f.write('std of testing data + site: ' + str(std_test) + '\n')
f.write('std of testing data no terms: ' + str(std_test_noresidterm) + '\n')
f.close()
    
#
#histo of residuals
#fig = plt.figure(figsize=(8,8))#tight_layout=True)
#ax = fig.add_subplot(111)
## N is the count in each bin, bins is the lower-limit of the bin
#labeltrain = r'$\mu_{train}:$ ' + str(mean_train) + '\n' + r'$\sigma_{train}:$ ' + str(std_train)
#labelvalid = r'$\mu_{valid}:$ ' + str(mean_valid) + '\n' + r'$\sigma_{valid}:$ ' + str(std_valid)
#labeltest = r'$\mu_{test}:$ ' + str(mean_test) + '\n' + r'$\sigma_{test}:$ ' + str(std_test)
#
#label=[labeltrain, labelvalid, labeltest]
#N, bins, patches = ax.hist([resid_train, resid_valid, resid_test], color=['blue', 'green', 'red'], label = label, bins=60)
#ax.set_xlim(-4,4)
#ax.set_xlabel('residuals')
#ax.set_ylabel('counts')
#plt.legend(fontsize = 16)
#plt.title('Residuals')
#plt.tight_layout()
##plt.savefig(workinghome + '/mixed_effects_' + name + '_siteterm/histoln.png')
#plt.savefig(workinghome + '/mixed_effects_' + name + '/histoln.png')

residual_histo(np.log(y_train), np.log(y_valid), np.log(y_test), ME_train_predicted, ME_valid_predicted, ME_test_predicted, workinghome, foldername, n='')


sitefile = workinghome1 + '/models/pckl/' + dbname + '/r/results_site.csv'
siteterms = np.genfromtxt(sitefile,delimiter=",", names = True, usecols = [0,1,2], dtype = None)
            
#setup_test_curves_ME(name, workinghome, foldername, siteparams, x_test, coefflist = params['Estimate'], siteterms = siteterms)
#setup_test_curves_scatter_ME(name, workinghome, foldername, siteparams, x_test, coefflist = params['Estimate'], siteterm = siteterms, pre_scatter = ME_test_predicted, obs_scatter = np.asarray(y_test), dist_scatter = np.asarray(x_test['R']), mag_scatter = np.asarray(x_test['mw']), sta = x_test['sta'])

#setup_test_curves_ME(name, workinghome, foldername, siteparams, x_test, coefflist = [a1,a2,a3,a4,a5,a6], siteterms = siteterms)
#setup_test_curves_scatter_ME(name, workinghome, foldername, siteparams, x_test, coefflist = [a1,a2,a3,a4,a5,a6], siteterm = siteterms, pre_scatter = ME_test_predicted, obs_scatter = np.asarray(y_test), dist_scatter = np.asarray(x_test['R']), mag_scatter = np.asarray(x_test['mw']), sta = x_test['sta'])

#############################################################################



def setup_test_curves_ME(site, workinghome, foldername, siteparams, x_test, coefflist, siteterms):
    if site == '5coeff':
        a1,a2,a3,a4,a5 = coefflist
    else:
        a1,a2,a3,a4,a5,a6 = coefflist
        
    mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
    dist = np.arange(1.,200.,2)
        
    mag_curve = np.array([100*[x] for x in mag_test])
    dist_curve = np.array([dist for i in range(len(mag_test))])

    n = len(mag_test)
     
    y_curve = []
    for k in range(len(siteterms['ID'])):
        sitename = siteterms['ID'][k]
        #make a prediction per site
        if site == '5coeff':
            bias = siteterms['Bias'][k]
        elif site == 'kappa': 
            bias = siteterms['Bias'][k]
            a6term = siteparams[1][k]
        else: 
            bias = siteterms['Bias'][k]
            a6term = siteparams[2][k]
            
        
        for i in range(n):
            if site == '5coeff':
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias
            else:
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias + a6*np.log(a6term/vref)
            y_curve.append(ME_test_predicted)

    
        plot_dist_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = sitename, n = '')


    mag_test = np.arange(2.7,4,0.1)
    dist = np.arange(25.,200.,25)

    n = len(dist)
    p = len(mag_test)
        
    mag_curve = np.array([mag_test for i in range(n)])
    dist_curve = np.array([p*[x] for x in dist])
    y_curve = []
    for k in range(len(siteterms['ID'])):
        sitename = siteterms['ID'][k]
        #make a prediction per site
        if site == '5coeff':
            bias = siteterms['Bias'][k]
        elif site == 'kappa': 
            bias = siteterms['Bias'][k]
            a6term = siteparams[1][k]
        else: 
            bias = siteterms['Bias'][k]
            a6term = siteparams[2][k]
            
#        y_curve = []
        for i in range(n):
            if site == '5coeff':
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias
            else:
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias + a6*np.log(a6term/vref)
            y_curve.append(ME_test_predicted)

        plot_mag_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = sitename, n = '')
    plt.close('all')

        
        
def setup_test_curves_scatter_ME(site, workinghome, foldername, siteparams, x_test, coefflist, siteterm, pre_scatter, obs_scatter, dist_scatter, mag_scatter, sta):
    if site == '5coeff':
        a1,a2,a3,a4,a5 = coefflist
    else:
        a1,a2,a3,a4,a5,a6 = coefflist
        
    mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
    dist = np.arange(1.,200.,2)
        
    mag_curve = np.array([100*[x] for x in mag_test])
    dist_curve = np.array([dist for i in range(len(mag_test))])

    n = len(mag_test)
        
    y_curve = []

    for k in range(len(siteterms['ID'])):
        sitename = siteterms['ID'][k]
        #make a prediction per site
        if site == '5coeff':
            bias = siteterms['Bias'][k]
        elif site == 'kappa': 
            bias = siteterms['Bias'][k]
            a6term = siteparams[1][k]
        else: 
            bias = siteterms['Bias'][k]
            a6term = siteparams[2][k]
            
        uniq = list(set(sta))
#        for i in range(len(uniq)):
        indx = sitename
        indexlist =np.where(sta==indx)[0]
        pre_scatter_sta = np.asarray([pre_scatter[j] for j in indexlist])
        obs_scatter_sta = np.asarray([obs_scatter[j] for j in indexlist])
        dist_scatter_sta = np.asarray([dist_scatter[j] for j in indexlist])
        mag_scatter_sta = np.asarray([mag_scatter[j] for j in indexlist])

            
#        y_curve = []
        for i in range(n):
            if site == '5coeff':
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias
            else:
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias + a6*np.log(a6term/vref)
            y_curve.append(ME_test_predicted)

#        plot_dist_curves_scatter(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = sitename, n = '')
        plot_dist_curves_scatter(preln_curve = np.asarray(y_curve), mag_curve = mag_curve, dist_curve = dist_curve, pre_scatter = pre_scatter_sta, obs_scatter = np.asarray(obs_scatter_sta), dist_scatter = np.asarray(dist_scatter_sta), mag_scatter = np.asarray(mag_scatter_sta), workinghome=workinghome, foldername = foldername, title = sitename)


    mag_test = np.arange(2.7,4,0.1)
    dist = np.arange(25.,200.,25)

    n = len(dist)
    p = len(mag_test)
        
    mag_curve = np.array([mag_test for i in range(n)])
    dist_curve = np.array([p*[x] for x in dist])
    y_curve = []

    for k in range(len(siteterms['ID'])):
        sitename = siteterms['ID'][k]
        #make a prediction per site
        if site == '5coeff':
            bias = siteterms['Bias'][k]
        elif site == 'kappa': 
            bias = siteterms['Bias'][k]
            a6term = siteparams[1][k]
        else: 
            bias = siteterms['Bias'][k]
            a6term = siteparams[2][k]
            
#        uniq = list(set(sta))
#        for i in range(len(uniq)):
#            indx = uniq[i]
#            indexlist =np.where(sta==indx)[0]
#            pre_scatter_sta = np.asarray([pre_scatter[j] for j in indexlist])
#            obs_scatter_sta = np.asarray([obs_scatter[j] for j in indexlist])
#            dist_scatter_sta = np.asarray([dist_scatter[j] for j in indexlist])
#            mag_scatter_sta = np.asarray([mag_scatter[j] for j in indexlist])
            
        uniq = list(set(sta))
#        for i in range(len(uniq)):
        indx = sitename
        indexlist =np.where(sta==indx)[0]
        pre_scatter_sta = np.asarray([pre_scatter[j] for j in indexlist])
        obs_scatter_sta = np.asarray([obs_scatter[j] for j in indexlist])
        dist_scatter_sta = np.asarray([dist_scatter[j] for j in indexlist])
        mag_scatter_sta = np.asarray([mag_scatter[j] for j in indexlist])

#        y_curve = []
        for i in range(n):
            if site == '5coeff':
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias
            else:
                ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias + a6*np.log(a6term/vref)
            y_curve.append(ME_test_predicted)

        plot_mag_curves_scatter(preln_curve = np.asarray(y_curve), mag_curve = mag_curve, dist_curve = dist_curve,  pre_scatter = pre_scatter_sta, obs_scatter = np.asarray(obs_scatter_sta), dist_scatter = np.asarray(dist_scatter_sta), mag_scatter = np.asarray(mag_scatter_sta), workinghome=workinghome, foldername = foldername, title = sitename)
    plt.close('all')

#foldername = 'mixed_effects_' + name        
#setup_test_curves_ME(name, workinghome, foldername, siteparams, x_test, coefflist = params['Estimate'], siteterms = siteterms)
#setup_test_curves_scatter_ME(name, workinghome, foldername, siteparams, x_test, coefflist = params['Estimate'], siteterm = siteterms, pre_scatter = ME_test_predicted, obs_scatter = np.asarray(y_test), dist_scatter = np.asarray(x_test['R']), mag_scatter = np.asarray(x_test['mw']), sta = x_test['sta'])
