#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:36:25 2019

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
import glob
from scipy.stats import pearsonr
import statsmodels
import statsmodels.stats.power
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)


plt.style.use("classic")

sns.set_context("poster")
sns.set_style('whitegrid')
g = Geod(ellps='clrk66') 

mpl.rcParams['font.size'] = 30
sns.set(font_scale=5)
#seed = 18
#np.random.seed(seed)

#workinghome = '/Users/aklimase/Documents/GMM_ML/model_AICtestvs30_1layer'
#workinghome = '/Users/aklimase/Documents/GMM_ML/mixed_effects_out'


def plotting(pre, obs, mag, dist, sta, foldername, outname, workinghome):
    sns.set_style('whitegrid')

    db = pickle.load(open('/Users/aklimase/Documents/GMM_ML/database_kappa.pckl', 'r'))

    preln = pre
    obsln = obs
    #go from ln to log10
    pre = pre*np.log10(np.e)
    obs = obs*np.log10(np.e)
    
    #histogram of dataset
    norm=plt.Normalize(np.min(obs),np.max(obs))
    colors = plt.cm.plasma(norm(obs))
    g=sns.jointplot(x = dist,y = mag, stat_func=None, height=10, color = 'darkgray', marginal_kws=dict(bins=15))
    g.ax_joint.cla()
    g.ax_joint.tick_params(axis='both', which='major', labelsize=30)
    g.ax_joint.tick_params(axis='both', which='minor', labelsize=30)
    g.ax_joint.scatter(dist,mag,s = 2, color = 'black')
    g.set_axis_labels(r'Record $R_{rup}$ (km)', 'Magnitude', fontsize=20)
    g.ax_joint.set_xlim(0,250)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.98, bottom=0.36)  # shrink fig so cbar is visible
    axHist = g.fig.add_axes([.15, .085, 0.67, 0.18])
    N, bins, patches = axHist.hist(obs, bins=np.arange(-7,0,0.25)) #, orientation=u'horizontal')
    plt.xlabel(r'$\log_{10} PGA$', fontsize = 20)
    plt.ylabel(r'counts', fontsize = 20)
    m = (int(max(N))/1000)*1000 + 1000
    plt.yticks(np.arange(0,m, 500))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='y', which='major', labelsize=20, rotation = 45)
    plt.tick_params(axis='both', which='minor', labelsize=20)  
    for bin, patch in zip(bins, patches):
        color = plt.cm.plasma(norm(bin))
        patch.set_facecolor(color)
    plt.savefig(workinghome + '/' + foldername + '/' + outname+ '/' + 'histo.png')

    #obs vs pre colored by distance
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.r), vmax=max(db.r))
    colors = [cmap(normalize(value)) for value in db.r]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,8))
    plt.scatter(obs, pre, s = 4, color = colors)
    plt.xlabel('observed (log pga)')
    plt.ylabel('predicted (log pga)')
    plt.axis([-7, 0, -7, 0])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(r'$\log_{10} R_{rup}$')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    plt.savefig(workinghome + '/' + foldername + '/' + outname+ '/' + 'obs_vs_pre_distance.png')    
    
    #obs vs pre colored by mag
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.mw), vmax=max(db.mw))
    colors = [cmap(normalize(value)) for value in db.mw]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,8))
    plt.scatter(obs, pre, s = 4, color = colors)
    plt.xlabel('observed (log pga)')
    plt.ylabel('predicted (log pga)')
    plt.axis([-7, 0, -7, 0])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label('M', fontweight = 'bold')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/'+ 'obs_vs_pre_mag.png')
    
    #obs vs pre colored by station
    fig = plt.figure(figsize=(10,10))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [obs[j] for j in indexlist]
        y = [pre[j] for j in indexlist]
        plt.scatter(x, y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]))
    plt.xlabel('observed (log pga)')
    plt.ylabel('predicted (log pga)')
    plt.axis([-7, 0, -7, 0])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 14)
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/'+ 'obs_vs_pre_sta.png')
    
    #pre vs mag (colored by distance)
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.r), vmax=max(db.r))
    colors = [cmap(normalize(value)) for value in db.r]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,8))
    plt.scatter(mag, pre, s = 4, color = colors)
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel(r'$\log_{10} PGA$')    
    plt.ylim([-7, 0])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(r'$R_{rup}$')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
#    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/' + 'pre_vs_mag.png')
    
    #pre vs. log dist colored by station
    fig = plt.figure(figsize=(10,10))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [mag[j] for j in indexlist]
        y = [pre[j] for j in indexlist]
        plt.scatter(x, y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]))
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 14)
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/pre_vs_mag_sta.png')
    
    #pre vs mag (colored by binned distance)
    uniq = list([0,25,50,100,125,150,175,200,250])
    # Set the color map to match the number of species
    hsv = plt.get_cmap('tab10')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    fig = plt.figure(figsize=(10,9))
    for i in range(len(uniq)-1):
        indx = uniq[i]
        indexlist =np.where((uniq[i] <= np.asarray(dist)) & (np.asarray(dist) < uniq[i+1]))[0]
        x = [mag[j] for j in indexlist]
        y = [pre[j] for j in indexlist]
        plt.scatter(x,y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]) + '-' + str(uniq[i+1]) )
    plt.legend(loc='lower right',prop={'size': 18})
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/' + 'bins_pre_vs_mag_dist.png')
    
    #obs vs mag (colored by distance)
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.r), vmax=max(db.r))
    colors = [cmap(normalize(value)) for value in db.r]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,8))
    plt.scatter(mag, obs, s = 4, color = colors)
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(r'$R_{rup}$')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/' + 'obs_vs_mag.png')
    
    #obs vs mag (colored by binned distance)
    uniq = list([0,25,50,100,125,150,175,200,250])
    # Set the color map to match the number of species
    hsv = plt.get_cmap('tab10')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    fig = plt.figure(figsize=(10,9))
    for i in range(len(uniq)-1):
        indx = uniq[i]
        indexlist =np.where((uniq[i] <= np.asarray(dist)) & (np.asarray(dist) < uniq[i+1]))[0]
        x = [mag[j] for j in indexlist]
        y = [obs[j] for j in indexlist]
        plt.scatter(x,y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]) + '-' + str(uniq[i+1]) )
    plt.legend(loc='lower right',prop={'size': 18})
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/' + 'bins_obs_vs_mag_dist.png')
    
    #pre vs log distance (colored by magnitude)
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.mw), vmax=max(db.mw))
    colors = [cmap(normalize(value)) for value in db.mw]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.log10(dist), pre, s = 4, color = colors)
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label('M', fontweight = 'bold')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    plt.savefig(workinghome + '/' + foldername + '/' + outname +  '/' + 'pre_vs_dist.png')
    
    #pre vs. log dist colored by station
    fig = plt.figure(figsize=(10,10))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [pre[j] for j in indexlist]
        plt.scatter(x, y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]))
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 14)
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/pre_vs_dist_sta.png')
    
    #pre vs log dist 
    uniq = list([2.7,2.8,2.9,3,3.25,3.5,4,4.5,5])
    # Set the color map to match the number of species
    hsv = plt.get_cmap('viridis')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(uniq)-1):
        indx = uniq[i]
        indexlist =np.where((uniq[i] <= np.asarray(mag)) & (np.asarray(mag) < uniq[i+1]))[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [pre[j] for j in indexlist]
        plt.scatter(x,y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]) + '-' + str(uniq[i+1]) )
    plt.legend(loc='lower left',prop={'size': 18})
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/' + 'bins_pre_vs_dist_mag.png')
    
    # obs vs log dist colored by binned mag
    uniq = list([2.7,2.8,2.9,3,3.25,3.5,4,4.5,5])
    # Set the color map to match the number of species
    hsv = plt.get_cmap('viridis')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(uniq)-1):
        indx = uniq[i]
        indexlist =np.where((uniq[i] <= np.asarray(mag)) & (np.asarray(mag) < uniq[i+1]))[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [obs[j] for j in indexlist]
        plt.scatter(x,y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]) + '-' + str(uniq[i+1]) )
    plt.legend(loc='lower left',prop={'size': 18})
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/' + 'bins_obs_vs_dist_mag.png')
  
    # obs vs log dist colored by mag
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.mw), vmax=max(db.mw))
    colors = [cmap(normalize(value)) for value in db.mw]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.log10(dist), obs, s = 4, color = colors)
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label('M', fontweight = 'bold')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/'+ 'obs_vs_dist.png')
    
    #obs vs log distance colored by station
    fig = plt.figure(figsize=(10,10))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [obs[j] for j in indexlist]
        plt.scatter(x, y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]))
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-7, 0])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 14)
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/obs_vs_dist_sta.png')
    
    #residuals vs dist
    resid = obsln-preln
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.mw), vmax=max(db.mw))
    colors = [cmap(normalize(value)) for value in mag]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,10))
    plt.scatter(np.log10(dist), resid, s = 4, color = colors)
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel('residual (ln)')
    plt.ylim([-7,7])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label('M', fontweight = 'bold')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    plt.savefig(workinghome + '/' + foldername + '/' + outname +  '/' + 'resid_vs_dist.png')
    
    #residuals vs dist colored by station
    fig = plt.figure(figsize=(10,10))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [resid[j] for j in indexlist]
        plt.scatter(x, y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]))
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel('residual (ln pga)')
    plt.ylim([-7,7])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 20)
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/resid_vs_dist_sta.png')


    #residuals vs dist binned
    fig = plt.figure(figsize=(10,10))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [resid[j] for j in indexlist]
        plt.scatter(x, y, s=10, facecolors='none',edgecolors=scalarMap.to_rgba(i), label=str(uniq[i]), alpha = 0.5)
    nbins = 12
    n, binedges = np.histogram(np.log10(dist), bins=nbins)
    sy, binedges = np.histogram(np.log10(dist), bins=nbins, weights=resid)
    sy2, binedges = np.histogram(np.log10(dist), bins=nbins, weights=resid*resid)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    
    plt.errorbar((binedges[1:] + binedges[:-1])/2, mean, yerr=std, fmt='k-',marker = '.', markersize = 10, lw = 3)
    
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel('residual (ln pga)')
    plt.ylim([-7,7])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 20, scatterpoints=1)
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/resid_vs_dist_binned.png')
    
    
    #residuals vs dist binned
    # definitions for the axes
    #[left, bottom, width, height]
    
    rect_scatter = [0.15, 0.15, 0.7, 0.73]
    rect_histy = [0.86, 0.15, 0.13, 0.73]

    fig = plt.figure(figsize=(10,8))
    ax_scatter = plt.axes(rect_scatter)
    ax_histy = plt.axes(rect_histy)

    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [resid[j] for j in indexlist]
        ax_scatter.scatter(x, y, s=10, facecolors='none',edgecolors=scalarMap.to_rgba(i), label=str(uniq[i]), alpha = 0.8)
    nbins = 12
    n, binedges = np.histogram(np.log10(dist), bins=nbins)
    sy, binedges = np.histogram(np.log10(dist), bins=nbins, weights=resid)
    sy2, binedges = np.histogram(np.log10(dist), bins=nbins, weights=resid*resid)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    ax_scatter.errorbar((binedges[1:] + binedges[:-1])/2, mean, yerr=std, fmt='k-',marker = '.', markersize = 15, lw = 3, capsize = 4, capthick = 2)
    ax_scatter.set_xlabel(r'$\log_{10} R_{rup}$')
    ax_scatter.set_ylabel('residual (ln pga)')
    ax_scatter.set_ylim([-7,7])
    ax_scatter.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=3, ncol=8, mode="expand", borderaxespad=0.,  scatterpoints=1,fontsize = 20,handletextpad=-0.8)
#    plt.tight_layout()
    #make new axes
#    ax_histy = plt.axes(rect_histy)
    ax_histy.hist(resid, bins=50, orientation = 'horizontal', facecolor='gray')
    ax_histy.set_ylim([-7,7])
    ax_histy.set_xticks([1000, 2000])    
    ax_histy.set_yticklabels([])

#    ax_histy.set_xticks(rotation=45)
    ax_histy.set_xticklabels([1000, 2000])
    ax_histy.set_xticklabels(ax_histy.get_xticklabels(), rotation=310)
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/resid_vs_dist_binned_histo.png')

    #residuals vs magnitude colored by station
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(db.r), vmax=max(db.r))
    colors = [cmap(normalize(value)) for value in dist]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    fig = plt.figure(figsize=(10,8))
    plt.scatter(mag, resid, s = 4, color = colors)
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel('residual (ln pga)')
    plt.ylim([-7,7])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(r'$R_{rup}$')#"$azimuth$ (\u00b0)"
    plt.savefig(workinghome + '/' + foldername + '/' + outname +  '/' + 'resid_vs_mag.png')
    
    #residuals vs dist colored by station
    fig = plt.figure(figsize=(10,11))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [mag[j] for j in indexlist]
        y = [resid[j] for j in indexlist]
        plt.scatter(x, y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]))
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel('residual (ln pga)')
    plt.ylim([-7,7])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 16)
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/resid_vs_mag_sta.png')
    
    #residuals vs mag binned in mag space
    fig = plt.figure(figsize=(10,11))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [mag[j] for j in indexlist]
        y = [resid[j] for j in indexlist]
        plt.scatter(x, y, s=4, color=scalarMap.to_rgba(i), label=str(uniq[i]))
    nbins = 12
    n, binedges = np.histogram(mag, bins=nbins)
    sy, binedges = np.histogram(mag, bins=nbins, weights=resid)
    sy2, binedges = np.histogram(mag, bins=nbins, weights=resid*resid)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    
    plt.errorbar((binedges[1:] + binedges[:-1])/2, mean, yerr=std, fmt='k-', marker = '.', markersize = 15, lw=3)
        
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel('residual (ln pga)')
    plt.ylim([-7,7])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 16)
    plt.tight_layout()
    
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/resid_vs_mag_binned.png')
    
    #residuals vs mag binned in mag space
    rect_scatter = [0.1, 0.12, 0.7, 0.75]
    rect_histy = [0.82, 0.12, 0.15, 0.75]
    
    fig = plt.figure(figsize=(10,8))
    ax_scatter = plt.axes(rect_scatter)
    ax_histy = plt.axes(rect_histy)
    
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [mag[j] for j in indexlist]
        y = [resid[j] for j in indexlist]
        ax_scatter.scatter(x, y, s=10, facecolors='none',edgecolors=scalarMap.to_rgba(i), label=str(uniq[i]),  alpha = 0.8)
    nbins = 12
    n, binedges = np.histogram(mag, bins=nbins)
    sy, binedges = np.histogram(mag, bins=nbins, weights=resid)
    sy2, binedges = np.histogram(mag, bins=nbins, weights=resid*resid)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    
    ax_scatter.errorbar((binedges[1:] + binedges[:-1])/2, mean, yerr=std, fmt='k-',marker = '.', markersize = 15, lw = 3, capsize = 4, capthick = 2)
    ax_scatter.set_xlabel('M', fontweight = 'bold')
    ax_scatter.set_ylabel('residual (ln pga)')
    ax_scatter.set_ylim([-7,7])

    ax_scatter.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=3, ncol=8, mode="expand", borderaxespad=0.,  scatterpoints=1,fontsize = 20,handletextpad=-0.8)#    plt.tight_layout()
    
    ax_histy.hist(resid, bins=50, orientation = 'horizontal', facecolor='gray')
    ax_histy.set_xticklabels(ax_histy.get_xticklabels(), rotation=45)
    ax_histy.set_ylim([-7,7])
    ax_histy.set_xticks([1000, 2000])    
    ax_histy.set_yticklabels([])

#    ax_histy.set_xticks(rotation=45)
    ax_histy.set_xticklabels([1000, 2000])
    ax_histy.set_xticklabels(ax_histy.get_xticklabels(), rotation=310)
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/resid_vs_mag_binned_histo.png')
    
    
    
    #violin plot for residuals at each station
    fig = plt.figure(figsize=(12,8))
    uniq = list(set(sta))
    hsv = plt.get_cmap('tab20')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=hsv)
    x = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x.append([resid[j] for j in indexlist])
    g = sns.violinplot(data = x)
    g.set_xticklabels(uniq, fontsize = 16)
    plt.xlabel('station')
    plt.ylabel('residual (ln)')
    plt.ylim([-7,7])
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/resid_station.png')

    std = round(np.std(obsln-preln),4)
    mean = round(np.mean(obsln-preln),4)
    #histo of residuals
    fig = plt.figure(figsize=(10,8))#tight_layout=True)
    ax = fig.add_subplot(111)
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax.hist(resid, bins=50)
    ax.set_xlim(-4,4)
    ax.text(0.7, 0.9,'std: ' + str(std) + '\n' + 'mean: ' + str(mean),  transform=ax.transAxes)
    ax.set_xlabel('residuals')
    ax.set_ylabel('counts')
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/' + 'histoln.png')
    plt.close('all')


def plot_az(pre, obs, mag, dist, sta, coor, foldername, outname, workinghome):
    pre = pre*np.log10(np.e)
    obs = obs*np.log10(np.e)
    
    fontsize = 28
    
    if not os.path.exists(workinghome + '/' + foldername + '/' + outname + '/stationplots'):
        os.mkdir(workinghome + '/' + foldername + '/' + outname + '/stationplots')
    
    stlon = list(coor['stlon'])
    stlat = list(coor['stlat'])
    elon = list(coor['elon'])
    elat = list(coor['elat'])
    azlist = []
    for i in range(len(stlon)):
        az12,az21,d = g.inv(stlon[i],stlat[i],elon[i],elat[i])
        azlist.append(az12)
    
#    cmap = mpl.cm.get_cmap('plasma')
    cmap = ListedColormap(sns.color_palette("husl",360).as_hex())

    normalize = mpl.colors.Normalize(vmin=0., vmax=360.)
    
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    
    #pre vs. log dist colored by station
    uniq = list(set(sta))
    
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [pre[j] for j in indexlist]
        az = [azlist[j] for j in indexlist]
        colors = [cmap(normalize(value)) for value in az]
        
        fig = plt.figure(figsize=(10.5,9))
        
        plt.scatter(x, y, s=4, color=colors)
        plt.xlabel(r'$\log_{10} R_{rup}$')#, fontsize = fontsize)
        plt.ylabel(r'$\log_{10} PGA$')#, fontsize = fontsize)
        plt.ylim([-7,-1])
        plt.xlim([1,2.4])
        
        fig.subplots_adjust(left=0.31,right=0.97, top = 0.90, bottom = 0.15)
        cbar_ax = fig.add_axes([0.16, 0.15, 0.02, 0.75])
        cbar = plt.colorbar(s_m, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position('left')
        cbar.set_label(r'$azimuth( ^{\circ}$)')#"$azimuth$ (\u00b0)"
        cbar.ax.tick_params()
        cbar_ax.yaxis.set_ticks_position('left')

        plt.savefig(workinghome + '/' + foldername + '/' + outname + '/stationplots' + '/az_pre_vs_dist_'+str(uniq[i])+'.png')
        
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [obs[j] for j in indexlist]
        az = [azlist[j] for j in indexlist]
        colors = [cmap(normalize(value)) for value in az]
        
        fig = plt.figure(figsize=(10.5,9))
        
        plt.scatter(x, y, s=4, color=colors)
        plt.xlabel(r'$\log_{10} R_{rup}$')#, fontsize = fontsize)
        plt.ylabel(r'$\log_{10} PGA$')#, fontsize = fontsize)
        plt.ylim([-7,-1])
        plt.xlim([1,2.4])
        
        fig.subplots_adjust(left=0.31,right=0.97, top = 0.90, bottom = 0.15)
        cbar_ax = fig.add_axes([0.16, 0.15, 0.02, 0.75])
        cbar = plt.colorbar(s_m, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position('left')
        cbar.set_label(r'$azimuth (^{\circ}$)')#"$azimuth$ (\u00b0)"
        cbar.ax.tick_params()
        cbar_ax.yaxis.set_ticks_position('left')

        plt.savefig(workinghome + '/' + foldername + '/' + outname + '/stationplots' + '/az_obs_vs_dist_'+str(uniq[i])+'.png')


    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [mag[j] for j in indexlist]
        y = [obs[j] for j in indexlist]
        az = [azlist[j] for j in indexlist]
        colors = [cmap(normalize(value)) for value in az]
        
        fig = plt.figure(figsize=(10.5,9))
        
        plt.scatter(x, y, s=4, color=colors)
        plt.xlabel('M', fontweight = 'bold')#, fontsize = fontsize)
        plt.ylabel(r'$\log_{10} PGA$')#, fontsize = fontsize)
        plt.ylim([-7,-1])
        plt.xlim([2.5,5])

        fig.subplots_adjust(left=0.31,right=0.97, top = 0.90, bottom = 0.15)
        cbar_ax = fig.add_axes([0.16, 0.15, 0.02, 0.75])
        cbar = plt.colorbar(s_m, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position('left')
        cbar.set_label(r'$azimuth (^{\circ}$)')#"$azimuth$ (\u00b0)"
        cbar.ax.tick_params()
        cbar_ax.yaxis.set_ticks_position('left')

        plt.savefig(workinghome + '/' + foldername + '/' + outname + '/stationplots' + '/az_obs_vs_mag_'+str(uniq[i])+'.png')
        
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [pre[j]- obs[j] for j in indexlist]
        az = [azlist[j] for j in indexlist]
        colors = [cmap(normalize(value)) for value in az]
        
        fig = plt.figure(figsize=(10.5,9))
        
        plt.scatter(x, y, s=4, color=colors)
        plt.xlabel('r$\log_{10} R_{rup}$')#, fontsize = fontsize)
        plt.ylabel('residuals (obs-pre)')#, fontsize = fontsize)
#        plt.ylim([-4,-4])
        plt.xlim([1,2.4])
        
        fig.subplots_adjust(left=0.31,right=0.97, top = 0.90, bottom = 0.15)
        cbar_ax = fig.add_axes([0.16, 0.15, 0.02, 0.75])
        cbar = plt.colorbar(s_m, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position('left')
        cbar.set_label(r'$azimuth (^{\circ}$)')#"$azimuth$ (\u00b0)"
        cbar.ax.tick_params()
        cbar_ax.yaxis.set_ticks_position('left')

        plt.savefig(workinghome + '/' + foldername + '/' + outname + '/stationplots' + '/az_resid_vs_dist_'+str(uniq[i])+'.png')


#    for i in range(len(uniq)):
#        indx = uniq[i]
#        indexlist =np.where(sta==indx)[0]
#        x = [mag[j] for j in indexlist]
#        y = [pre[j]- obs[j] for j in indexlist]
#        az = [azlist[j] for j in indexlist]
#        colors = [cmap(normalize(value)) for value in az]
#        
#        fig = plt.figure(figsize=(10,10))
#        
#        plt.scatter(x, y, s=4, color=colors)
#        plt.xlabel('magnitude')
#        plt.ylabel('residuals (obs-pre)')
##        plt.ylim([-4,-4])
#        
#        fig.subplots_adjust(right=0.82)
#        cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
#        cbar = plt.colorbar(s_m, cax=cbar_ax)
#        cbar.set_label(ur"azimuth (\u00b0)")#"$azimuth$ (\u00b0)"
#        cbar.ax.tick_params()
#        plt.savefig(workinghome + '/' + foldername + '/' + outname + '/stationplots' + '/az_resid_vs_mag_'+str(uniq[i])+'.png')

    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [mag[j] for j in indexlist]
        y = [pre[j] for j in indexlist]
        az = [azlist[j] for j in indexlist]
        colors = [cmap(normalize(value)) for value in az]
        
        fig = plt.figure(figsize=(10.5,9))
        
        plt.scatter(x, y, s=4, color=colors)
        plt.xlabel('M', fontweight = 'bold')#, fontsize = fontsize)
        plt.ylabel(r'$\log_{10} PGA$')#, fontsize = fontsize)
        plt.ylim([-7,-1])
        plt.xlim([2.5,5])
        
        fig.subplots_adjust(left=0.31,right=0.97, top = 0.90, bottom = 0.15)
        cbar_ax = fig.add_axes([0.16, 0.15, 0.02, 0.75])
        cbar = plt.colorbar(s_m, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position('left')
        cbar.set_label(r'$azimuth (^{\circ}$)')#"$azimuth$ (\u00b0)"
        cbar.ax.tick_params()
        cbar_ax.yaxis.set_ticks_position('left')

        plt.savefig(workinghome + '/' + foldername + '/' + outname + '/stationplots' + '/az_pre_vs_mag_'+str(uniq[i])+'.png')
        
    fig = plt.figure(figsize=(10.5,9))
    x = np.log10(dist)
    y = pre
    az = azlist[j]
    plt.scatter(x, y, s=4, color=colors)
    plt.xlabel(r'$\log_{10} R_{rup}$')#, fontsize = fontsize)
    plt.ylabel(r'$\log_{10} PGA$')#, fontsize = fontsize)
    plt.xlim([1,2.4])
    plt.ylim([-7,-1])
        
    fig.subplots_adjust(left=0.31,right=0.97, top = 0.90, bottom = 0.15)
    cbar_ax = fig.add_axes([0.16, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar_ax.yaxis.set_label_position('left')
    cbar.set_label(r'$azimuth( ^{\circ}$)')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    cbar_ax.yaxis.set_ticks_position('left')


    plt.savefig(workinghome + '/' + foldername + '/' + outname + '/az_pre_vs_dist.png')
    
    
    vircmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(mag), vmax=max(mag))
    
    s_m = mpl.cm.ScalarMappable(cmap = vircmap, norm=normalize)
    s_m.set_array([]) 
    
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        x = [np.log10(dist[j]) for j in indexlist]
        y = [pre[j] for j in indexlist]
        m = [mag[j] for j in indexlist]
    
        colors = [vircmap(normalize(value)) for value in m]
        
        fig = plt.figure(figsize=(10.5,9))
        
        plt.scatter(x, y, s=4, color=colors)
        plt.xlabel(r'$\log_{10} R_{rup}$')#, fontsize = fontsize)
        plt.ylabel(r'$\log_{10} PGA$')#, fontsize = fontsize)
        plt.xlim([1,2.4])
        plt.ylim([-7,-1])
        
        fig.subplots_adjust(left=0.31,right=0.97, top = 0.90, bottom = 0.15)
        cbar_ax = fig.add_axes([0.16, 0.15, 0.02, 0.75])
        cbar = plt.colorbar(s_m, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position('left')
        cbar.set_label(r'$azimuth( ^{\circ}$)')#"$azimuth$ (\u00b0)"
        cbar.ax.tick_params()
        cbar_ax.yaxis.set_ticks_position('left')

        plt.savefig(workinghome + '/' + foldername + '/' + outname + '/stationplots' + '/mag_pre_vs_dist_'+str(uniq[i])+'.png')
    
    plt.close('all')

# '/Users/aklimase/Documents/GMM_ML/Talapas_run/nostressdrop/'
#    f = glob.glob(workingdir + 'model_AICtestnosite*/AICsig.txt')
#savename = 'kappa'
      
def plot_AIC(workingdir, f, savename):
    #def plot AIC   
#    workingdir = '/Users/aklimase/Documents/GMM_ML/'
#    f = glob.glob(workingdir + 'model_AICtestnosite*/AICsig.txt')
    f = sorted(f)
    
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    color = ['blue', 'purple', 'red','orange', 'darkgreen', 'magenta']
    
    for i in range(len(f)):
        data = np.genfromtxt(f[i],  names = True, dtype = None)
        label = f[i].split('/')[-1].split('_')[1][0]
        hunits_list = data['hiddenunits']
        AIClist = np.asarray(data['AIC'])
        sigmalist = data['sigma_valid']
        
        ax1.scatter(hunits_list, AIClist, s=20, label = label, c =  color[i], edgecolor = color[i])
        ax1.plot(hunits_list, AIClist,c = color[i])
        ax1.set_ylabel('AIC')
        ax2.scatter(hunits_list, sigmalist, s=20, label = label, c =  color[i], edgecolor =color[i])
        ax2.plot(hunits_list, sigmalist, c =  color[i])
        plt.xlabel('number of hidden units')
        ax2.set_ylabel('sigma')    
    ax1.set_xlim([-1,51]) 
    ax2.set_xlim([-1,51])
    ax1.set_ylim([-3000,3000])
    ax2.set_ylim([0.85,1.05])   
#    ax1.tick_params(axis='both', which='major', labelsize=16)
#    ax2.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 16)
    plt.tight_layout()
#    plt.show()
#    plt.savefig(workingdir + '/'+ 'AICsig_' + savename + '.png')
    
#    f = sorted(f)
    
#    fig = plt.figure(figsize=(10,8))
#    color = ['blue', 'purple', 'red','orange', 'darkgreen', 'magenta']
#    
#    for i in range(len(f)):
#        data = np.genfromtxt(f[i],  names = True, usecols = [0,1,2], dtype = None)
#        label = f[i].split('/')[-2].split('_')[2]
#        hunits_list = data['hiddenunits']
#        AIClist = data['AIC']
#        sigmalist = data['sigma']
#        plt.scatter(AIClist, sigmalist, label = label,c =  color[i], edgecolor = color[i], s = 15)
#        
#    ax1.tick_params(axis='both', which='major', labelsize=16)
#    ax2.tick_params(axis='both', which='major', labelsize=16)
#    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 16)
    plt.tight_layout()
    plt.show()
    plt.savefig(workingdir + '/'+ 'AICvssigma_' + savename + '.png')
    
def plot_AIC_layers(workingdir, f, savename):
    #def plot AIC   
#    workingdir = '/Users/aklimase/Documents/GMM_ML/'
#    f = glob.glob(workingdir + 'modelsearch_kappa/AICsig_*.txt')
    f = sorted(f)
    
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    color = ['blue', 'green', 'red']
    marker = ['o','^','x']
    
    for i in range(len(f)):
        data = np.genfromtxt(f[i],  names = True, dtype = None)
        label = data['hiddenlayers'][0]
    
        AIClist = data['AIC']
        sigmalist = data['sigma_valid']
        hunitslist = data['hiddenunits']
        hunits = []
        for j in range(len(AIClist)):
        #for each
            if label == 1:
                hunits.append([hunitslist[j]])
            if label == 2:
                hunits.append([hunitslist[j].split('_')[0], hunitslist[j].split('_')[1]])
            if label == 3:
                hunits.append([hunitslist[j].split('_')[0], hunitslist[j].split('_')[1], hunitslist[j].split('_')[2]])
        inlayer= 2
        mlist = []
        for k in range(len(AIClist)):
            m = int(hunits[k][0])*(inlayer + 1)
            for j in range(len(hunits[k])-1):
                m += int(hunits[k][j+1])*(int(hunits[k][j])+1)
            m += 1*(int(hunits[k][-1])+1)
            mlist.append(m)
        ax1.scatter(mlist, AIClist, s=30, label = label, c =  color[i], edgecolor = color[i], marker = marker[i])
        ax1.set_ylabel('AIC')
        ax2.scatter(mlist, sigmalist, s=30, label = label, c =  color[i], edgecolor =color[i], marker = marker[i])
        plt.xlabel('number of parameters')
        ax2.set_ylabel(r'$\sigma$ residuals')    
    ax1.set_xlim([0,400]) 
    ax2.set_xlim([0,400])
    ax1.set_ylim([-5500,3000])
    ax2.set_ylim([0.75,1.05]) 
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax1.locator_params(nbins=6)
    ax2.locator_params(nbins=6)
#
#    ax1.legend(loc=1, title="layers", borderaxespad=0., frameon=True, fontsize = 20, labelspacing = 0.1)
#    ax1.get_legend().get_title().set_fontsize('22')
    plt.suptitle('Neural Network with ' + savename, fontsize = 24)
    plt.tight_layout()
#    plt.show()    
    plt.savefig(workingdir + 'AICvssigma_' + savename + '.png')
    
    index = np.where(AIClist == min(AIClist))[0][0]
    print 'minimum AIC configuration: ', min(AIClist), hunits[index], mlist[index]
    
#    f = glob.glob(workingdir + 'modelsearch_nosite/AIC*')
#
#    AICall = np.asarray([])
#    unitslistall = np.asarray([])
#    for i in range(len(f)):
#        data = np.genfromtxt(f[i],  names = True,  dtype = None)
#        #label = f[i].split('/')[-2].split('_')[2]
#        hunits_list = np.asarray(data['hiddenunits'])
#        AIClist = np.asarray(data['AIC'])
#        sigmalist = data['sigma_valid']
#        AICall = np.concatenate((AICall, AIClist),axis = None)
#        unitslistall = np.concatenate((unitslistall, hunits_list),axis = None)
#    
#    print [x for y, x in sorted(zip(AICall, unitslistall))][0:10]
#    print sorted(AICall)[0:10]

#    f = sorted(f)
    
#    fig = plt.figure(figsize=(10,8))
#    color = ['blue', 'purple', 'red','orange', 'darkgreen', 'magenta']
#    
#    for i in range(len(f)):
#        data = np.genfromtxt(f[i],  names = True, usecols = [0,1,2], dtype = None)
#        label = f[i].split('/')[-2].split('_')[2]
#        hunits_list = data['hiddenunits']
#        AIClist = data['AIC']
#        sigmalist = data['sigma']
#        plt.scatter(AIClist, sigmalist, label = label,c =  color[i], edgecolor = color[i], s = 15)
#        
#    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0., fontsize = 16)
#    plt.tight_layout()
#    plt.show()
#    plt.savefig(workingdir + '/'+ 'AICvssigma_' + savename + '.png')


def plotsiteterms(obsln, preln, sta, gmpename, foldername, workinghome):
    resid = obsln - preln
    topdir = '/Users/aklimase/Documents/GMM_ML'
    siteterms = topdir + '/models/pckl/' + gmpename + '/r/results_site.csv'
    data = np.genfromtxt(siteterms,delimiter=",", skip_header = 1, dtype = None, encoding = None)

    siteterm = data['f1']
    names = data['f0']
    
    uniq = sorted(list(set(sta)))
    sitemean = []
    for i in range(len(uniq)):
        indx = uniq[i]
        indexlist =np.where(sta==indx)[0]
        sitemean.append(np.mean([resid[j] for j in indexlist]))
        
    #write sitemean to a file
    data = np.transpose(np.asarray([uniq, sitemean], dtype = object))
    header = 'station' + '\t' + 'sitemean'
    np.savetxt(workinghome + '/' + foldername + '/sitemean.txt', data, header = header, delimiter = '\t', fmt = ['%s','%10.5f'])
  
    rval, pval = pearsonr(siteterm, sitemean)
    power = statsmodels.stats.power.tt_solve_power(effect_size = rval, nobs = len(siteterm), alpha = 0.05)
    
    label1 = 'Pearson R: ' + "{:.4f}".format(rval)
    label3 = 'power: ' + "{:.4f}".format(power)
    label2 = 'pvalue: ' + "{:.4f}".format(pval) 
    
    plt.figure(figsize = (10,10))
    plt.scatter(siteterm, sitemean, s=100,edgecolor = None)
    plt.plot([-1.5,1], [-1.5,1], ls = '--', color = 'black')
    for i in range(len(names)):
        an = plt.annotate(names[i], xy = (siteterm[i] + 0.05, sitemean[i]))
        an.draggable()
        plt.axis('scaled')
        plt.xlabel('GMPE site term')
        plt.ylabel('ANN mean site residual')
        plt.xlim(-1.5,1)
        plt.ylim(-1.5,1)
        plt.annotate(label1 + '\n' + label2 + '\n' + label3, xy=(0.65, 0.02), xycoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))

        plt.tight_layout()
        plt.savefig(workinghome + '/' + foldername + '/' + 'siteterms_' + gmpename + '.png')


        
def plot_dist_curves(preln, mag, dist, workinghome, foldername, title, n):

    #go from log 10 to ln
    pre = preln*np.log10(np.e)
#    obs = obs*np.log10(np.e)
    #pga vs distance for various magnitudes
    fig = plt.figure(figsize=(10,8))
    plt.title('site = ' + title)

    color = ['blue', 'green', 'red', 'purple', 'pink', 'yellow', 'black']
    for i in range(len(mag)):
        plt.plot(np.log10(dist[i]), pre[i], color = color[i], label = mag[i][0])
    plt.legend(loc = 'lower left')
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-6, -1.5])
    plt.xlim([1,2.5])
    plt.tight_layout()

#    plt.show()
    plt.savefig(workinghome + '/' + foldername + '/' + 'curves' + str(n) + '/dist_' + title + '.png')

def plot_mag_curves(preln, mag, dist, workinghome, foldername, title, n):

    #go from log 10 to ln
    pre = preln*np.log10(np.e)

    fig = plt.figure(figsize=(10,8))
    plt.title('site = ' + title)

    color = ['blue', 'green', 'red', 'purple', 'pink', 'yellow', 'black']
    for i in range(len(dist)):
        plt.plot(mag[i], pre[i], color = color[i], label = dist[i][0])
    plt.legend(loc = 'upper left', ncol = 2)
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-6, -1.5])
    plt.xlim([2.5,6])
    plt.tight_layout()
#    plt.show()
    plt.savefig(workinghome + '/' + foldername + '/' + 'curves' + str(n) + '/mag_' + title + '.png')
    plt.close()
    
    
def setup_test_curves(site, scaler, workinghome, foldername, siteparams, ANN_list, ndir):
    num_models = len(ANN_list)
    
    if site == '5coeff':
        mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
        dist = np.arange(1.,200.,0.5)
        
        mag_curve = np.array([len(dist)*[x] for x in mag_test])
        dist_curve = np.array([dist for i in range(len(mag_test))])
        y_curve = []
        
        for i in range(len(mag_test)):
            ###################    
            d = {'mw': mag_curve[i], 'R': dist_curve[i]}    
            
            df = pd.DataFrame(data=d)
            x_curve_scale = scaler.transform(df)
            
            pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
            avg_pre = np.average(pre_list, axis = 0)
            y_curve.append(avg_pre)

        plot_dist_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = 'all', n = ndir)
    
    else:
        for k in range(len(siteparams[0])):
            sitename = siteparams[0][k]
            if site == 'kappa':
                siteparam = siteparams[1][k]
            else: 
                siteparam = siteparams[2][k]
        
            mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
            dist = np.arange(1.,200.,0.5)
        
            mag_curve = np.array([len(dist)*[x] for x in mag_test])
            site_curve = np.array([len(dist)*[siteparam] for i in range(len(mag_test))])
            dist_curve = np.array([dist for i in range(len(mag_test))])
            ####################
            y_curve = []
        
            for i in range(len(mag_test)):
                ###############
                d = {'mw': mag_curve[i],'R': dist_curve[i],'vs30': site_curve[i]}    
            
                df = pd.DataFrame(data=d)
                x_curve_scale = scaler.transform(df)
                pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
                avg_pre = np.average(pre_list, axis = 0)
                y_curve.append(avg_pre)

            
            plot_dist_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = sitename, n = ndir)
        
        
    if site == '5coeff':
        mag_test = np.arange(2.7,4,0.1)
        dist = np.arange(25.,200.,25)

        n = len(dist)
        p = len(mag_test)
            
        mag_curve = np.array([mag_test for i in range(n)])
        dist_curve = np.array([p*[x] for x in dist])
        #########
        y_curve = []
        
        for i in range(n):
            ###############    
            d = {'mw': mag_curve[i], 'R': dist_curve[i]}    
            
            df = pd.DataFrame(data=d)
            x_curve_scale = scaler.transform(df)
            pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
            avg_pre = np.average(pre_list, axis = 0)
            y_curve.append(avg_pre)

        plot_mag_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = 'all',n = ndir)

        
    else:
        for k in range(len(siteparams[0])):
            sitename = siteparams[0][k]
            if site == 'kappa':
                siteparam = siteparams[1][k]
            else: 
                siteparam = siteparams[2][k]
            mag_test = np.arange(2.7,4,0.1)
            dist = np.arange(25.,200.,25)
            
            n = len(dist)
            p = len(mag_test)
            
            mag_curve = np.array([mag_test for i in range(n)])
            site_curve = np.array([p*[siteparam] for i in range(n)])
            dist_curve = np.array([p*[x] for x in dist])
            y_curve = []
            
            for i in range(n):
                d = {'mw': mag_curve[i], 'R': dist_curve[i],'vs30': site_curve[i]}    
                df = pd.DataFrame(data=d)
                x_curve_scale = scaler.transform(df)
#                y_curve.append(ANN.predict(x_curve_scale))
                pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
                avg_pre = np.average(pre_list, axis = 0)
                y_curve.append(avg_pre)
    
            plot_mag_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = sitename, n = ndir)

    plt.close('all')
    
    
#with stressdrop
def setup_test_curves_sdrop(site, scaler, workinghome, foldername, siteparams, ANN_list, ndir):
    num_models = len(ANN_list)
    
    if site == 'none':
        mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
        dist = np.arange(1.,200.,2)
        #########
        s = np.log10(5)*np.ones(len(dist))
        
        mag_curve = np.array([100*[x] for x in mag_test])
        dist_curve = np.array([dist for i in range(len(mag_test))])
        ##########3
        sdrop_curve = np.array([s for i in range(len(mag_test))])
        y_curve = []
        
        for i in range(len(mag_test)):
            ###################    
            d = {'mw': mag_curve[i], 'sdrop': sdrop_curve[i], 'R': dist_curve[i]}    
            
            df = pd.DataFrame(data=d)
            x_curve_scale = scaler.transform(df)
            
            pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
#            yhats = [model.predict(testX) for model in members]
            avg_pre = np.average(pre_list, axis = 0)
#            y_curve.append(ANN.predict(x_curve_scale))
            y_curve.append(avg_pre)

        plot_dist_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = 'all', n = ndir)
    
    else:
        for k in range(len(siteparams[0])):
            sitename = siteparams[0][k]
            if site == 'kappa':
                siteparam = siteparams[1][k]
            else: 
                siteparam = siteparams[2][k]
                
        
            mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
            dist = np.arange(1.,200.,2)
            #########
            s = np.log10(5)*np.ones(len(dist))
        
            mag_curve = np.array([100*[x] for x in mag_test])
            site_curve = np.array([100*[siteparam] for i in range(len(mag_test))])
            dist_curve = np.array([dist for i in range(len(mag_test))])
            ####################
            sdrop_curve = np.array([s for i in range(len(mag_test))])
            y_curve = []
        
            for i in range(len(mag_test)):
                ###############
                d = {'mw': mag_curve[i],'sdrop': sdrop_curve[i],'R': dist_curve[i],'vs30': site_curve[i]}    
            
                df = pd.DataFrame(data=d)
                x_curve_scale = scaler.transform(df)
                pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
                avg_pre = np.average(pre_list, axis = 0)
                y_curve.append(avg_pre)

            
            plot_dist_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = sitename, n = ndir)
        
        
    if site == 'none':
        mag_test = np.arange(2.7,4,0.1)
        dist = np.arange(25.,200.,25)
        s = np.log10(5)*np.ones(len(dist))

        
        n = len(dist)
        p = len(mag_test)
            
        mag_curve = np.array([mag_test for i in range(n)])
        dist_curve = np.array([p*[x] for x in dist])
        #########
        sdrop_curve = np.array([s for i in range(n)])
        y_curve = []
        
        for i in range(n):
            ###############    
            d = {'mw': mag_curve[i],'sdrop': sdrop_curve[i], 'R': dist_curve[i]}    
            
            df = pd.DataFrame(data=d)
            x_curve_scale = scaler.transform(df)
            pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
            avg_pre = np.average(pre_list, axis = 0)
            y_curve.append(avg_pre)


        plot_mag_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = 'all',n = ndir)

        
    else:
        for k in range(len(siteparams[0])):
            sitename = siteparams[0][k]
            if site == 'kappa':
                siteparam = siteparams[1][k]
            else: 
                siteparam = siteparams[2][k]
            mag_test = np.arange(2.7,4,0.1)
            dist = np.arange(25.,200.,25)
            s = np.log10(5)*np.ones(len(mag_test))
            
            n = len(dist)
            p = len(mag_test)
            
            mag_curve = np.array([mag_test for i in range(n)])
            site_curve = np.array([p*[siteparam] for i in range(n)])
            dist_curve = np.array([p*[x] for x in dist])
            sdrop_curve = np.array([s for i in range(n)])
            y_curve = []
            
            for i in range(n):
                d = {'mw': mag_curve[i], 'sdrop': sdrop_curve[i], 'R': dist_curve[i],'vs30': site_curve[i]}    
                df = pd.DataFrame(data=d)
                x_curve_scale = scaler.transform(df)
#                y_curve.append(ANN.predict(x_curve_scale))
                pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
                avg_pre = np.average(pre_list, axis = 0)
                y_curve.append(avg_pre)
    
            plot_mag_curves(preln = np.asarray(y_curve), mag = mag_curve, dist = dist_curve, workinghome=workinghome, foldername = foldername, title = sitename, n = ndir)

    plt.close('all')

    
def residual_histo(y_train, y_valid, y_test, ANN_train_predicted, ANN_valid_predicted, ANN_test_predicted, workinghome, foldername, n):
    sns.set(font_scale=2)
    sns.set_style('whitegrid')


    #test,training, validation data
    resid_train = np.asarray(y_train) - ANN_train_predicted
    std_train = round(np.std(resid_train),4)
    mean_train = round(np.mean(resid_train),4)
    
    resid_valid = np.asarray(y_valid) - ANN_valid_predicted
    std_valid = round(np.std(resid_valid),4)
    mean_valid = round(np.mean(resid_valid),4)
    
    resid_test = np.asarray(y_test) - ANN_test_predicted
    std_test = round(np.std(resid_test),4)
    mean_test = round(np.mean(resid_test),4)
    
    #histo of residuals
    fig = plt.figure(figsize=(10,9))#tight_layout=True)
    ax = fig.add_subplot(111)
    # N is the count in each bin, bins is the lower-limit of the bin
    labeltrain = r'$\mu_{train}:$ ' + str('{:.4f}'.format(mean_train)) + '\n' + r'$\sigma_{train}:$ ' + str('{:.4f}'.format(std_train))
    labelvalid = r'$\mu_{valid}:$ ' + str('{:.4f}'.format(mean_valid)) + '\n' + r'$\sigma_{valid}:$ ' + str('{:.4f}'.format(std_valid))
    labeltest = r'$\mu_{test}:$ ' + str('{:.4f}'.format(mean_test)) + '\n' + r'$\sigma_{test}:$ ' + str('{:.4f}'.format(std_test)) 
    
    label=[labeltrain, labelvalid, labeltest]
    N, bins, patches = ax.hist([resid_train, resid_valid, resid_test], color=['blue', 'green', 'red'], label = label, bins=60)
    ax.set_xlim(-4,4)
    #    ax.text(0.7, 0.9,'std: ' + str(std) + '\n' + 'mean: ' + str(mean),  transform=ax.transAxes)
    ax.set_xlabel('residuals')
    ax.set_ylabel('counts')
    plt.legend(fontsize = 20)
    plt.title('Residuals')
    plt.tight_layout()
    plt.savefig(workinghome + '/' + foldername + '/histoln' + str(n) + '.png')
    plt.close()



def plot_dist_curves_scatter(preln_curve, mag_curve, dist_curve, pre_scatter, obs_scatter, dist_scatter, mag_scatter, workinghome, foldername, title):

    #go from log 10 to ln
    pre_curve = preln_curve*np.log10(np.e)
    pre_scatter_log10 = pre_scatter*np.log10(np.e)
    #obs = obs*np.log10(np.e)
    #pga vs distance for various magnitudes
    fig = plt.figure(figsize=(10,8.5))
    plt.title('site = ' + title)
    
    #
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(mag_scatter), vmax=max(mag_scatter))
    colors = [cmap(normalize(value)) for value in mag_scatter]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])

    #color = ['blue', 'green', 'red', 'purple', 'pink', 'yellow', 'black']
    for i in range(len(mag_curve)):
        plt.plot(np.log10(dist_curve[i]), pre_curve[i], color = cmap(normalize(mag_curve[i][0])), label = mag_curve[i][0])
    
    plt.scatter(np.log10(dist_scatter), pre_scatter_log10, color = colors, s = 2, alpha = 0.3)
    plt.legend(loc = 'lower left', title = 'M', fontweight = 'bold')
    plt.xlabel(r'$\log_{10} R_{rup}$')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-6, -1.5])
    plt.xlim([1,2.5])
    
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label('M', fontweight = 'bold')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()

#    plt.tight_layout()

    plt.show()
    plt.savefig(workinghome + '/' + foldername + '/' + 'curves' + '/scatter_dist_' + title + '.png')

def plot_mag_curves_scatter(preln_curve, mag_curve, dist_curve, pre_scatter, obs_scatter, mag_scatter, dist_scatter, workinghome, foldername, title):

    #go from log 10 to ln
    pre_curve = preln_curve*np.log10(np.e)
    pre_scatter_log10 = pre_scatter*np.log10(np.e)

    fig = plt.figure(figsize=(10,8.4))
    plt.title('site = ' + title)
    
    cmap = mpl.cm.get_cmap('plasma')
    normalize = mpl.colors.Normalize(vmin=min(dist_scatter), vmax=max(dist_scatter))
    colors = [cmap(normalize(value)) for value in dist_scatter]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])

#    color = ['blue', 'green', 'red', 'purple', 'pink', 'yellow', 'black']
    for i in range(len(dist_curve)):
        plt.plot(mag_curve[i], pre_curve[i], color =  cmap(normalize(dist_curve[i][0])), label = dist_curve[i][0])
    
    plt.scatter(mag_scatter, pre_scatter_log10, color = colors, s = 2, alpha = 0.3)
    
    plt.legend(loc = 'upper left', title = r'$\log_{10} R_{rup}$', ncol = 2)
    plt.xlabel('M', fontweight = 'bold')
    plt.ylabel(r'$\log_{10} PGA$')
    plt.ylim([-6, -1.5])
    plt.xlim([2.8, 4.5])
    
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(r'$R_{rup}$')#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params()
    
    #plt.tight_layout()
#    plt.show()
    plt.savefig(workinghome + '/' + foldername + '/' + 'curves' + '/scatter_mag_' + title + '.png')
    plt.close()

    
def setup_test_curves_scatter(site, scaler, workinghome, foldername, siteparams, ANN_list, pre_scatter, obs_scatter, dist_scatter, mag_scatter, sta):
    num_models = len(ANN_list)
    
    if site == '5coeff':
        mag_test = [2.8,3.2,3.6,4]
        dist = np.arange(1.,200.,0.5)
        
        mag_curve = np.array([len(dist)*[x] for x in mag_test])
        dist_curve = np.array([dist for i in range(len(mag_test))])
        y_curve = []
        
        for i in range(len(mag_test)):
            ###################    
            d = {'mw': mag_curve[i], 'R': dist_curve[i]}    
            
            df = pd.DataFrame(data=d)
            x_curve_scale = scaler.transform(df)
            
            pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
#            yhats = [model.predict(testX) for model in members]
            avg_pre = np.average(pre_list, axis = 0)
#            y_curve.append(ANN.predict(x_curve_scale))
            y_curve.append(avg_pre)

        plot_dist_curves_scatter(preln_curve = np.asarray(y_curve), mag_curve = mag_curve, dist_curve = dist_curve, pre_scatter = pre_scatter, obs_scatter = np.asarray(obs_scatter), dist_scatter = np.asarray(dist_scatter), mag_scatter = np.asarray(mag_scatter), workinghome=workinghome, foldername = foldername, title = 'all')
    
    else:
        for k in range(len(siteparams[0])):
            sitename = siteparams[0][k]
            if site == 'kappa':
                siteparam = siteparams[1][k]
            else: 
                siteparam = siteparams[2][k]
                
            uniq = list(set(sta))
            for i in range(len(uniq)):
                indx = uniq[i]
                indexlist =np.where(sta==indx)[0]
                pre_scatter_sta = np.asarray([pre_scatter[j] for j in indexlist])
                obs_scatter_sta = np.asarray([obs_scatter[j] for j in indexlist])
                dist_scatter_sta = np.asarray([dist_scatter[j] for j in indexlist])
                mag_scatter_sta = np.asarray([mag_scatter[j] for j in indexlist])

        
            mag_test = [2.8,3.2,3.6,4]
            dist = np.arange(1.,200.,0.5)
        
            mag_curve = np.array([len(dist)*[x] for x in mag_test])
            site_curve = np.array([len(dist)*[siteparam] for i in range(len(mag_test))])
            dist_curve = np.array([dist for i in range(len(mag_test))])
            ####################
            y_curve = []
        
            for i in range(len(mag_test)):
                ###############
                d = {'mw': mag_curve[i],'R': dist_curve[i],'vs30': site_curve[i]}    
            
                df = pd.DataFrame(data=d)
                x_curve_scale = scaler.transform(df)
                pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
                avg_pre = np.average(pre_list, axis = 0)
                y_curve.append(avg_pre)

            
            plot_dist_curves_scatter(preln_curve = np.asarray(y_curve), mag_curve = mag_curve, dist_curve = dist_curve, pre_scatter = pre_scatter_sta, obs_scatter = np.asarray(obs_scatter_sta), dist_scatter = np.asarray(dist_scatter_sta), mag_scatter = np.asarray(mag_scatter_sta), workinghome=workinghome, foldername = foldername, title = sitename)
        
        
    if site == '5coeff':
        mag_test = np.arange(2.7,4.5,0.1)
        dist = np.arange(25.,250.,50.)

        n = len(dist)
        p = len(mag_test)
            
        mag_curve = np.array([mag_test for i in range(n)])
        dist_curve = np.array([p*[x] for x in dist])
        #########
        y_curve = []
        
        for i in range(n):
            ###############    
            d = {'mw': mag_curve[i], 'R': dist_curve[i]}    
            
            df = pd.DataFrame(data=d)
            x_curve_scale = scaler.transform(df)
            pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
            avg_pre = np.average(pre_list, axis = 0)
            y_curve.append(avg_pre)

        plot_mag_curves_scatter(preln_curve = np.asarray(y_curve), mag_curve = mag_curve, dist_curve = dist_curve,  pre_scatter = pre_scatter, obs_scatter = np.asarray(obs_scatter), dist_scatter = np.asarray(dist_scatter), mag_scatter = np.asarray(mag_scatter), workinghome=workinghome, foldername = foldername, title = 'all')

    else:
        for k in range(len(siteparams[0])):
            sitename = siteparams[0][k]
            if site == 'kappa':
                siteparam = siteparams[1][k]
            else: 
                siteparam = siteparams[2][k]
                
            uniq = list(set(sta))
            for i in range(len(uniq)):
                indx = uniq[i]
                indexlist =np.where(sta==indx)[0]
                pre_scatter_sta = np.asarray([pre_scatter[j] for j in indexlist])
                obs_scatter_sta = np.asarray([obs_scatter[j] for j in indexlist])
                dist_scatter_sta = np.asarray([dist_scatter[j] for j in indexlist])
                mag_scatter_sta = np.asarray([mag_scatter[j] for j in indexlist])

            mag_test = np.arange(2.7,4.5,0.1)
            dist = np.arange(25.,250.,50.)
            
            n = len(dist)
            p = len(mag_test)
            
            mag_curve = np.array([mag_test for i in range(n)])
            site_curve = np.array([p*[siteparam] for i in range(n)])
            dist_curve = np.array([p*[x] for x in dist])
            y_curve = []
            
            for i in range(n):
                d = {'mw': mag_curve[i], 'R': dist_curve[i],'vs30': site_curve[i]}    
                df = pd.DataFrame(data=d)
                x_curve_scale = scaler.transform(df)
                pre_list = [ANN_list[i].predict(x_curve_scale) for i in range(num_models)]
                avg_pre = np.average(pre_list, axis = 0)
                y_curve.append(avg_pre)
            plot_mag_curves_scatter(preln_curve = np.asarray(y_curve), mag_curve = mag_curve, dist_curve = dist_curve,  pre_scatter = pre_scatter_sta, obs_scatter = np.asarray(obs_scatter_sta), dist_scatter = np.asarray(dist_scatter_sta), mag_scatter = np.asarray(mag_scatter_sta), workinghome=workinghome, foldername = foldername, title = sitename)

    plt.close('all')
    
#################################   
def setup_curves_compare(site, scaler, workinghome, foldername, siteparams, ANN_list,vref, refsite, pre_scatter, obs_scatter, dist_scatter, mag_scatter, sta):
    num_models = len(ANN_list)
    sta  = np.asarray(list(sta))
        
    if len(pre_scatter) == 1:
        title = 'refsite_' + refsite
    else:
        title = 'refsite_scatter_' + refsite
        indexlist = np.where(sta==refsite)[0]
        pre_scatter = np.asarray([pre_scatter[j] for j in indexlist])
        obs_scatter = np.asarray([obs_scatter[j] for j in indexlist])
        dist_scatter = np.asarray([dist_scatter[j] for j in indexlist])
        mag_scatter = np.asarray([mag_scatter[j] for j in indexlist])
    
    #read in coefficients
    paramsfile = '/Users/aklimase/Documents/GMM_ML/models/pckl/database_' + site + '/r/results_fixed.csv'
    params = np.genfromtxt(paramsfile,delimiter=",", names = True, usecols = [1,2,3], dtype = float)
    n = len(params['Estimate'])
    if n == 5:
        a1,a2,a3,a4,a5 = params['Estimate']
    else:
        a1,a2,a3,a4,a5,a6 = params['Estimate']
        
    #read in results site
    #use PFO for siteterm bc vs30 = 763
    sitefile = '/Users/aklimase/Documents/GMM_ML/models/pckl/database_' + site + '/r/results_site.csv'
    siteterms_me = np.genfromtxt(sitefile,delimiter=",", names = True, usecols = [0,1,2], dtype = None)
    ind = np.where(siteterms_me['ID'] == refsite)[0][0]
    bias = siteterms_me['Bias'][ind]
            
    #get vref(vs30 or kappa) for the site
    if site =='kappa':
        sitetermfile = '/Users/aklimase/Documents/GMM_ML/catalog/tstar_site.txt'
        siteterms_ann = np.genfromtxt(sitetermfile, names = True, usecols = [0,1], dtype = None)
        ind = np.where(siteterms_ann['site'] == refsite)[0][0]
        vrefsite = siteterms_ann['tstars'][ind]
    else: #site is vs30 or 5coeff, vref is vs30
        sitetermfile = '/Users/aklimase/Documents/GMM_ML/catalog/vs30_sta.txt'
        siteterms_ann = np.genfromtxt(sitetermfile, names = True, usecols = [0,3], dtype = None)
        ind = np.where(siteterms_ann['Sta'] == refsite)[0][0]
        vrefsite = siteterms_ann['Vs30'][ind]


    mag_test = [2.8,3,3.2,3.4,3.6,3.8,4]
    dist = np.arange(1.,200.,0.5)
    
    mag_curve = np.array([len(dist)*[x] for x in mag_test])
    site_curve = np.array([len(dist)*[vrefsite] for i in range(len(mag_test))])
    dist_curve = np.array([dist for i in range(len(mag_test))])
    y_curve_ann = []
    y_curve_me = []
    
    #for each curve
    for i in range(len(mag_test)):
        ###################    
        if site =='5coeff':
            d = {'mw': mag_curve[i], 'R': dist_curve[i]}    
        else:   
            d = {'mw': mag_curve[i], 'R': dist_curve[i], 'vs30': site_curve[i]}    
        
        df = pd.DataFrame(data=d)
        x_curve_scale = scaler.transform(df)
        
        pre_list = [ANN_list[j].predict(x_curve_scale) for j in range(num_models)]
        avg_pre = np.average(pre_list, axis = 0)
        y_curve_ann.append(avg_pre)
                
        if n == 5:
            ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias
        else:
            ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i]+ a6*np.log(vrefsite/vref)  + bias 
        y_curve_me.append(ME_test_predicted)


    plot_dist_curves_both(preln_ANN = np.asarray(y_curve_ann), preln_ME = np.asarray(y_curve_me), mag_curve = mag_curve, dist_curve = dist_curve, pre_scatter = pre_scatter, obs_scatter = np.asarray(obs_scatter), dist_scatter = np.asarray(dist_scatter), mag_scatter = np.asarray(mag_scatter), workinghome=workinghome, foldername = foldername, title = title)
        
    mag_test = np.arange(2.7,5,0.1)
    dist = np.arange(25.,200.,25)

#    n = len(dist)
    p = len(mag_test)
        
    mag_curve = np.array([mag_test for i in range(len(dist))])
    site_curve = np.array([p*[vref] for i in range(len(mag_test))])
    dist_curve = np.array([p*[x] for x in dist])
    #########
    y_curve_ann = []
    y_curve_me = []
    
    for i in range(len(dist)):
        ###############    
        if site =='5coeff':
            d = {'mw': mag_curve[i], 'R': dist_curve[i]}
        else:
            d = {'mw': mag_curve[i], 'R': dist_curve[i], 'vs30': site_curve[i]}
        
        df = pd.DataFrame(data=d)
        x_curve_scale = scaler.transform(df)
        pre_list = [ANN_list[j].predict(x_curve_scale) for j in range(num_models)]
        avg_pre = np.average(pre_list, axis = 0)
        y_curve_ann.append(avg_pre)
        
        if n == 5:
            ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i] + bias
        else:
            ME_test_predicted = a1 + a2*mag_curve[i] + a3*(8.5-mag_curve[i])**2. + a4*np.log(dist_curve[i]) + a5*dist_curve[i]+ a6*np.log(vrefsite/vref) + bias 
        y_curve_me.append(ME_test_predicted)


    plot_mag_curves_both(preln_ANN = np.asarray(y_curve_ann), preln_ME = np.asarray(y_curve_me), mag_curve = mag_curve, dist_curve = dist_curve, pre_scatter = pre_scatter, obs_scatter = np.asarray(obs_scatter), dist_scatter = np.asarray(dist_scatter), mag_scatter = np.asarray(mag_scatter), workinghome=workinghome, foldername = foldername, title = title)

    plt.close('all')

def plot_dist_curves_both(preln_ANN, preln_ME, mag_curve, dist_curve, pre_scatter, obs_scatter, dist_scatter, mag_scatter, workinghome, foldername, title):
#    mpl.rcParams['font.size'] =22
    fontsize = 30

    #go from log 10 to ln
    pre_ANN = preln_ANN*np.log10(np.e)
    pre_ME = preln_ME*np.log10(np.e)
    pre_scatter_log10 = pre_scatter*np.log10(np.e)
    obs_scatter_log10 = obs_scatter*np.log10(np.e)

#    obs = obs*np.log10(np.e)
    #pga vs distance for various magnitudes
    fig = plt.figure(figsize=(10,9))
    #plt.title('site = ' + title)
    
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=2.7, vmax=5.5)
    colors = [cmap(normalize(value)) for value in mag_scatter]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])

    for i in range(len(mag_curve)):
        #plot twice with edgecolor
        plt.plot(np.log10(dist_curve[i]), pre_ANN[i].flatten(), color = 'white', lw=6)
        plt.plot(np.log10(dist_curve[i]), pre_ANN[i].flatten(), color = cmap(normalize(mag_curve[i][0])), label = mag_curve[i][0],  lw=4)
        plt.plot(np.log10(dist_curve[i]), pre_ME[i].flatten(), color = 'white', lw=6, ls = '--')
        plt.plot(np.log10(dist_curve[i]), pre_ME[i].flatten(), color = cmap(normalize(mag_curve[i][0])), ls = '--', lw=4)

    plt.scatter(np.log10(dist_scatter), obs_scatter_log10, facecolors='none',edgecolors=colors, s = 30, alpha = 0.8)

    plt.legend(loc = 'lower left', title = 'M', ncol = 1, fontsize =24, labelspacing = 0.1)
    plt.xlabel(r'$\log_{10} R_{rup}$')#, fontsize =fontsize)
    plt.ylabel(r'$\log_{10} PGA$')#, fontsize =fontsize)
    plt.ylim([-7, -1])
    plt.xlim([1,2.4])
    plt.tick_params(axis='both', which='major')#, labelsize=fontsize)

    
    fig.subplots_adjust(bottom=0.15)

    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label('M', fontweight = 'bold')#"$azimuth$ (\u00b0)"
    cbar_ax.tick_params(axis='both', which='major')#, labelsize=fontsize)

#    cbar.ax.tick_params()

#    plt.show()
    plt.savefig(workinghome + '/' + foldername + '/' + 'curves' + '/compare_dist_' + title + '.png')

def plot_mag_curves_both(preln_ANN, preln_ME, mag_curve, dist_curve, pre_scatter, obs_scatter, dist_scatter, mag_scatter,workinghome, foldername, title):
    fontsize = 30

    #go from log 10 to ln
    pre_ANN = preln_ANN*np.log10(np.e)
    pre_ME = preln_ME*np.log10(np.e)
    pre_scatter_log10 = pre_scatter*np.log10(np.e)
    obs_scatter_log10 = obs_scatter*np.log10(np.e)


    fig = plt.figure(figsize=(10,9))
    #plt.title('site = ' + title)
    
    cmap = mpl.cm.get_cmap('plasma')
    normalize = mpl.colors.Normalize(vmin=10., vmax=230.)
    colors = [cmap(normalize(value)) for value in dist_scatter]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])

#    color = ['blue', 'green', 'red', 'purple', 'pink', 'yellow', 'black']
    for i in range(len(dist_curve)):
        #plot twice for outline
        plt.plot(mag_curve[i], pre_ANN[i], color =  'white',lw=6)
        plt.plot(mag_curve[i], pre_ANN[i], color =  cmap(normalize(dist_curve[i][0])), label = dist_curve[i][0],lw=4)
        plt.plot(mag_curve[i], pre_ME[i], color =  'white',lw=6, ls = '--')
        plt.plot(mag_curve[i], pre_ME[i], color = cmap(normalize(dist_curve[i][0])), ls = '--',lw=4)

    plt.scatter(mag_scatter, obs_scatter_log10, facecolors='none',edgecolors=colors, s = 30, alpha = 0.8)

    plt.legend(loc = 'lower right', ncol = 1, title = r'$R_{rup}$', fontsize =24, labelspacing = 0.1)
    plt.xlabel('M', fontweight = 'bold')#, fontsize =fontsize)
    plt.ylabel(r'$\log_{10} PGA$')#, fontsize =fontsize)
    plt.ylim([-7, -1])
    plt.xlim([2.5,5])
    plt.tick_params(axis='both', which='major')#, labelsize=fontsize)

#    plt.tight_layout()
#    plt.show()
    
    fig.subplots_adjust(bottom=0.15)

    
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label(r'$R_{rup}$')#"$azimuth$ (\u00b0)"
    cbar_ax.tick_params(axis='both', which='major')#, labelsize=fontsize)

    plt.savefig(workinghome + '/' + foldername + '/' + 'curves/compare_mag_' + title + '.png')
    plt.close()
    

