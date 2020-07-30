#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:18:30 2019

@author: aklimase
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from xgboost import XGBRegressor
from scipy.stats import pearsonr
import statsmodels
import statsmodels.stats.power
import seaborn as sns
plt.style.use("classic")

sns.set_context("poster")
#sns.set(font_scale=2)
sns.set_style('whitegrid')

fontsize = 22
mpl.rcParams['font.size'] =fontsize

modelsite = 'kappa'
MEmodel = 'kappa'
modelsite_name = r'$\kappa_0$'
ANNmodel = 'kappa_hiddenlayers_3_units_6_8_6_final' # 'kappa_hiddenlayers_3_units_6_8_6' #'vs30_hiddenlayers_3_units_8_8_6'

#modelsite = 'nosite'
#MEmodel = '5coeff'
#modelsite_name = 'nosite' 
#ANNmodel = 'nosite_hiddenlayers_3_units_4_6_2_final' #'vs30_hiddenlayers_3_units_8_8_6'

#modelsite = 'vs30'
#modelsite_name = r'$V_{S30}$'   #r'$\kappa_0$'
#MEmodel = 'vs30'
#ANNmodel = 'vs30_hiddenlayers_3_units_8_8_6_final'

###################
workinghome = '/Users/aklimase/Documents/GMM_ML/'



t = workinghome + 'catalog/tstar_site.txt'
tstarcat = np.genfromtxt(t, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)
                         
v = workinghome + 'catalog/vs30_sta.txt'
vs30cat = np.genfromtxt(v, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)

sitename =  tstarcat['site']
k = tstarcat['tstars']
vs30 = vs30cat['Vs30']

#siteparams = [sitename, k, vs30]
d = {'sta': sitename,'kappa': k,'vs30': vs30}
df_siteparams = pd.DataFrame(data=d)

#read in model site mean
sitemean = workinghome + 'best_ANN_kfold/' + ANNmodel + '/sitemean.txt'
sitemeancat = np.genfromtxt(sitemean, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)
#ANNsiteresid = [sitemeancat['station'], sitemeancat['sitemean']]
d = {'sta': sitemeancat['station'],'sitemean': sitemeancat['sitemean']}
df_ANNsiteresid =  pd.DataFrame(data=d)

#site residual mixed effects
MEsitefile = workinghome + 'models/pckl/database_' + MEmodel + '/r/results_site.csv'
MEsiteterms = np.genfromtxt(MEsitefile,delimiter=",", names = True, usecols = [0,1,2], dtype = None)
MEsiteresid = [MEsiteterms['ID'], MEsiteterms['Bias']]
d = {'sta': MEsiteterms['ID'],'sitemean': MEsiteterms['Bias']}
df_MEsiteresid =  pd.DataFrame(data=d)

#plot k (andvs30) vs site terms



#def plot_sitepparams(x, y, names, xlabel, ylabel, title, savename):
#    rval, pval = pearsonr(x,y)
#    power = statsmodels.stats.power.tt_solve_power(effect_size = rval, nobs = len(x), alpha = 0.05)
#    
#    label1 = 'Pearson R: ' + "{:.4f}".format(rval)
#    label3 = 'power: ' + "{:.4f}".format(power)
#    label2 = 'pvalue: ' + "{:.4f}".format(pval) 
#    
#    plt.figure(figsize = (10,10))
#    plt.scatter(x,y, s=100,edgecolor = None)
#    for i in range(len(names)):
#        an = plt.annotate(names[i], xy = (x[i],y[i]))
#        an.draggable()
#    #    plt.axis('scaled')
#        plt.xlabel(xlabel, fontsize = fontsize)
#        plt.ylabel(ylabel, fontsize = fontsize)
#        plt.title(title)
#    #    plt.xlim(-1.5,1)
#        plt.ylim(-1.5,1)
#        plt.annotate(label1 + '\n' + label2 + '\n' + label3, xy=(0.02, 0.03), xycoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
#        plt.tick_params(axis='both', which='major', labelsize=fontsize)
#
#        plt.tight_layout()
#        plt.savefig(workinghome + 'paperfigs_comparesites/' + savename+ '.png')
##        plt.close()
#
#plot_sitepparams(x= df_siteparams['kappa'], y = df_MEsiteresid['sitemean'], names  = df_MEsiteresid['sta'], xlabel = r'$\kappa_0$', ylabel = 'site residual', title = r'ME '+ MEmodel +' site mean vs. $\kappa_0$', savename = 'ME' + MEmodel+ '_kappa')
#plot_sitepparams(x= df_siteparams['vs30'], y = df_MEsiteresid['sitemean'], names  = df_MEsiteresid['sta'], xlabel = r'$V_{S30}$', ylabel = 'site residual', title = r'ME '+ MEmodel +' site mean vs. $V_{S30}$', savename = 'ME' + MEmodel + '_vs30')
###
#plot_sitepparams(x= df_siteparams['kappa'], y = df_ANNsiteresid['sitemean'], names  = df_ANNsiteresid['sta'], xlabel = r'$\kappa_0$', ylabel = 'site residual', title = r'ANN ' + modelsite + ' site mean vs. $\kappa_0$', savename = 'ANN' + modelsite + '_kappa')
#plot_sitepparams(x= df_siteparams['vs30'], y = df_ANNsiteresid['sitemean'], names  = df_ANNsiteresid['sta'], xlabel = r'$V_{S30}$', ylabel = 'site residual', title = r'ANN ' + modelsite + ' site mean vs. $V_{S30}$', savename = 'ANN' + modelsite + '_vs30')
##
        
        
        
        
def plot_sitepparams_compare(x, y, names, xlabel, ylabel, title, savename, colorby, barname):
    rval, pval = pearsonr(x,y)
    power = statsmodels.stats.power.tt_solve_power(effect_size = rval, nobs = len(x), alpha = 0.05)
    
    label1 = 'Pearson R: ' + "{:.4f}".format(rval)
    label3 = 'power: ' + "{:.4f}".format(power)
    label2 = 'pvalue: ' + "{:.4f}".format(pval) 
    
    print names, colorby
#    #color by kappa or vs30
    #set the color map
    cmap = mpl.cm.viridis
#    cmap = mpl.cm.plasma

#    norm = mpl.colors.Normalize(vmin=min(colorby), vmax=max(colorby))
    norm = mpl.colors.Normalize(vmin=0.01, vmax=0.06)

    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=norm)
    s_m.set_array([])

#    fig, ax = plt.subplots()
    
    ax=sns.jointplot(x,y, kind = 'kde',stat_func=None, height=10, color = 'darkgray')
    ax.ax_joint.cla()

    
    #clear axis
    ax.ax_joint.scatter(x,y, s=30, edgecolor = s_m.to_rgba(colorby), c = s_m.to_rgba(colorby))
    ax.ax_joint.set_xlabel(xlabel)
    ax.ax_joint.set_ylabel(ylabel)
    ax.ax_joint.set_xlim(-1.5,1)
    ax.ax_joint.set_ylim(-1.5,1)
    
    
    for i in range(len(names)):
        an = ax.ax_joint.annotate(names[i], xy = (x[i],y[i]))
        an.draggable()
        
    ax.ax_joint.annotate(label1 + '\n' + label2 + '\n' + label3, xy=(0.57, 0.025), xycoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
    ax.ax_joint.plot([-2,2],[-2,2], ls = '--', color = 'black', zorder = -100)
    
    #add mean and std lines
    ax.ax_joint.axvline(x=np.mean(x), c='black', lw=1)
    ax.ax_joint.axvline(x=np.mean(x)+np.std(x), c='black', lw=1, ls = '--')
    ax.ax_joint.axvline(x=np.mean(x)-np.std(x), c='black', lw=1, ls = '--')
    
    ax.ax_joint.axhline(y=np.mean(y), c='black', lw=1)
    ax.ax_joint.axhline(y=np.mean(y)+np.std(x), c='black', lw=1, ls = '--')
    ax.ax_joint.axhline(y=np.mean(y)-np.std(x), c='black', lw=1, ls = '--')


    plt.subplots_adjust(bottom=0.21, right = 0.87)
    cbar_ax =  ax.fig.add_axes([0.14, 0.085, 0.60, 0.02]) 
    cbar = plt.colorbar(s_m, cax=cbar_ax, orientation = 'horizontal')
    cbar = plt.colorbar(s_m, cax=cbar_ax, orientation = 'horizontal', format="%.2f", ticks=[0.01,0.02,0.03,0.04,0.05,0.06])

    cbar.set_label(barname, fontsize = 22)#"$azimuth$ (\u00b0)"
    cbar.ax.tick_params(labelsize = 18)
    cbar.ax.set_xticklabels([0.01,0.02,0.03,0.04,0.05,0.06])
    
    plt.subplots_adjust(top=0.94)
    ax.fig.suptitle(title)
    
    plt.savefig(workinghome + 'paperfigs_comparesites/' + savename+ '_v2.png')

#plot_sitepparams_compare(x= df_MEsiteresid['sitemean'], y = df_ANNsiteresid['sitemean'], names  = df_ANNsiteresid['sta'], xlabel = 'MEML site residual', ylabel = 'ANN site residual', title = r'ANN vs MEML ' + modelsite_name + ' site residuals', savename = 'ANN_MEML_' + modelsite, colorby = df_siteparams[modelsite], barname = modelsite_name)
plot_sitepparams_compare(x= df_MEsiteresid['sitemean'], y = df_ANNsiteresid['sitemean'], names  = df_ANNsiteresid['sta'], xlabel = 'MEML site residual', ylabel = 'ANN site residual', title = r'ANN vs MEML ' + modelsite_name + ' site residuals', savename = 'ANN_MEML_' + modelsite, colorby = df_siteparams[modelsite], barname = modelsite_name)

#5coeff
            
def plot_sitepparams_compare_nocolor(x, y, names, xlabel, ylabel, title, savename):
    rval, pval = pearsonr(x,y)
    power = statsmodels.stats.power.tt_solve_power(effect_size = rval, nobs = len(x), alpha = 0.05)
    
    label1 = 'Pearson R: ' + "{:.4f}".format(rval)
    label3 = 'power: ' + "{:.4f}".format(power)
    label2 = 'pvalue: ' + "{:.4f}".format(pval) 
    
    ax=sns.jointplot(x,y, kind = 'kde',stat_func=None, height=10, color = 'darkgray')
    ax.ax_joint.cla()

    
    #clear axis

    ax.ax_joint.scatter(x,y, s=30, edgecolor = 'blue', c ='blue')
    ax.ax_joint.set_xlabel(xlabel)
    ax.ax_joint.set_ylabel(ylabel)
    ax.ax_joint.set_xlim(-1.5,1)
    ax.ax_joint.set_ylim(-1.5,1)
    
    
    for i in range(len(names)):
        an = ax.ax_joint.annotate(names[i], xy = (x[i],y[i]))
        an.draggable()
        
    ax.ax_joint.annotate(label1 + '\n' + label2 + '\n' + label3, xy=(0.57, 0.025), xycoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
    ax.ax_joint.plot([-2,2],[-2,2], ls = '--', color = 'black', zorder = -100)

    #add mean and std lines
    ax.ax_joint.axvline(x=np.mean(x), c='black', lw=1)
    ax.ax_joint.axvline(x=np.mean(x)+np.std(x), c='black', lw=1, ls = '--')
    ax.ax_joint.axvline(x=np.mean(x)-np.std(x), c='black', lw=1, ls = '--')
    
    ax.ax_joint.axhline(y=np.mean(y), c='black', lw=1)
    ax.ax_joint.axhline(y=np.mean(y)+np.std(x), c='black', lw=1, ls = '--')
    ax.ax_joint.axhline(y=np.mean(y)-np.std(x), c='black', lw=1, ls = '--')




    plt.subplots_adjust(bottom=0.21, right = 0.87)
    
    plt.subplots_adjust(top=0.94)
    ax.fig.suptitle(title)

    
#    plt.tight_layout()
    plt.savefig(workinghome + 'paperfigs_comparesites/' + savename+ '_v2.png')
plot_sitepparams_compare_nocolor(x= df_MEsiteresid['sitemean'], y = df_ANNsiteresid['sitemean'], names  = df_ANNsiteresid['sta'], xlabel = 'MEML site residual', ylabel = 'ANN site residual', title = r'ANN vs MEML ' + modelsite_name + ' site residuals', savename = 'ANN_MEML_' + modelsite)



