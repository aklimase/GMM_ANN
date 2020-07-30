#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:46:53 2019

@author: aklimase
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from datetime import datetime
from datetime import timedelta
import dread
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.io.img_tiles as cimgt
import matplotlib.ticker as mtick
import cPickle as pickle
#from shapely.geometry.polygon import Polygon
import matplotlib.path as mpltPath
import dread as dr
#from obj import source_obj
from numpy import ones,r_,c_
from scipy.stats import binned_statistic_dd
from scipy.stats import pearsonr
import statsmodels
import statsmodels.stats.power
from scipy import stats
import seaborn as sns


plt.style.use("classic")
plt.rcParams['axes.axisbelow'] = True
mpl.rcParams.update({'font.size': 18})
mpl.rcParams['figure.subplot.left'] = 0.08
mpl.rcParams['figure.subplot.right'] = 0.95

#read in all event spectra files
top_dir = '/Users/aklimase/Desktop/USGS/project'

sfile = open(top_dir + '/source_params/Brune_fit_params_lessthan_res_3sigma_withfc.pckl','r')
sobj = pickle.load(sfile)
sfile.close()
#evid, m_cat, m_fit, lat, lon, depth, fc, log_stressdrop, log_stressdrop_sig, time

faultfile= top_dir + '/catalogs/faults/Holocene_LatestPleistocene_118.0w_115.0w_32.3n_34.4n.pckl'
fault_segments=dr.read_obj_list(faultfile)
        
        

    
dlon = 0.075*4
dlat = 0.05*4
ddep = 30.0
lon_edges = np.arange(-118, -115 + dlon, dlon)
lat_edges = np.arange(32.2, 34.4 + dlat, dlat)
dep_edges = np.arange(0,35,ddep)
bindims = [lon_edges, lat_edges, dep_edges]

grid = bindims
object1 = sobj
#object2 = shearer_obj
parameter1 = sobj.log_stressdrop
#parameter2 = np.log10(shearer_obj.constant_stressdrop)
statistic = 'median'


sample = c_[object1.lon, object1.lat, object1.depth]
x = parameter1
med = np.median(x)
bindims = grid
statistic_s,bin_edges,binnumber = binned_statistic_dd(sample, x, statistic=statistic, bins=bindims)
statistic_std1,bin_edges,binnumber = binned_statistic_dd(sample, x, statistic= 'std', bins=bindims)


stat = statistic
parameter1 = statistic_s
std1 = statistic_std1
binedges = bin_edges
#name1 = 'stressdrop'
name1 = 'log_stressdrop'



def grid_plot(stat, parameter1, std1, name1, binedges):
    #compare all cells
    flat1 = np.ndarray.flatten(parameter1)
    flaterr1 = np.ndarray.flatten(std1)
    #statistics
    x1 = []
    x1err = []
    for i in range(len(flat1)):
        if str(flat1[i]) != 'nan':
            x1.append(flat1[i])
            x1err.append(flaterr1[i]) 
    
 
    xedges, yedges = binedges[0:2]
    X,Y = np.meshgrid(xedges, yedges)
    
#    print X
#    print Y
#    print parameter1
    if name1 == 'counts':
        cmap = plt.get_cmap('Greens')
        norm1 = mpl.colors.Normalize(vmin= min([f for f in flat1 if str(f) != 'nan']), vmax = 50.)
        norm1_std = mpl.colors.Normalize(vmin= min([f for f in flaterr1 if str(f) != 'nan']), vmax = max([f for f in flaterr1 if str(f) != 'nan']))

#        norm1 = mpl.colors.Normalize(vmin= min([f for f in flat1 if str(f) != 'nan']), vmax = max([f for f in flat1 if str(f) != 'nan']))

    else:
        cmap = plt.get_cmap('coolwarm')
        norm1 = mpl.colors.Normalize(vmin= min([f for f in flat1 if str(f) != 'nan']), vmax = max([f for f in flat1 if str(f) != 'nan']))
#        norm1 = mpl.colors.Normalize(vmin= min([f for f in flat1 if str(f) != 'nan']), vmax = 100.)

        norm1_std = mpl.colors.Normalize(vmin= min([f for f in flaterr1 if str(f) != 'nan']), vmax = max([f for f in flaterr1 if str(f) != 'nan']))

    
#    norm1 = mpl.colors.Normalize(vmin= min([f for f in flat1 if str(f) != 'nan']), vmax = max([f for f in flat1 if str(f) != 'nan']))

    dep_edges = binedges[2]
    for k in range(len(dep_edges)-1):
        fig = plt.figure(figsize = (12,10))
        ax = plt.axes()
        fig.suptitle('event depth: ' + str(dep_edges[k]) + '-' + str(dep_edges[k+1]) + ' km')
    
        for segment_i in range(len(fault_segments)):
            fault=fault_segments[segment_i]
            fault_z=np.zeros(len(fault))
            plt.plot(fault[:,0],fault[:,1],fault_z,color='k')
    
        c1 = plt.pcolormesh(X,Y, parameter1[:,:,k].T, cmap = cmap, norm = norm1)
        cbar = fig.colorbar(c1, ax=ax)
        cbar.set_label(name1, fontsize = 22)
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.set_title(name1)
        ax.set_xlim(-118.01, -114.99)
        ax.set_ylim(32.29, 34.41)
    
        plt.show()

        plt.savefig('/Users/aklimase/Documents/GMM_ML/stressdrop/map_' + stat + '_' + name1 + '_' + '_depth_' + str(dep_edges[k]) + '-' + str(dep_edges[k+1])+ '.png')
        plt.close()
        
        
        
        fig = plt.figure(figsize = (12,10))
        ax = plt.axes()
        fig.suptitle('event depth: ' + str(dep_edges[k]) + '-' + str(dep_edges[k+1]) + ' km')
    
        for segment_i in range(len(fault_segments)):
            fault=fault_segments[segment_i]
            fault_z=np.zeros(len(fault))
            plt.plot(fault[:,0],fault[:,1],fault_z,color='k')
    
        c1 = plt.pcolormesh(X,Y, std1[:,:,k].T, cmap = cmap, norm = norm1_std)
        cbar = fig.colorbar(c1, ax=ax)
        cbar.set_label(name1, fontsize = 22)
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.set_title(name1)
        ax.set_xlim(-118.01, -114.99)
        ax.set_ylim(32.29, 34.41)
    
        plt.show()

        plt.savefig('/Users/aklimase/Documents/GMM_ML/stressdrop/map_std_' + stat + '_' + name1 + '_' + '_depth_' + str(dep_edges[k]) + '-' + str(dep_edges[k+1])+ '.png')
        plt.close()


    return X, Y, parameter1






X,Y,parameter1 = grid_plot(stat = statistic, parameter1 = statistic_s,std1 = statistic_std1,binedges = bin_edges,name1 = name1)

lonedges = binedges[0]
latedges = binedges[1]


lon0 = []
lon1 = []
lat0 = []
lat1 = []
avg = []
for i in range(len(lonedges)-1):
    for j in range(len(latedges)-1):
#        print lonedges[i], lonedges[i+1], latedges[j], latedges[j+1]
#        print parameter1[i,j]
        lon0.append(lonedges[i])
        lon1.append(lonedges[i+1])
        lat0.append(latedges[j])
        lat1.append(latedges[j+1])
        avg.append(round(parameter1[i][j][0], 4))

np.savetxt('/Users/aklimase/Documents/GMM_ML/stressdrop/test.txt', np.asarray([lon0, lon1, lat0, lat1, avg]).T, fmt=['%6.2f', '%6.2f', '%6.2f', '%6.2f', '%6.4f'], delimiter='\t', header='lon0  \t lon1  \t lat0  \t lat1  \t avg_log_stressdrop')
        


#dlon = 0.075*2
#dlat = 0.05*2
#ddep = 30.0
#lon_edges = np.arange(-118, -115+dlon, dlon)
#lat_edges = np.arange(32.2, 34.4+dlat, dlat)
#dep_edges = np.arange(0, 35, ddep)
#bindims = [lon_edges, lat_edges, dep_edges]

grid = bindims
object1 = sobj
parameter1 = sobj.log_stressdrop
statistic = 'count'

sample = c_[object1.lon, object1.lat, object1.depth]
x = parameter1
med = np.median(x)
bindims = grid
statistic_s,bin_edges,binnumber = binned_statistic_dd(sample, x, statistic=statistic, bins=bindims)
statistic_std1,bin_edges,binnumber = binned_statistic_dd(sample, x, statistic= 'std', bins=bindims)

stat = statistic
parameter1 = statistic_s
std1 = statistic_std1
binedges = bin_edges
name1 = 'counts'

X,Y,counts = grid_plot(stat = statistic, parameter1 = statistic_s,std1 = statistic_std1, binedges = bin_edges, name1 = 'counts')





lon0 = []
lon1 = []
lat0 = []
lat1 = []
avg = []
for i in range(len(lonedges)-1):
    for j in range(len(latedges)-1):
#        print lonedges[i], lonedges[i+1], latedges[j], latedges[j+1]
#        print parameter1[i,j]
        lon0.append(lonedges[i])
        lon1.append(lonedges[i+1])
        lat0.append(latedges[j])
        lat1.append(latedges[j+1])
        avg.append(counts[i][j][0])

np.savetxt('/Users/aklimase/Documents/GMM_ML/stressdrop/counts.txt', np.asarray([lon0, lon1, lat0, lat1, avg]).T, fmt='%6.2f %6.2f %6.2f %6.2f %6.0f', delimiter='\t', header='lon0  \t lon1  \t lat0  \t lat1  \t counts')


