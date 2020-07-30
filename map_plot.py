#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:46:58 2019

@author: aklimase
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import dread as dr
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cPickle as pickle
#from obj import fc_obj
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
mpl.rcParams.update({'font.size': 14})


#plot events and stuff
def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')
    




#read in data base
#loop and find all events that meet station, pga, and distance criteria
#find the event depth
db = pickle.load(open('/Users/aklimase/Documents/GMM_ML/database_kappa.pckl', 'r'))
elon = []
elat = []
edep = []
mw = []
pga = []

indsta = np.where(db.sta == 'SWS')[0][0]
stlon = db.stlon[indsta]
stlat = db.stlat[indsta]

for i in range(len(db.sta)):
    log10pga = np.log10(db.pga[i]/9.81) #log 10 pga in g
    if (db.sta[i] == 'SWS' and log10pga < -5.0 and db.r[i] >= 10**(1.2) and db.r[i] <= 10**(2.0)):
        elon.append(db.elon[i])
        elat.append(db.elat[i])
        edep.append(db.edepth[i])
        mw.append(db.mw[i])
        pga.append(log10pga)
        
#color events by depth or pga
cmap = mpl.cm.get_cmap('magma_r')
normalize = mpl.colors.Normalize(vmin=min(edep), vmax=max(edep))
colors = [cmap(normalize(value)) for value in edep]
s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
s_m.set_array([])




#plt.style.use("classic")
plt.rcParams['axes.axisbelow'] = True
mpl.rcParams.update({'font.size': 18})

top_dir = '/Users/aklimase/Documents/GMM_ML/catalog'

faultfile= '/Users/aklimase/Desktop/USGS/project/catalogs/faults/Holocene_LatestPleistocene_118.0w_115.0w_32.3n_34.4n.pckl'
fault_segments=dr.read_obj_list(faultfile)

stamen_terrain = cimgt.Stamen('terrain-background')

fig = plt.figure(figsize = (10,7))
ax = plt.axes(projection=stamen_terrain.crs)
ax.set_extent([-118.01, -114.99, 32.29, 34.41])

#elevation
ax.add_image(stamen_terrain, 10, alpha = 0.7)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='both', length = 5, width = 1)

#parallels and meridians
xticks = [-118, -117, -116, -115]
yticks = [32.5, 33.0, 33.5, 34.0]
g1 = ax.gridlines(xlocs=xticks, ylocs=yticks, draw_labels = True, zorder = -100)
g1.xformatter = LONGITUDE_FORMATTER
g1.yformatter = LATITUDE_FORMATTER

scale_bar(ax, 100)

#add faults
for segment_i in range(len(fault_segments)):
    fault=fault_segments[segment_i]
    fault_z=np.zeros(len(fault))
    ax.plot(fault[:,0],fault[:,1],fault_z, lw = 0.5, alpha = 0.5, color='k', transform=ccrs.Geodetic())

fault=fault_segments[0]
fault_z=np.zeros(len(fault))
ax.plot(fault[:,0],fault_z, lw = 0.5, alpha = 0.5, color='k', transform=ccrs.Geodetic(), label = 'faults')

#compass
ax.text(0.28, 0.03,u'\u25B2 \nN ', ha='center', fontsize=12, family='Arial', rotation = 0, transform=ax.transAxes)
#sites
plt.scatter(elon, elat, marker='.', s=40, color = colors, alpha=1.0, transform=ccrs.Geodetic(), label = 'events')

plt.scatter(stlon, stlat, marker='^', s=40, color = 'black', alpha=1.0, transform=ccrs.Geodetic(), label = db.sta[indsta])
plt.legend(scatterpoints = 1, fontsize = 16)


fig.subplots_adjust(right=0.65)
cbar_ax = fig.add_axes([0.85, 0.18, 0.1, 0.63])
#cbar = plt.colorbar(s_m, cax=cbar_ax)
#cbar.set_label('depth (km)')#"$azimuth$ (\u00b0)"
#cbar.ax.tick_params()
plt.ylabel(r'event depth (km)', fontsize = 20)
plt.xlabel(r'counts', fontsize = 20)
plt.yticks(np.arange(0,16,2))

N, bins, patches = cbar_ax.hist(edep, bins=np.arange(0,16,1), orientation='horizontal')
for bin, patch in zip(bins, patches):
    color = plt.cm.magma_r(normalize(bin))
    patch.set_facecolor(color)

plt.show()

plt.savefig(top_dir + '/SWS_pga_-5.0_depthcolor.png')







#make a histogram of depths 
fig = plt.figure(figsize = (6,4))

#norm=plt.Normalize(0,16)
#colors = plt.cm.magma_r(norm(edep))


N, bins, patches = plt.hist(edep, bins=np.arange(0,16,1)) #, orientation=u'horizontal')
plt.xlabel(r'event depth (km)', fontsize = 20)
plt.ylabel(r'counts', fontsize = 20)

plt.subplots_adjust(bottom = 0.2)
#m = (int(max(N))/1000)*1000 + 1000
#plt.yticks(np.arange(0,m, 500))
#plt.tick_params(axis='both', which='major', labelsize=20)
#plt.tick_params(axis='y', which='major', labelsize=20, rotation = 45)
#plt.tick_params(axis='both', which='minor', labelsize=20)  
for bin, patch in zip(bins, patches):
    color = plt.cm.magma_r(normalize(bin))
    patch.set_facecolor(color)

plt.savefig(top_dir + '/SWS_pga_-5.0_depthhisto.png')
