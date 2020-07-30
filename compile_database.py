#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:52:22 2019

@author: aklimase
"""

import glob
import numpy as np
import obspy
from obspy import read
import matplotlib.pyplot as plt
import os.path as path
import os
import waveforms as wf
import dread
import cPickle as pickle
import cdefs

db_evnum = []
db_sta = []
db_N = []
db_ml = []
db_mw = []
db_DA = []
db_DV = []
db_r = []
db_vs30 = []
db_kappa = []
db_elat = []
db_elon = []
db_edepth = []
db_stlat = []
db_stlon = []
db_stelv = []
db_sourcei = []
db_receiveri = []
db_vs30method = []
db_pga_snr = []
db_pgv_snr = []

data = pickle.load(open('/Users/aklimase/Documents/GMM_ML/v3anza2013_pgrid_5sta_res4.pckl', 'r'))

cut_dir = '/Users/aklimase/Documents/GMM_ML/corrected_filtered/'

event_dirs = glob.glob(cut_dir + 'Event_*')

#print 'Number of files: ', len(sac_files)

events = [os.path.basename(x) for x in event_dirs]

recordfiles = glob.glob(cut_dir + 'Event_*/*')

#find events in catalog 
catalog = '/Users/aklimase/Documents/GMM_ML/catalog/all_paths_M2.5_USGS_Catalog.txt'
pga = '/Users/aklimase/Documents/GMM_ML/catalog/all_pga.txt'
t = '/Users/aklimase/Documents/GMM_ML/catalog/tstar_site.txt'

pcat = np.genfromtxt(pga, comments = '#', delimiter = ' ', dtype = None, encoding = None)

cat = np.genfromtxt(catalog, comments = '#', delimiter = '|', dtype = None, encoding = None)
                    
tstarcat = np.genfromtxt(t, comments = '#', delimiter = '\t', dtype = None, encoding = None, names = True)


#loop through records
#for i in range(0, 1):#evnum
for i in range(len(cat)):

    time = obspy.core.utcdatetime.UTCDateTime(cat[i][1].split('.')[0])
    ev = str(time.year).zfill(4) + '_' + str(time.month).zfill(2) + '_' + str(time.day).zfill(2) + '_' + str(time.hour).zfill(2) + '_' + str(time.minute).zfill(2) + '_' + str(time.second).zfill(2)

    print ev
    event = ev
    
#    event = db_evnum[i]
    folder = event_dirs[i].split('/')[-2]
    recordpaths = glob.glob(cut_dir +'Event_' + event +'/*_*_HHN*.SAC')#full path for only specified channel
#    print recordpaths
    stns = [(x.split('/')[-1]).split('_')[1] for x in recordpaths]
    for j in range(len(stns)):
        #match for station info
#        print stns[j]
        for k in range(len(data.sta)):
            if data.sta[k] == stns[j]:
#                print stns[j]
                
                
                mag = cat[i][10]
                if mag < 3.5:
                    M = 0.884 + 0.754*mag#0.884 + 0.667*ml, 754
                    db_ml.append(mag)
                else:
                    M = mag
                    db_ml.append(0)
                db_mw.append(M)
                
                    #evnum
                db_evnum.append(i)
#                db_evnum.append(ev)

                #evlat
                db_elat.append(cat[i][2])
                #evlon
                db_elon.append(cat[i][3])
                #evdep
                db_edepth.append(cat[i][4])
                ####
                #match pga
                for l in range(len(pcat)):
                    if ('Event_' + ev) == pcat[l][0]:
                        if stns[j] == pcat[l][1]:
                            db_DA.append(pcat[l][2])
                            break
                
                db_sta.append(stns[j])
                db_stelv.append(data.stelv[k])
                db_stlat.append(data.stlat[k])
                db_stlon.append(data.stlon[k])
                db_vs30.append(data.vs30[k])
                db_vs30method.append(data.vs30_method[k])
                
                ind = np.where(tstarcat['site']==(stns[j]))
                db_kappa.append(tstarcat['tstars'][ind])
                
                dist =  dread.compute_rrup(cat[i][3], cat[i][2], cat[i][4], data.stlon[k], data.stlat[k], -1*data.stelv[k]) #in km
                db_r.append(dist)
                
                print ev, stns[j], data.stelv[k], data.stlat[k], dist, data.vs30[k], tstarcat['tstars'][ind], M, cat[i][2]

                break

  #make other things 0      
db_N = np.zeros(len(db_evnum))   
db_DV = np.zeros(len(db_evnum))   
db_sourcei = np.zeros(len(db_evnum))   
db_receiveri = np.zeros(len(db_evnum))   
db_pga_snr = np.zeros(len(db_evnum))   
db_pgv_snr = np.zeros(len(db_evnum))


#database_vs30 = cdefs.db(np.asarray(db_evnum), np.asarray(db_sta), np.asarray(db_N), np.asarray(db_ml), np.asarray(db_mw), np.asarray(db_DA), np.asarray(db_DV), np.asarray(db_r), np.asarray(db_vs30), np.asarray(db_elat), np.asarray(db_elon), np.asarray(db_edepth), np.asarray(db_stlat), np.asarray(db_stlon), np.asarray(db_stelv), np.asarray(db_sourcei), np.asarray(db_receiveri), vs30_method=None)
#pickle.dump(database_vs30 , open( "database_vs30.pckl", "wb" ) )

#database_test = cdefs.db(np.asarray(db_evnum), np.asarray(db_sta), np.asarray(db_N), np.asarray(db_ml), np.asarray(db_mw), np.asarray(db_DA), np.asarray(db_DV), np.asarray(db_r), np.asarray(db_kappa), np.asarray(db_elat), np.asarray(db_elon), np.asarray(db_edepth), np.asarray(db_stlat), np.asarray(db_stlon), np.asarray(db_stelv), np.asarray(db_sourcei), np.asarray(db_receiveri), vs30_method=None)
#pickle.dump(database_test , open( "database_test.pckl", "wb" ) )
#  
'''
Initiate the class by giving database information.
        Input:
            event:          Array with event number (event)
                            0
            sta:            Array/list with station name (sta)
                            get from the file name
            N:              Array with station number (N)
                            0 or number of stations recorded on
            ml:             Array with local mag (ml)
            mw:             Array with moment mag (mw)
                            we have ml if events smaller than 3.5 and mw if larger
            DA:             Array with PGA in m/s/s
                            from file written above
            DV:             Array with PGV in m/s
                            z0
            r:              Array with Rrup, source to site distance (r)
                            from Rrup (Rhyp catalog)
            vs30:           Array with vs30 (in m/s)
                            from file, one kappa and one vs30
            elat:           Array with event latitude
            elon:           Array with event longitude
            edepth:         Array with event depth (km), positive
                            read in from sac header or file
            stlat:          Array with station latitude
            stlon:          Array with station longitude
                            from vs30
            stelv:          Array with statin elevation (km), positive
                            check dataset from Valerie, anza website, iris
            source_i:       Array with source index for raytracing
                            0
            receiver_i:     Array with receiver index for raytracing
                            0
            vs30_method:    Array with string values of the Vs30 measurement method, useful if using several types
                            from file
            pga_snr:        Array with PGA signal to noise ratio
                            with scipy or by hand from the time series
            pgv_snr:        Array with PGV signal to noise ratio
'''





