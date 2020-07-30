#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:19:58 2019

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

cut_dir = '/Users/aklimase/Desktop/USGS/project/Andrews_inversion/all_paths/cutdata_s/'
sac_files = glob.glob(cut_dir + 'Event_*/*.SAC')

##read in uncorrected
##filter again 0 - 0.1 Hz
#and corrected
#use np.diff or obspy to differentiate for acceleration
#find max abs value of a

def find_pga(tr):#velocity
# Acceleration
    data = np.gradient(tr, 0.01)
    if abs(max(data)) >= abs(min(data)):
        m_acc = abs(max(data))
    else:
        m_acc = abs(min(data))
    return m_acc


# Set parameters for resp file:
location = '*'
tsunit = 'VEL'
channel = 'HH*'
nyquistf = 50



# Pre-filter for instrument response correction:
#   The last number should be the nyquist frequency; second to last reduces
#   the filter towards the nyquist; first two are for the "water level"
# for BH Nyquist frequency is 20, for HH Nyquist frequency is 50
prefilt = (0.0,0.1,0.7*nyquistf,nyquistf)

event_dirs = glob.glob(cut_dir + 'Event_*')

print 'Number of files: ', len(sac_files)


events = [os.path.basename(x) for x in event_dirs]
corr_dir = '/Users/aklimase/Documents/GMM_ML/corrected_filtered'

#make a directory for each event
for i in range(len(events)):
    if not path.exists(corr_dir+ '/' + events[i]):
        os.makedirs(corr_dir + '/' + events[i])
        
        
        

eventpaths = glob.glob(cut_dir + 'Event_*/*')


#make response files in response directory for each combo of networks and stations
response_path = '/Users/aklimase/Desktop/USGS/project/Andrews_inversion/all_paths/respfiles'

#        
        
hard_start = 0
print(hard_start)

pga = []
event_name = []
s = []


for i in range(0, len(event_dirs)):
#for i in range(0, 10):
    event = path.basename(event_dirs[i])
    folder = event_dirs[i].split('/')[-2]
    print event
    print i

    recordpaths = glob.glob(cut_dir + '/' + event +'/*_*_HHN*.SAC')#full path for only specified channel
#    print recordpaths
    stns = [(x.split('/')[-1]).split('_')[1] for x in recordpaths]
    for j in range(len(stns)):
        recordpath_E = glob.glob(cut_dir + event +'/*_' + stns[j] + '_HHE*.SAC')
        recordpath_N = glob.glob(cut_dir + event +'/*_' + stns[j] + '_HHN*.SAC')
        if(len(recordpath_E) == 1 and len(recordpath_N) == 1):
            #North component
            base_N = path.basename(recordpath_E[0])
            base_E = path.basename(recordpath_N[0])
    
            network = base_N.split('_')[0]
            station = base_N.split('_')[1]
            full_channel_N = base_N.split('_')[2]
            
            stream = read(recordpath_N[0])
            tr_N = stream[0]
            
            full_channel_E = base_E.split('_')[2]
            
            stream = read(recordpath_E[0])
            tr_E = stream[0]
            
            #find response file
            respfile = response_path + '/' + network + '.' + station + '.' + 'HH*' + '.resp'
            
            if(tr_E.stats.npts > 0) and (tr_N.stats.npts > 0):
                tr_E.detrend(type = 'demean')#removes mean of data
                tr_E.detrend(type = 'simple')#rtrend linear from first and last samples
                #rewrite to a sac file
                tr_E.write('temp.sac', format = 'sac')
                sacfile = 'temp.sac'
                icorr_sacfile = corr_dir + '/' + event + '/' + base_E
                #uncorrected_sac_file,resp_file,corrected_sac_file,water_level_bds,resp_unit
                wf.remove_response(sacfile,respfile,icorr_sacfile,prefilt,tsunit)#prefilt values for HH
        
                stream = read(icorr_sacfile)
                tr_E = stream[0]
                pga_E = find_pga(tr_E)
                
                tr_N.detrend(type = 'demean')#removes mean of data
                tr_N.detrend(type = 'simple')#rtrend linear from first and last samples
                #rewrite to a sac file
                tr_N.write('temp.sac', format = 'sac')
                sacfile = 'temp.sac'
                icorr_sacfile = corr_dir + '/' + event + '/' + base_N
                #uncorrected_sac_file,resp_file,corrected_sac_file,water_level_bds,resp_unit
                wf.remove_response(sacfile,respfile,icorr_sacfile,prefilt,tsunit)#prefilt values for HH
        
                stream = read(icorr_sacfile)
                tr_N = stream[0]
                pga_N = find_pga(tr_N)
                
                ##################3
                #plot here
                print base_N
#                fig = tr_E.plot()
                # Two subplots, the axes array is 1-d
                fig, ax =  plt.subplots(4,1, figsize =(10,10))
                ax[0].plot(tr_N.times(),tr_N.data)
                ax[1].plot(tr_N.times(),np.gradient(tr_N.data, 0.01))
                ax[2].plot(tr_E.times(),tr_E.data)
                ax[3].plot(tr_E.times(),np.gradient(tr_E.data, 0.01))
#                axarr[2] = np.gradient(tr, 0.01)
                fig.savefig('/Users/aklimase/Documents/GMM_ML/data/' + base_N + '.png')    
                plt.close()
                
            avg_pga = np.sqrt(pga_N*pga_E)
            pga.append(avg_pga)
            event_name.append(event)
            s.append(station)
            
#                print event
#                print stns[j]
#                print 
                
                
                
#
#X = np.c_[event_name, s,pga]
#np.savetxt(corr_dir + '/all_pga.txt', X, fmt='%s', delimiter=' ', newline='\n', header='event' + '\t'+ 'stn' + '\t' + 'pga(m/s2)')
