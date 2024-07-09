import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import nmrglue as ng
import matplotlib.pyplot as plt


from bruker.api.topspin import Topspin
from bruker.data.nmr import *
from Dosyfuncs import conv_grad_b

top = Topspin()

def _ppm_scale(i_parameteres): # receives dic with experimental parameters
    
    f1=i_parameteres["procs"]["ABSF1"]
    f2=i_parameteres["procs"]["ABSF2"]
    fts=i_parameteres["procs"]["FTSIZE"]
    ppm_values= np.around((np.flip(np.linspace(f2,f1,fts))),decimals=2)

    return ppm_values


def _get_intrng(file_):

    intfile = pd.read_csv(file_, names=[0,1], delim_whitespace=True , header  = 1, usecols = [0,1])
    intfile = [tuple(intfile.loc[i]) for i in range(len(intfile))]

    return intfile

def _new_integrate(Xscale, data, dictionary, intfile, nregion = None, *args):

    uc = ng.fileio.fileiobase.uc_from_freqscale(Xscale, dictionary['acqus']['BF1'])

    if nregion == None:
    
        if len(intfile) == 1:
            
            integration = np.array([ng.analysis.integration.integrate(data[i],uc,intfile[0], unit = 'ppm') for i in range(len(data))])

        else:
            
            integration = np.array([ng.analysis.integration.integrate(data[i],uc,intfile, unit = 'ppm') for i in range(len(data))])
    
    else:
    
        if len(intfile) == 1:
            
            integration = np.array([ng.analysis.integration.integrate(data[i],uc,intfile[0],unit = 'ppm',noise_limits = nregion) for i in range(len(data))])

        else:
           
            integration = np.array([ng.analysis.integration.integrate(data[i],uc,intfile,unit = 'ppm',noise_limits = nregion) for i in range(len(data))])

    return integration


def integrate_peaks(data_array, datasetname, eps = None, firstexpno = None):

    
    regions = _get_intrng(str(top.getInstallationDirectory()) + '/exp/stan/nmr/lists/intrng/' + datasetname)

    if firstexpno == None:

        ppm_array = _ppm_scale(data_array[0])
        peakintegration = _new_integrate(ppm_array, data_array[1],data_array[0], regions, eps) 

    else:

        ppm_array = _ppm_scale(data_array[0][0])
        peakintegration = np.array([_new_integrate(ppm_array, data_array[i][1],data_array[i][0], regions, eps) for i in range (len(data_array))])

    return ppm_array, peakintegration


def int_limits(IntegReg, ppmScale):

    mat=np.array([(IntegReg[i][0], IntegReg[i][1]) for i in range(0,len(IntegReg))])
    mat=np.around(mat,decimals=2)

# extract the peaks and ppm scale
    
    IntReg = [(np.where( ppmScale == mat[f][1]) , np.where (ppmScale == mat[f][0])) for f in range (0, len(IntegReg))]

    return IntReg

def return_results(ppm_xdata, peaks_ydata, integ_Limits):

    results=[]

    for n in range (0,len(peaks_ydata)):
        espectro1d=peaks_ydata[n][:]
        peak=[]
        peak_scale=[]
        
        for g in range(0, len(integ_Limits)):
            t = espectro1d[(integ_Limits[g][1])[0][0]:(integ_Limits[g][0])[0][0]+1]
            y = ppm_xdata[(integ_Limits[g][1])[0][0]:(integ_Limits[g][0])[0][0]+1]    
            peak.append(t)
            peak_scale.append(y)    

        tup=[]

        for k in range (0,len(peak)):
            plt.plot(peak_scale[k], peak[k].cumsum() / 100. + peak[k].max(), 'g-')
            plt.plot(peak_scale[k], [0] * len(peak_scale[k]), 'r-')
         #  plt.show()
            t = (peak_scale[k], peak_scale[k][-1], peak[k].sum())
            tup.append(t)        
        
        results.append(tup)
    
    r= [results[g][d][2] for d in range(0,len(integ_Limits)) for g in range(0,len(peaks_ydata))]
    r=[r[i:i + len(peaks_ydata)] for i in range(0, len(r), len(peaks_ydata))]
    
    return r

def read_ts_peak_list(peaklist, peak_amplitude = None):
    
    tree = ET.parse(peaklist)
    root = tree.getroot()
    TopSpin_peaks = np.flip(np.sort([float(value.attrib['F1']) for value in root.iter('Peak1D')]))

    if peak_amplitude != None:

        TopSpin_peaks_intensity = np.flip(np.sort([float(value.attrib['intensity']) for value in root.iter('Peak1D')]))

    else:

        TopSpin_peaks_intensity = None
        
    return TopSpin_peaks, TopSpin_peaks_intensity



def read_expno_ramp(path, firstexpno = None, lastexpno = None, DOSY = 'False'):

    if firstexpno is None:

        try :

            gradlist = np.genfromtxt(path + '/difflist') 

        except:

            gradlist = np.genfromtxt(path + '/vdlist')

        dataarray = ng.bruker.read_pdata(path  + '/pdata/1')

        if DOSY == 'True':     

            gradlist = conv_grad_b(gradlist, dataarray)

        name = path.split('/')[-3]  

    else:

        try:
        
            gradlist = np.array([np.genfromtxt(path + str(expno) + '/difflist') for expno in range(firstexpno, lastexpno)])

        except:
        
            gradlist = np.array([np.genfromtxt(path + str(expno) + '/vdlist') for expno in range(firstexpno, lastexpno)])
        
        dataarray=[ng.bruker.read_pdata(path + str(expno) +'/pdata/1') for expno in range(firstexpno, lastexpno)]

        if DOSY == 'True':
            
            gradlist = np.array([conv_grad_b(grad, dataarray[index]) for index, grad in enumerate(gradlist)])

        name = path.split('/')[-2]
    
        
    return gradlist, dataarray, name
