#! /usr/bin/env python3

import sys
from Bruker_Pseudo2D import read_expno_ramp, integrate_peaks
from p2DFitRoutines import fit_routine

from bruker.api.topspin import Topspin
from bruker.data.nmr import *

top = Topspin()
dp = top.getDataProvider()

def main(path, eps = None,  firstexpno = None, lastexpno = None,  fitmodel_ = 'Exp' , Nexp_ = 1, norm_ = 'y', fiterror_ = 'n', sterror_ = 'n', DOSY_ = False, save_ = 'n', prettyexcel_ = 'n', dataname = None):
    """
    add path to data folder, either /opt/topspin/../1 or /opt/topspin/../ FirstExpno LastExpno. eps_ tupple for a noise region of the spectrum 
    """
    
    Bvalues, Spectra, datasetname = read_expno_ramp(path , firstexpno, lastexpno)

    if dataname == None:

        ppm, peakintegration = integrate_peaks(Spectra, datasetname, eps, firstexpno = firstexpno)
    
    else:
        
        ppm, peakintegration = integrate_peaks(Spectra, dataname, eps, firstexpno = firstexpno)
    
    results = fit_routine(Bvalues, Spectra, peakintegration, fitmodel = fitmodel_ , Nexp = Nexp_, eps =  eps, norm = norm_, fiterror = fiterror_, sterror = sterror_, DOSY = DOSY_, save = save_, datasetname = datasetname, prettyexcel_ = prettyexcel_)

    return results

if __name__ == '__main__':
    main(*sys.argv)
