#! /usr/bin/python3
# -*- coding: utf-8 -*-

########################################################################################
### here some descrition 

"""
Created on Thu Dec 19 16:48:33 2024

@author: julien
"""

########################################################################################

import sys
sys.settrace
import pickle
import os
import time
import datetime as dt
import pandas as pd
import numpy as np
import random
import statistics
import scipy
from scipy.integrate import simpson
from scipy.stats import shapiro
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from dash import Dash, html, dcc, callback, Output, Input
from tqdm import tqdm # progressbar
from multiprocessing.pool import ThreadPool
import threading
from juliacall import Main as jl, convert as jlconvert
from lmfit.models import VoigtModel
from sklearn.decomposition import PCA, FastICA
from spectres import spectral_resampling_numba
from lmfit.models import PseudoVoigtModel

# chande dir 
os.chdir("/home/spegeochert/Data/LIBS/LIBS_analysis_code/Analysis_Pipeline_article/LIBS_analysis_testing")

# import external julia background removal
jl.include("background_removal.jl")
jl.include("IFF.jl")

# global settings 
Overide_processing = False  #Set to true to reduce memory usage
Load_all_wl = True          #Set to false to reduce memory usage

# plotly config
plotly_config = {
    'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 1080,
    'width': 1920,
    'scale' : 1  # Multiply title/legend/axis/canvas sizes by this factor
    }
}


### Python version of the AirPLS algorithm by Zhang et al., 2010, slower than the Julia version
def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    output
        the fitted background vector
    '''
    # comment 
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    output
        the fitted background vector
    '''
    # comment        
    m=len(x)
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARNING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

### Peak finding functions
def find_peak_indices(peak_positions, interpolated_list):
    """
    Find the indices of peak positions in an interpolated list.
    Args:
        peak_positions (list or np.ndarray): List of peak positions by wavelength.
        interpolated_list (list or np.ndarray): Interpolated wavelength list.
    Returns:
        list: Indices of the closest values in the interpolated list.
    """
    # comment               
    interpolated_list = np.array(interpolated_list)
    peak_positions = np.array(peak_positions)
    # Find the insertion points for each peak
    indices = np.searchsorted(interpolated_list, peak_positions)
    # Refine to get the closest index
    refined_indices = []
    for peak, idx in zip(peak_positions, indices):
        # Handle edge cases
        if idx == 0:
            refined_indices.append(0)
        elif idx == len(interpolated_list):
            refined_indices.append(len(interpolated_list) - 1)
        else:
            # Compare the closest neighbors
            left_diff = abs(interpolated_list[idx - 1] - peak)
            right_diff = abs(interpolated_list[idx] - peak)
            refined_indices.append(idx - 1 if left_diff < right_diff else idx)
    return refined_indices

# fit single peak
def fit_single_peak(x, y, peak_positions, window, thresholdmultiplier, x_base, y_base, verbose = False, plot = False):
    results = []
    for pos in peak_positions:
        if verbose:
            print ("\n")
            print (f"* Peak at {pos}")
        # set 
        mask = (x_base > pos - window) & (x_base < pos + window)
        x_base2 = x_base[mask]
        y_base2 = y_base[mask]
        # Select region around the peak
        mask = (x > pos - window) & (x < pos + window)
        x_fit_ref = x[mask]
        #print (f"Range is {x_fit_ref[0]}-{x_fit_ref[-1]} nm")
        y_fit_ref = y[mask]
        newwindow = window
        # loop  
        nrep = 0      
        while True:
            # Limit the data to the current mask
            x_masked = x[mask]
            y_masked = y[mask]
            # Find the closest maximum (the real peak) to the peak detected with wavelet transform
            closest_mask = (x_masked > pos - 0.2) & (x_masked < pos + 0.2)
            closest_values = y_masked[closest_mask]
            if nrep == 0:
                if (pos > 421) & (pos < 422):
                    newwindow = 0.5
                    mask = (x > pos - newwindow) & (x < pos + newwindow)
                if (pos > 493) & (pos < 494):
                    newwindow = 0.5
                    mask = (x > pos - newwindow) & (x < pos + newwindow)
                elif max(closest_values) < 1000:
                     newwindow = 1
                     if max(closest_values) < 300:
                         newwindow = 0.5
                     mask = (x > pos - newwindow) & (x < pos + newwindow)
            #print(max(closest_values))
            # Compute the threshold based on the current maximum within the mask
            threshold = max(y_masked[~closest_mask]) * thresholdmultiplier
            if verbose:
                print(f"Window size: {newwindow}, Peak value: {max(closest_values)}, Threshold: {threshold}")
            # Check the condition
            if max(closest_values) >= threshold : 
                if verbose: print("breaking by threshold")
                break
            if newwindow < 0.4: 
                if verbose: print("breaking by window")
                break
            if nrep > 20:
                if verbose: print("breaking by nrep")            
                break
            # Shrink the window
            newwindow -= 0.05
            mask = (x > pos - newwindow) & (x < pos + newwindow)
            nrep +=1
        # get x_fit and y_fit
        x_fit = x[mask]
        #print (f"Fit range is {x_fit[0]}-{x_fit[-1]} nm")
        y_fit = y[mask] 
        # Define the Voigt model
        model = PseudoVoigtModel(prefix='p1_')
        params = model.make_params()
        # Initial parameter guesses
        params['p1_center'].set(value=pos, min=pos-0.1, max=pos+0.1)
        params['p1_sigma'].set(value=1, min=0.1, max=10)
        params['p1_amplitude'].set(value=max(y_fit), min=0)
        #params['p1_gamma'].set(value=1, min=0.1, max=10)
        # Perform the fit
        #baseline = jl.airPLS(np.array(y_fit), 100, 1, 100)
        fit_result = model.fit(y_fit, params, x=x_fit)
        # Extract fit, max and FWHM
        area = simpson(y=fit_result.best_fit, x=x_fit)
        max_value = max(fit_result.best_fit)
        fwhm = fit_result.params['p1_fwhm'].value
        results.append({'center': pos, 'max_value': max_value, 'area': area, 'fwhm': fwhm})
        # plot 
        if plot:
            # Plot the fit
            plt.plot(x_fit_ref, y_fit_ref, 'b-', label='Akima1D')
            plt.plot(x_base2, y_base2, 'o', label='Base Data')
            #plt.plot(x_fit, baseline, 'r-', label='Baseline')
            plt.plot(x_fit, fit_result.best_fit, 'r--', label='Fit')
            plt.legend()
            plt.title(f'Peak at {pos}')
            plt.show()
    return results

# fit double peaks
def fit_double_peaks(x, y, peak_positions, window, x_base, y_base, verbose = False, plot = False):
    peak_positions = [peak_positions[i:i+2] for i in range(0, len(peak_positions), 2)]
    results = []
    for pos in peak_positions:
        if verbose:
            print ("\n")
            print (f"* Peaks at {pos[0]}, {pos[1]}")
        # Select region around the peak
        mask = (x > pos[0] - window) & (x < pos[1] + window)
        #print([pos[0], pos[1]])
        x_fit_ref = x[mask]
        y_fit_ref = y[mask]
        # comment
        mask = (x_base > pos[0] - window) & (x_base < pos[1] + window)
        x_base2 = x_base[mask]
        y_base2 = y_base[mask]
        # comment
        closest_mask = (x_fit_ref > pos[0] - 0.2) & (x_fit_ref < pos[0] + 0.2) | (x_fit_ref > pos[1] - 0.2) & (x_fit_ref < pos[1] + 0.2)
        closest_values = y_fit_ref[closest_mask]
        if verbose: print(max(closest_values))      # debugging print?
        # comment
        if max(closest_values) < 500:
            if verbose: print("Under 500")
            mask = (x > pos[0] - 0.5) & (x < pos[1] + 0.5)
            x_fit = x[mask]
            y_fit = y[mask]
        elif max(closest_values) < 3000:
            if verbose: print("Under 3000")
            mask = (x > pos[0] - 1) & (x < pos[1] + 1)
            x_fit = x[mask]
            y_fit = y[mask]
        else:
            if verbose: print ("Over 1000")
            x_fit = x_fit_ref
            y_fit = y_fit_ref
        # Define the Voigt model
        model = PseudoVoigtModel(prefix='p1_') + PseudoVoigtModel(prefix='p2_')
        params = model.make_params()
        # Initial parameter guesses
        params['p1_center'].set(value=pos[0], min=pos[0]-0.1, max=pos[0]+0.1)
        params['p1_sigma'].set(value=1, min=0.1, max=100)
        params['p1_amplitude'].set(value=max(y_fit), min=0)
        params['p2_center'].set(value=pos[1], min=pos[1]-0.1, max=pos[1]+0.1)
        params['p2_sigma'].set(value=1, min=0.1, max=100)
        params['p2_amplitude'].set(value=max(y_fit), min=0)
        #params['p1_gamma'].set(value=1, min=0.1, max=10)
        # Perform the fit
        fit_result = model.fit(y_fit, params, x=x_fit)
        # Extract area and FWHM
        peak1 = PseudoVoigtModel(prefix='p1_').eval(params=fit_result.params, x=x_fit)
        peak2 = PseudoVoigtModel(prefix='p2_').eval(params=fit_result.params, x=x_fit)
        area1= simpson(y= peak1, x=x_fit)
        max1 = max(peak1)
        fwhm1 = fit_result.params['p1_fwhm'].value
        area2= simpson(y= peak2, x=x_fit)
        max2 = max(peak2)
        fwhm2 = fit_result.params['p2_fwhm'].value
        results.append({'center': pos[0], 'max_value':max1, 'area': area1, 'fwhm': fwhm1})
        results.append({'center': pos[1], 'max_value':max2, 'area': area2, 'fwhm': fwhm2})
        # plot 
        if plot:
            # Plot the fit
            plt.plot(x_fit_ref, y_fit_ref, 'b-', label='Akima1D')
            plt.plot(x_base2, y_base2, 'o', label='Base Data')
            plt.plot(x_fit, fit_result.best_fit, 'r--', label='Fit')
            plt.legend()
            plt.title(f'Peak at {pos}')
            plt.show()
    return results

#=======================================================================================

### Class for building and ploting pca objects from LIBS_dataset objects
class LIBS_PCA:
    def __init__(self):
        self.n_comp = None
        self.n_samples = None
        self.n_variables = None
    # fit method
    def fit(self, LIBS_dataset, n_comp, metadata = ["test_number", "shot_number","index"], processing = "base", waveRange = 0, Index = [], subselection = 0, excluded_shots = range(1,8), ICA = False):
        if not subselection:
            wlmin = len(metadata)
            wlmax = -1
        if isinstance(subselection, list):
            wlmin = len(metadata)+subselection[0]
            wlmax = subselection[1]
        #Calls the get_dataframe function to get data usable for IFF       
        data = LIBS_dataset.buildDataframe(metadata = metadata, processing = processing, waveRange = waveRange, Index = Index, interpolate=True, excluded_shots = excluded_shots)
        self.metadata = data.iloc[:,:len(metadata)]
        self.variables = data.iloc[:,len(metadata)+wlmin:wlmax]
        self.n_comp = n_comp
        self.n_variables = self.variables.shape[1]
        self.n_samples = len(self.variables)
        if ICA:
            self.pca = FastICA(n_components=self.n_comp)
        else:
            self.pca = PCA(n_components=self.n_comp)
        self.pca.fit(self.variables)
    # plot eboulis
    def plot_eboulis(self):
        plt.bar(
            range(1,len(self.pca.explained_variance_)+1),
            self.pca.explained_variance_
            )
        plt.plot(
            range(1,len(self.pca.explained_variance_ )+1),
            np.cumsum(self.pca.explained_variance_),
            c='red',
            label='Cumulative Explained Variance')
        plt.legend(loc='upper left')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance (eignenvalues)')
        plt.title('Scree plot')
        self.eboulis_plot = plt
        plt.show()
    # project        
    def project(self, LIBS_dataset = None):
        if LIBS_dataset == None:
            self.scores = pd.DataFrame(self.pca.transform(self.variables))
        else :
           print("Function not yet implemented :)")
    # plot results     
    def plot_results(self, pc_x = 1, pc_y = 2, group="Formation", hover_data=['test_number','shot_number', 'Nom']):
            datatoplot = pd.concat([self.metadata, self.scores], axis=1)
            self.results_plot = px.scatter(datatoplot, x=pc_x-1, y=pc_y-1, color=group, hover_data=hover_data, template="plotly_white")
            xtext = f"PC {pc_x} ({np.round(self.pca.explained_variance_ratio_[pc_x-1]*100,2)}%)"
            ytext = f"PC {pc_y} ({np.round(self.pca.explained_variance_ratio_[pc_y-1]*100,2)}%)"
            self.results_plot.update_layout(
            xaxis=dict(
                title=dict(
                    text=xtext
                )
            ),
            yaxis=dict(
                title=dict(
                text=ytext
            )
            ))
            self.results_plot.show(config=plotly_config)
    # plot loadings            
    def plot_loadings(self):      
        loadings = pd.DataFrame(self.pca.components_.transpose())
        loadings.columns = list(self.pca.explained_variance_ratio_)
        loadings = pd.concat([pd.DataFrame(self.variables.columns, columns=["wavelength"]), loadings], axis=1)
        loadings_long = pd.melt(loadings, id_vars=['wavelength'], value_vars=list(self.pca.explained_variance_ratio_))
        fig = px.line(loadings_long, x="wavelength", y="value",color="variable", template="plotly_white")
        self.loading_plot = fig
        fig.show(config=plotly_config)
        
#=======================================================================================

#Class to build LIBS spectra objects, with associated processing methods and metadata managment.

class LIBS_spectra:
    def __init__(self):
        self.values = None # primitive attribute, actually raw-values
        self.wavelength = None
        self.test_number = None
        self.datetime = None
        self.shot_number = None
        self.processing = {}
        self.metadata = {}
        self.areas = {}
    # print options
    def __str__(self):
        return f"{self.test_number}.{self.shot_number} - {self.datetime} - {list(self.processing.keys())}"
    def __repr__(self):
        return f"{self.test_number}.{self.shot_number} - {self.datetime} - {list(self.processing.keys())}"
    # simple plot the spectra (obolete?)
    def Plot_spectra(self, processings): 
        # colors
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(processings)))
        for i, y in enumerate(processings): 
            plt.plot(np.array(self.wavelength), np.array(self.processing[y]), label=y, color=colors[i])
        plt.show()
    # load and set data
    def loadData(self, file, wavelength_out = None):
        # load 
        data_df = pd.read_csv(file, engine="pyarrow")
        #data_df = pd.read_csv(file, engine="c")
        data_ref = os.path.basename(file).split('_')
        # process fields 
        self.test_number = int(data_ref[0])
        self.shot_number = int(data_ref[-1].split('.')[0])
        self.datetime = dt.datetime(int(data_ref[1]), int(data_ref[2]), int(data_ref[3]), 
                        int(data_ref[4][0:2]),int(data_ref[4][2:4]), int(data_ref[4][4:6])) # ? hour
        # wavelength staff 
        if Load_all_wl: 
            self.wavelength = data_df['wavelength']
            self.values = data_df['intensity']
        else: 
            self.values = list(self._Interpolate(data_df['wavelength'], data_df['intensity'], wavelength_out))
        self.processing["base"] = self.values # pointer to save memory
        # add metadata 
        self.metadata["test_number"] = self.test_number
        self.metadata["shot_number"] = self.shot_number
        self.metadata["datetime"] = self.datetime
        if Overide_processing == True:
            self.metadata["processing"] = "base"
        #Find the cutoffs at loading
        self.find_cutoff()
    # perform simpson integration
    def Simpson(self, key, processing="base", Min=False, Max=False): 
        # ternary operator for specific values
        Min_index = 0 if Min == False else np.argmin(np.abs(np.array(self.wavelength) - Min))
        Max_index = len(self.wavelength)-1 if Max == False else np.argmin(np.abs(np.array(self.wavelength) - Max))
        # call simpron integration
        self.areas[key] = simpson(y = list(self.processing[processing])[Min_index : Max_index], x = list(self.wavelength)[Min_index : Max_index])
    # background removal
    def background_removal(self, processing="base", lambda_=100, differences=1, Julia = True):
        # take the data according to processing name 
        data = self.processing[processing]
        # Seperate the spectra according to the three detectors in the spectrometer. 
        detector1 = data[np.abs(self.wavelength-self.detector1[0]).argmin():np.abs(self.wavelength-self.detector1[1]).argmin()]
        detector2 = data[np.abs(self.wavelength-self.detector2[0]).argmin():np.abs(self.wavelength-self.detector2[1]).argmin()]
        detector3 = data[np.abs(self.wavelength-self.detector3[0]).argmin():]
        # Apply airPLS to each segment
        if (Julia): 
            res_detector1 = jl.airPLS(np.array(detector1), lambda_, differences)
            res_detector2 = jl.airPLS(np.array(detector2), lambda_, differences)
            res_detector3 = jl.airPLS(np.array(detector3), lambda_, differences)  
        else: 
            res_detector1 = airPLS(detector1, lambda_, differences)
            res_detector2 = airPLS(detector2, lambda_, differences)
            res_detector3 = airPLS(detector3, lambda_, differences)
        # Remove the background and store the info
        if Overide_processing == True:
            background = np.concatenate((res_detector1, res_detector2, res_detector3))
            self.processing["base"] = data - background
            self.metadata["processing"] = self.metadata["processing"]+'/AirPLS'
        else:
            # Merge back the results (store background as an atribute in the class) 
            self.background = np.concatenate((res_detector1, res_detector2, res_detector3))
            self.processing[processing + '/AirPLS'] = data - self.background 
    # comment
    def snv(self, processing="base", split = False, wlrange = []):
        if wlrange:
            data = self.processing[processing][wlrange[0]:wlrange[-1]]
        else:
            data = self.processing[processing]        
        if split == True:
            # Cut the spectra for each detector
            detector1 = data[np.abs(self.wavelength-self.detector1[0]).argmin():np.abs(self.wavelength-self.detector1[1]).argmin()]
            detector2 = data[np.abs(self.wavelength-self.detector2[0]).argmin():np.abs(self.wavelength-self.detector2[1]).argmin()]
            detector3 = data[np.abs(self.wavelength-self.detector3[0]).argmin():]
            # Apply SNV to each segment
            snv_detector1 = (detector1 - np.mean(detector1)) / np.std(detector1)
            snv_detector2 = (detector2 - np.mean(detector2)) / np.std(detector2)
            snv_detector3 = (detector3 - np.mean(detector3)) / np.std(detector3)
            # Combine results
            data_snv = np.concatenate((snv_detector1, snv_detector2, snv_detector3))
        else:
            data_snv = (data - np.mean(data)) / np.std(data)
        # store the info
        if Overide_processing == True:
            self.processing["base"] = data_snv
            self.metadata["processing"] = self.metadata["processing"]+'/SNV'
        else:
            self.processing[processing + '/SNV'] = data_snv
    # comment
    def areanorm(self, processing="base", split = False, wlrange = []):
        if wlrange:
            data = self.processing[processing][wlrange[0]:wlrange[-1]]
        else:
            data = self.processing[processing]        
        if split == True:
            # Cut the spectra for each detector
            detector1 = data[np.abs(self.wavelength-self.detector1[0]).argmin():np.abs(self.wavelength-self.detector1[1]).argmin()]
            detector2 = data[np.abs(self.wavelength-self.detector2[0]).argmin():np.abs(self.wavelength-self.detector2[1]).argmin()]
            detector3 = data[np.abs(self.wavelength-self.detector3[0]).argmin():]
            # Apply TAN to each segment
            norm_detector1 = detector1 / simpson(y=detector1, x=self.wavelength[np.abs(self.wavelength-self.detector1[0]).argmin():np.abs(self.wavelength-self.detector1[1]).argmin()]) 
            norm_detector2 = detector2 / simpson(y=detector2, x=self.wavelength[np.abs(self.wavelength-self.detector2[0]).argmin():np.abs(self.wavelength-self.detector2[1]).argmin()])
            norm_detector3 = detector3 / simpson(y=detector3, x=self.wavelength[np.abs(self.wavelength-self.detector3[0]).argmin():])
            # Combine results
            data_norm = np.concatenate((norm_detector1, norm_detector2, norm_detector3))
        else:
            data_norm = data / simpson(y=data, x=self.wavelength)
        # store the info
        if Overide_processing == True:
            self.processing["base"] = data_norm
            self.metadata["processing"] = self.metadata["processing"]+'/TAN'
        else:
            self.processing[processing + '/TAN'] = data_norm
    # comment
    def backgroundnorm(self, processing="base", wlrange = []):
        if wlrange:
            data = self.processing[processing][wlrange[0]:wlrange[-1]]
        else:
            data = self.processing[processing]        
        
            data_backgroundnorm = (data / self.background)
        # store the info
        if Overide_processing == True:
            self.processing["base"] = data_backgroundnorm
            self.metadata["processing"] = self.metadata["processing"]+'/BGN'
        else:
            self.processing[processing + '/BGN'] = data_backgroundnorm   
    # resample    
    def resample(self, wl_grid, wl_err):
        wl = self.wavelength
        data = self.processing['base']
        # split data by detectors
        detector1 = data[0:np.abs(wl-self.detector1[1]).argmin()]
        detector2 = data[np.abs(wl-self.detector2[0]).argmin():np.abs(wl-self.detector2[1]).argmin()]
        detector3 = data[np.abs(wl-self.detector3[0]).argmin():]
        # split wavelength by detectors  
        wl_detector1 = wl[0:np.abs(wl-self.detector1[1]).argmin()]
        wl_detector2 = wl[np.abs(wl-self.detector2[0]).argmin():np.abs(wl-self.detector2[1]).argmin()]
        wl_detector3 = wl[np.abs(wl-self.detector3[0]).argmin():]
        # split wl_err by detectors        
        err_detector1 = wl_err[0:np.abs(wl-self.detector1[1]).argmin()]
        err_detector2 = wl_err[np.abs(wl-self.detector2[0]).argmin():np.abs(wl-self.detector2[1]).argmin()]
        err_detector3 = wl_err[np.abs(wl-self.detector3[0]).argmin():]
        # comment 
        if len(err_detector3) > len(detector3):
            err_detector1 = err_detector1[:len(detector1)]
        if len(err_detector3) < len(detector3):
            detector3 = detector3[:len(err_detector3)]
            wl_detector3 = wl_detector3[:len(err_detector3)]
        # debugging line
        #print([self,self.metadata['index'],[len(detector1),len(wl_detector1),len(err_detector1)],[len(detector2),len(wl_detector2),len(err_detector2)],[len(detector3),len(wl_detector3),len(err_detector3)]])       
        # run NUMBA
        det1_resampled, resamplingerrors1 = spectral_resampling_numba.spectres_numba(np.array(wl_grid), np.array(wl_detector1), np.array(detector1), np.array(err_detector1), fill=0, verbose=False)
        det2_resampled, resamplingerrors2 = spectral_resampling_numba.spectres_numba(np.array(wl_grid), np.array(wl_detector2), np.array(detector2), np.array(err_detector2), fill=0, verbose=False)
        det3_resampled, resamplingerrors3 = spectral_resampling_numba.spectres_numba(np.array(wl_grid), np.array(wl_detector3), np.array(detector3), np.array(err_detector3), fill=0, verbose=False)
        # agreggate and store 
        self.processing["base"] = det1_resampled + det2_resampled + det3_resampled
        self.wavelength = wl_grid
        self.metadata['Resampled'] = True
    # find cutoff
    def find_cutoff(self):
        self.detector1 = []
        self.detector2 = []
        self.detector3 = []
        # detector1 / detector2
        cutoff1 = [np.abs(self.wavelength - 364).argmin(),np.abs(self.wavelength - 366).argmin()]
        wl_cutoff1 = self.wavelength[cutoff1[0]:cutoff1[1]]
        values_cutoff1 = self.processing['base'][cutoff1[0]:cutoff1[1]]
        values_der1 = np.diff(values_cutoff1)
        ind1 = np.abs(values_der1 - np.max(values_der1)).argmin()
        # detector2 / detector3                   
        cutoff2 = [np.abs(self.wavelength - 619).argmin(),np.abs(self.wavelength - 621).argmin()]
        wl_cutoff2 = self.wavelength[cutoff2[0]:cutoff2[1]]
        values_cutoff2 = self.processing['base'][cutoff2[0]:cutoff2[1]]
        values_der2 = np.diff(values_cutoff2)
        ind2 = np.abs(values_der2 - np.max(values_der2)).argmin()
        # store as attribute 
        self.detector1 = [self.wavelength[0], np.array(wl_cutoff1)[ind1+1]]
        self.detector2 = [np.array(wl_cutoff1)[ind1+1], np.array(wl_cutoff2)[ind2+1]]
        self.detector3 = [np.array(wl_cutoff2)[ind2+1], np.array(self.wavelength)[-1]]
    # linear interpolation
    def _Interpolate(self, x, y, x_out):
        # if is a list (from, to, by), build a np range 
        if isinstance(x_out, list): 
            x_out = np.arange(x_out[0], x_out[1], x_out[2])
        # interpolate 
        return np.interp(x_out, np.array(x), np.array(y))
    # fit peacks
    def fit_peaks(self, peaks_to_fit, doublepeaks_to_fit, processing, verbose=False, showplot=False):
        # add atribute 
        self.peaks = {}
        # x and y fits
        x_fit = np.linspace(self.wavelength[0], self.wavelength[len(self.wavelength)-1], len(self.wavelength)*50)
        y_fit = Akima1DInterpolator(self.wavelength, self.processing[processing])(x_fit)
        # comment 
        peaks = find_peaks_cwt(self.processing[processing], widths = np.arange(0.5,5))
        wavelengths = self.wavelength[peaks]
        peaks_interpolated = find_peak_indices(wavelengths, x_fit)
        specific_peaks_indexes = [min(peaks_interpolated, key=lambda x:abs(x_fit[x]-y)) for y in peaks_to_fit]
        fit_single_results = fit_single_peak(x_fit, y_fit, x_fit[specific_peaks_indexes], 3, 1, self.wavelength, self.processing[processing], verbose, showplot)                          
        double_peaks_indexes = [min(peaks_interpolated, key=lambda x:abs(x_fit[x]-y)) for y in [item for sublist in doublepeaks_to_fit for item in sublist]]
        fit_double_results = fit_double_peaks(x_fit, y_fit, x_fit[double_peaks_indexes], 2, self.wavelength, self.processing[processing], verbose, showplot)                          
        # comment     
        for i in range(0,len(fit_single_results)):
            self.peaks[peaks_to_fit_ref[i]]=fit_single_results[i]
        for i in range(0,len(fit_double_results)):
            self.peaks[doublepeaks_to_fit_ref[i]]=fit_double_results[i]
        for index, peak in enumerate(peaks_to_fit_ref):
            if abs(self.peaks[peak]["center"] - peaks_to_fit[index]) > 0.3:
                self.peaks[peak]["center"] = 0
                self.peaks[peak]["area"] = 0
                self.peaks[peak]["max_value"] = 0
                self.peaks[peak]["fwhm"] = 0
        for index, peak in enumerate(doublepeaks_to_fit_ref):
            if abs(self.peaks[peak]["center"] - [item for sublist in doublepeaks_to_fit for item in sublist][index]) > 0.3:
                self.peaks[peak]["center"] = 0
                self.peaks[peak]["area"] = 0
                self.peaks[peak]["max_value"] = 0
                self.peaks[peak]["fwhm"] = 0
#=======================================================================================

#Class to build and manage a LIBS_dataset object, which is a collection of LIBS spectra objects

class LIBS_dataset: 
    def __init__(self):
        # basic empty attribute
        self.spectra = []    
    # Info about the class
    def __str__(self):
        return f"Dataset of {len(self.spectra)} LIBS spectra"
    # creates LIBS_spectra objects and load data  
    def loadData(self, path):
        print("loading data...\n")
        # Lista de archivos CSV a procesar
        files_to_process = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    files_to_process.append(os.path.join(root, file))
        if Load_all_wl: 
            # Crear un Pool de procesos para paralelizar el trabajo
            with ThreadPool() as pool:
                results = pool.map(self._process_file, files_to_process)
                # Agregar los resultados al atributo spectra
            self.spectra.extend(filter(None, results))
        else: 
            # do the first
            self._process_file(files_to_process[0])
            # take the wavelength of he first as reference 
            wavelength_out = self.spectra[0].wavelength
            arguments = [(file, wavelength_out) for file in files_to_process]
            with ThreadPool() as pool:
                # Usar pool.map con la lista de tuplas
                results = pool.map(self._process_file_interpol, arguments)
                self.spectra.extend(filter(None, results))
    # auxiliar function for multiprocess (not try-except)
    def _process_file(self, file):
        ls = LIBS_spectra()
        ls.loadData(file, wavelength_out = None)
        return ls
    # auxiliar function for multiprocess with interpolation (not try-except)
    def _process_file_interpol(self, args):
        file, wavelength_out = args  # Desempacar los argumentos
        ls = LIBS_spectra()
        ls.loadData(file, wavelength_out)
        return ls
    # match metadata 
    # creates a dictionary inside the LIBS class with keyas as "getColumns" and values as data in the file
    # to do: multiprocess
    def matchMetadata(self, filePath = [], matchName = [], getColumns = []):
        if not filePath:
            for index, i in tqdm(enumerate(self.spectra)):
                i.metadata["index"] = index
        else:        
            # load file 
            df = pd.read_csv(filePath, engine="pyarrow")
            # loop til spectra
            for index, i in tqdm(enumerate(self.spectra)): 
                # take id 
                Id = i.test_number
                # check if test_number is in the data 
                if not Id in list(df[matchName]): 
                    print(f"caution!: test number {Id} not present in the data file")
                # take vaues
                else: 
                    values = df.loc[df[matchName] == Id, getColumns].values.tolist()[0]
                    # build dict and add the index
                    i.metadata.update(zip(getColumns, values))
                    i.metadata["index"] = index
    # dataframe builder 
    def buildDataframe(self, metadata, processing = "base", interpolate = False, waveRange = 0, Index = [], excluded_shots=[]): 
        if interpolate == True:        
            if isinstance(waveRange, list):                                 # if waveRange is a list of tree values, use an equal-interval sequence (from, to, by)
                x_out = np.arange(waveRange[0], waveRange[1], waveRange[2]) 
            elif isinstance(waveRange, int):                                # if waveRange is an integer, use the wavelength that index in self.spectra
                x_out = np.array(self.spectra[waveRange].wavelength)
            else:                                                           # if is an np.array, use the values of it (needs elif)
                x_out = waveRange
            dfcolumns = metadata + list(np.around(x_out, 2)) # round for label (aboid floating coma error) 
        else:
            dfcolumns = metadata + self.spectra[0].wavelength[0:5839].tolist()
        tmplist = []
        # set list to loop  
        if not Index: 
            SPECTRA = self.spectra
        else: 
            SPECTRA = [self.spectra[i] for i in Index]
        # loop til SPECTRA
        for i in tqdm(SPECTRA):
            # Extract the metadata
            if i.shot_number in excluded_shots:
                continue
            meta = [i.metadata[j] for j in metadata]
            if interpolate == True :
                #Interpolate using the specified array
                vals = list(i._Interpolate(i.wavelength, i.processing[processing], x_out))
            else:
                vals = i.processing[processing][:5839].tolist()
            #Append to list of list
            tmplist.append(meta + vals)
        # build the df 
        df = pd.DataFrame(tmplist, columns=dfcolumns)
        # return
        return df
    # build dataframe of peaks 
    def buildDataframePeaks(self, metadata, Index = [], features = ["max_value", "area", "fwhm"]): 
        dfcolumns = metadata + [i + "_" + j for i in peaks_to_fit_ref + doublepeaks_to_fit_ref for j in features]
        tmplist = []
        # set list to loop  
        if not Index: 
            SPECTRA = self.spectra
        else: 
            SPECTRA = [self.spectra[i] for i in Index]
        # loop til spectra
        for i in tqdm(SPECTRA):
            # Extract the metadata
            meta = [i.metadata[j] for j in metadata]
            vals = [i.peaks[ref][val] for ref in peaks_to_fit_ref + doublepeaks_to_fit_ref for val in features]
            # Append to list of list
            tmplist.append(meta + vals)
        # build the df     
        df = pd.DataFrame(tmplist, columns=dfcolumns)
        # return
        return df
    # organize simpson integration data
    def integrate(self, key, exclude = 8, processing="base", Min=False, Max=False):
        tmp_data = []
        for i in self.spectra:
            i.Simpson(processing, Min, Max, key) # perform simpson integration 
            tmp_data.append((i.test_number, i.shot_number, i.areas[key]))
        # build the data frame
        self.simp_df = pd.DataFrame(tmp_data, columns=["test_number", "shot_number", "simpson"])
        # locus
        values = [i for i in range(1, 13) for _ in range(4)]
        locus = {(i+1): values[i] for i in range(48)}
        self.simp_df['locus'] = [locus[i] for i in self.simp_df['shot_number']]
        # shot in locus
        values = [i for _ in range(12) for i in range(1, 5)]
        shot_locus = {(i+1): values[i] for i in range(48)}
        self.simp_df['shot_locus'] = [shot_locus[i] for i in self.simp_df['shot_number']]
        # obtain the matrix (measure x shot number)
        self.simp_mat = self.simp_df.pivot(index="test_number", columns="shot_number", values="simpson")
        self.simp_mat = self.simp_mat.iloc[:, exclude:] # eclude the n first columns
    # set as reference data
    def reference_data(self): 
        # vectors of mean and std (for each shot) 
        self.ref_mean = self.simp_mat.mean()
        self.ref_std = self.simp_mat.std()
        # mean of means and mean of std
        self.ref_mean_means = np.mean(self.ref_mean)
        self.ref_mean_stds = np.sum(self.ref_std**2 / len(self.ref_std))**(1/2)
    # simple plot simpsons
    def plot_simpsons(self, by):
        boxplot = self.simp_df.boxplot(column='simpson', by=by)
        boxplot.show() 
    #Launch dash plot of all spectra and processing in dataset
    def plot_spectra(self, scale = False):
        #Initialise spectra
        app = Dash()
        app.layout = [
            html.H1(children=self.__str__(), style={'textAlign':'center'}),
            dcc.Dropdown([str(self.spectra[i]) for i in range(len(self.spectra))], [str(self.spectra[i]) for i in range(len(self.spectra))][0], id='dropdown-selection'),
            dcc.Graph(id='graph-content')
            ]
        @callback(
            Output('graph-content', 'figure'),
            Input('dropdown-selection', 'value')
            )
        def update_graph(value):
            index = [str(self.spectra[i]) for i in range(len(self.spectra))].index(value)
            spectra = self.spectra[index]
            spectra_df = pd.DataFrame(columns=["wavelength", "spectra", "intensity"])
            for k in spectra.processing.keys(): #Scaling makes no sense from a statistical point of vue but is interesting for visualisation, hence the option being present as in the plot function but not as a processing
                if scale:
                    spectra_df = pd.concat([spectra_df, pd.DataFrame({
                        "wavelength": spectra.wavelength,
                        "spectra": np.repeat(k, len(spectra.wavelength)),
                        "intensity": spectra.processing[k]/max(spectra.processing[k])*100
                        })])
                else:    
                    spectra_df = pd.concat([spectra_df, pd.DataFrame({
                        "wavelength": spectra.wavelength,
                        "spectra": np.repeat(k, len(spectra.wavelength)),
                        "intensity": spectra.processing[k]
                        })])
            return px.line(spectra_df, x='wavelength', y='intensity', color='spectra', template='simple_white')
        app.run_server(debug=True)
        
    def iff(self, metadata = ["test_number", "shot_number","index"], processing = "base", waveRange = 0, Index = [], subselection = []):
        if not subselection:
            wlmin = len(metadata)
            wlmax = -1
        if isinstance(subselection, list):
            wlmin = 3+subselection[0]
            wlmax = subselection[1]
        #Calls the get_dataframe function to get data usable for IFF       
        data = self.buildDataframe(metadata, processing, waveRange, Index)
        #Calls the iff julia code
        ind, freq = jl.iff(jlconvert(jl.Matrix, data.iloc[:,wlmin:wlmax]), 10000)
        #Formats the results with a bit of index trickery (julia indexing starts at one) and puts the score as an attribute        
        for idx, x in enumerate(ind):
            self.spectra[data["index"][x-1]].metadata["iff_score"] = freq[idx]
    # resample dataset
    def RessampleDataset(self, detectors = [[187.8,365],[365,620],[620,946.2]], resolution = [0.1,0.15,0.2], Index = []):
        wl_err = []
        wl_grid = np.concat([np.arange(detectors[0][0],detectors[0][1],resolution[0]), np.arange(detectors[1][0],detectors[1][1],resolution[1]), np.arange(detectors[2][0],detectors[2][1],resolution[2])])
        data = self.buildDataframe(metadata = ["test_number", "shot_number","index"], processing = "base", waveRange=0, Index=Index)
        for i in range(3,data.shape[1]):
            wl_err.append(data[data.columns[i]].std())
        for i in tqdm(self.spectra):
            i.resample(wl_grid, wl_err)
            i.find_cutoff()


#def main():
#    start_time = time.time()
    # load data
#    DATA = LIBS_dataset()
#    DATA.loadData('./Chert/')
    # match metadata 
#    DATA.matchMetadata(filePath="LIBS_analyses_Elena.csv", matchName="N_test", getColumns=["Jour", "Nom", "Formation"])
    # build dataframe 
#    metadata = ["Nom", "Formation", "index"]
#    processing = "base"
#    waveRange = 0 # [187.7, 675, 0.96] # from, to, by
#    waveRange = np.array(DATA.spectra[0].wavelength)[0:4000]
#    import random as rd
#     Index = rd.sample(range(0, len(DATA.spectra)), 100)
#    df = DATA.buildDataframe(metadata, processing, waveRange, Index)
#    print("--- %s seconds ---" % (time.time() - start_time))
            
#if __name__ == "__main__":
#    main()

