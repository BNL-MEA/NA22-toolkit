# Packages necessary
# General
import scipy
import h5py
import pandas as pd
import numpy as np
import xarray as xr
from itertools import compress
import os



# Fitting
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
from time import perf_counter
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import linregress
from sklearn.metrics import r2_score


# Plotting
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from plotly.subplots import make_subplots
from matplotlib.cm import get_cmap


# X-ray database
import xraydb as xdb

# Smoothing
import pywt
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


########## Normalization ##########
# * creates a new dataset in an h5 file containing the normalized counts from the detector
# * good for FIJI/ImageJ analysis
def normalization(file):
    with h5py.File(coarse_filename4, 'a') as h5file:
        counts = h5file['xrfmap/detsum/counts'][:]
        ion_chamber_data = h5file['xrfmap/scalers/val'][:, :, 0]
        counts_norm = counts/ion_chamber_data[:, :, np.newaxis]
        dataset = h5file.create_dataset('xrfmap/detsum/counts_norm',data = counts_norm)


def normalize(data):
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)


# calculate the distance between values x1 and x2
def distance(x1,x2):
    return np.sqrt((x1 - x2)**2)


########## Combine lists and remove duplicates ##########
# function primarily used to combine lists of identified elements in Spectrum into a single list
# * Inputs
#     1. N number of lists all with the same dimensions (Mx5)
# * Outputs
#     2. processed_list: Mx5 dimensional list containing all identified elements with no duplicates
def compile_and_remove_duplicates(*lists):
    """
    Compile multiple lists into a single list and remove duplicate rows across all lists.

    Args:
        *lists: Variable number of lists.

    Returns:
        A list with unique rows across all input lists, with an added index column,
        without the original first column, and sorted based on the third column.
    """
    combined_list = []
    for sublist in lists:
        combined_list.extend(sublist)

    # Remove the original first column from each sublist
    without_first_column = [row[1:] for row in combined_list]

    unique_set = set()
    unique_list = []

    for row in without_first_column:
        first_two_values = tuple(row[:2])
       
        if first_two_values not in unique_set:
            unique_set.add(first_two_values)
            unique_list.append(row)

    
    # Sort based on the third column (index 2)
    sorted_unique_list = sorted(unique_list, key=lambda x: x[2])      
    
    # Add an index column
    processed_list = [[idx] + row for idx, row in enumerate(sorted_unique_list, start=1)]
    
    return processed_list
    



########## Handle user inputs ##########
## Function to Convert user_input into slice
# * Inputs
#     1. user_inputs must be from built-in 'input()' function
# * Outputs
#     2. slice of range inputted by user
def input_to_slice(user_input):
    try:
        # Split the input into start, stop, and step components
        parts = user_input.split(':')

        # Convert parts to integers or leave them as None if not provided
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        

        # Create and return the slice
        return slice(start, stop)
    except ValueError:
        print("Invalid input. Please use the format 'start:stop'.")


def equalize_lengths(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)

    # Compare lengths
    if len1 > len2:
        # Remove elements from the end of arr1
        arr1 = arr1[:len2]
    elif len2 > len1:
        # Remove elements from the end of arr2
        arr2 = arr2[:len1]

    return arr1, arr2        



        
########## Smooth Data ##########
# Denoises data using a wavelet transform. Then smooths data using the Savitzky-Golay filter 
# which increases data precision without distorting signal tendency. fits successive sub-sets
# of adjacent data points with a low degree polynomial via linear least squares. The code 
# automatically determines the optimum polynomial degree and data window based on the data 
# provided. For application on fluorescence data it's helpful to convert y to log scale prior
# to entering in algorithm.
# Inputs: 
#   1. x : x data in 1D np.array
#   2. y : y data in 1D np.array 
# Outputs:
#   1. smoothed_y : smoothed y data as 1D np.array
def denoise_and_smooth_data(x,y):
    ########## Smooth data (Savitzky-Golay filter)##########
    # Define ranges of window sizes and polynomial degrees to try
    window_sizes = range(5, 30, 2)  # Adjust as needed
    polynomial_degrees = range(2, 5)  # Adjust as needed
    
    # Perform k-fold cross-validation to choose optimal window size and polynomial degree
    kf = KFold(n_splits=5, shuffle=True, random_state = 42)
    best_mse = float('inf')
    best_window_size = None
    best_poly_degree = None
    
    for window_size in window_sizes:
        for poly_degree in polynomial_degrees:
            fold_mse = 0
            for train_index, val_index in kf.split(x):
                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]
                # Apply Savitzky-Golay filter with current window size and polynomial degree
                smoothed_y = savgol_filter(y_train, window_size, poly_degree)
                
               
                # Evaluate smoothed data on validation set
                x_train, smoothed_y = equalize_lengths(x_train, smoothed_y)
                val_predictions = np.interp(x_val, x_train, smoothed_y)
                fold_mse += mean_squared_error(y_val, val_predictions)
            
            fold_avg_mse = fold_mse / kf.n_splits
            
            # Update best parameters if current ones are better
            if fold_avg_mse < best_mse:
                best_mse = fold_avg_mse
                best_window_size = window_size
                best_poly_degree = poly_degree
    
    # Apply Savitzky-Golay filter with the best parameters
    smoothed_y = savgol_filter(y, best_window_size, best_poly_degree)

    
    ########## Denoise data (wavelet transform) ##########
    # Perform wavelet decomposition
    wavelet = 'db4'  # Choose a wavelet type, e.g., Daubechies 4
    levels = 8  # Number of decomposition levels
    coeffs = pywt.wavedec(smoothed_y, wavelet, level=levels)
    
    # Define range of threshold values to try
    threshold_values = np.linspace(0.001, 0.5, 500)  # Adjust as needed
    
    # Perform k-fold cross-validation to choose optimal threshold value
    kf = KFold(n_splits=5, shuffle=True, random_state = 42)
    best_mse = float('inf')
    best_threshold = None
    
    for threshold in threshold_values:
        fold_mse = 0
        for train_index, val_index in kf.split(x):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = smoothed_y[train_index], smoothed_y[val_index]
            
            # Perform wavelet denoising with current threshold
            thresholded_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            denoised_y = pywt.waverec(thresholded_coeffs, wavelet)
            
            # Interpolate denoised signal at original data points
            x, denoised_y = equalize_lengths(x, denoised_y)
            interpolated_denoised_y = np.interp(x_train, x, denoised_y)
            
            # Evaluate interpolated denoised data on validation set
            x_train, interpolated_denoised_y = equalize_lengths(x_train, interpolated_denoised_y)
            val_predictions = np.interp(x_val, x_train, interpolated_denoised_y)
            fold_mse += mean_squared_error(y_val, val_predictions)
        
        fold_avg_mse = fold_mse / kf.n_splits
        
        # Update best threshold value if current one is better
        if fold_avg_mse < best_mse:
            best_mse = fold_avg_mse
            best_threshold = threshold

    # Perform final denoising with the best threshold value
    thresholded_coeffs = [pywt.threshold(c, best_threshold, mode='soft') for c in coeffs]
    denoised_and_smoothed_y = pywt.waverec(thresholded_coeffs, wavelet)


 
    return denoised_and_smoothed_y


########## Identify Elements ##########
## Function to find element fluorescence line that matches each peak
# * Inputs: 
#     1. elements: string of elements supplied by user consisting of those expected in the sample
#     2. peaks: list of energies corresponding to each peak (eV)
#     3. tolerance: percentage of acceptable difference between the database x-ray fluorescence energy and the supplied peak energies
#     4. incident_energy: energy in eV of the incident x-ray beam
# * Outputs: 
#     1. matched_fluor_lines: list of matched fluorescence lines showing the peak name, the element, the fluorescence line name, the energy (eV), and the relative intensity
#     2. matched_df: pandas dataframe of the matches
def identify_element_match(elements, peaks, tolerance, incident_energy):
    line_name_int = []
    identified_element = []
    peak_intensity = [] 
    energy_match = []
    matched_peak = []
    for element in elements:
        xray_line = xdb.xray_lines(element)
        line_names = list(xdb.xray_lines(element))
    
        for i in range(len(line_names)):
            fluor_energy = list(xray_line.values())[i][0] # output fluorscence energy of the selected element in the i-th index
            rel_intensity = list(xray_line.values())[i][1] # output relative intensity of the selected element in the i-th index
            absorption_edge_id = list(xray_line.values())[i][2][:2] # output absoprtion edge name of the selected element in the i-th index
            absorption_edge_energy = list(xdb.xray_edge(element,absorption_edge_id))[0] # absorption edge energy in eV
            
            if absorption_edge_energy < incident_energy:            
                # find fluorscence line that matches to each peak
                for j, peak in enumerate(peaks):
                    largest_value = max(peak,fluor_energy)
                    peak_diff = (abs(fluor_energy - peak)/ largest_value)*100
        
                    # find values within set tolerance threshold percent
                    if peak_diff <= tolerance:
                        identified_element.append(element)
                        line_name_int.append(line_names[i])
                        energy_match.append(float(fluor_energy))
                        peak_intensity.append(float(rel_intensity))
                        matched_peak.append(int(j+1))
    
    # element_emission_line = [item1 + '_' + item2 for item1, item2 in zip(identified_element, line_name_int)]
    
    # Output list of matched elements, the fluorescence line name, and the energy (eV)
    matched_fluor_lines = sorted([list(a) for a in zip(matched_peak, identified_element, line_name_int, energy_match, peak_intensity)])
    
    column_names =  ["Peak #", "Element", "Emission Line", "Energy (eV)", "Relative Intensity"]
    matched_df = pd.DataFrame(data = matched_fluor_lines, columns = column_names)
    matched_df = matched_df.drop_duplicates()
    
    # making list in the same order as dataframe
    line_name_int = matched_df['Emission Line'].tolist()
    energy_match = matched_df['Energy (eV)'].tolist()
    rel_intensity = matched_df["Relative Intensity"].tolist()
    
    # Removing repeats and averaging fluor line of elements with same emission lines (i.e. averaging Ce_Ka1, Ce_Ka2, etc. to make single peak representing Ce_Ka)
    unique_peak = matched_df['Peak #'].unique()
    matched_peaks = []
    matched_energy = []
    rel_int = []
    line_name = []
    matched_element = []
    for i in unique_peak:
        idx_peak = matched_df['Peak #'] == i
        peak_elements = matched_df['Element'][idx_peak]
        for j in set(peak_elements):
            idx_element = matched_df['Element'] == j
            idx_peak_element = idx_peak & idx_element
            
            if sum(idx_peak_element) > 1: 
                matched_peaks.append(i)
                matched_element.append(j)
                energy_int = list(compress(energy_match,idx_peak_element))
                intensity = list(compress(rel_intensity,idx_peak_element))
                line_names = list(compress(line_name_int,idx_peak_element))
                line_name.append(line_names[0][:-1])
                temp_e = 0
                temp_int = 0
                for k in range(sum(idx_peak_element)):
                    temp_e += energy_int[k]*intensity[k]
                    temp_int += intensity[k]
                matched_energy.append(float(temp_e/temp_int))
                rel_int.append(float(temp_int))
            if sum(idx_peak_element) == 1:
                matched_peaks.append(i)
                matched_element.append(j)
                energy_int = list(compress(energy_match,idx_peak_element))
                matched_energy.append(np.mean(energy_int))
                rel_int.extend(list(compress(rel_intensity,idx_peak_element)))
                line_name.extend(list(compress(line_name_int,idx_peak_element)))
    
    
    # Output list of matched elements, the fluorescence line name, and the energy (eV)
    matched_fluor_lines = sorted([list(a) for a in zip(matched_peaks, matched_element, line_name, matched_energy, rel_int)], key=lambda l:l[3])
    
    column_names =  ["Peak #", "Element", "Emission Line", "Energy (eV)", "Relative Intensity"]
    matched_df = pd.DataFrame(data = matched_fluor_lines, columns = column_names)

    
    return matched_fluor_lines, matched_df


########## Defining Gaussians ##########
# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * std_dev**2))



########## define the sum of multiple gaussians ##########
def multi_gaussians(x, *params):
    num_gaussians = len(params) // 3
    result = np.zeros_like(x)
    
    for i in range(num_gaussians):
        amp, mean, stddev = params[i*3 : (i+1)*3]
        result += gaussian(x, amp, mean, stddev)
    
    return result
    


########## Fit background ##########
## Function to fit spectra background
# arpls approach to fit background
r"""
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Abstract

    Baseline correction methods based on penalized least squares are successfully
    applied to various spectral analyses. The methods change the weights iteratively
    by estimating a baseline. If a signal is below a previously fitted baseline,
    large weight is given. On the other hand, no weight or small weight is given
    when a signal is above a fitted baseline as it could be assumed to be a part
    of the peak. As noise is distributed above the baseline as well as below the
    baseline, however, it is desirable to give the same or similar weights in
    either case. For the purpose, we propose a new weighting scheme based on the
    generalized logistic function. The proposed method estimates the noise level
    iteratively and adjusts the weights correspondingly. According to the
    experimental results with simulated spectra and measured Raman spectra, the
    proposed method outperforms the existing methods for baseline correction and
    peak height estimation.

    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector

    """

def arpls(y, lam=1e3, ratio=0.01, itermax=10000):
    N = len(y)
#  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)        
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z




## Spectra Fitting Function
def peak_fitting(x, y, peaks, window):
    window = round(window/2)
    
    # Fit background data
    baseline_fit = arpls(y)

    # Fit Gaussian to each peak
    peak_tails = []
    bounds_lower_all = []
    bounds_upper_all = []
    popt_all = []
    y_baseline_subtracted = y - baseline_fit
    
    for peak_index in peaks:
        x_peak = x[max(0, peak_index - window): min(len(x), peak_index + window)]
        y_peak = y_baseline_subtracted[max(0, peak_index - window): min(len(y), peak_index + window)]

        # setting gaussian parameters
        amplitude = max(y_peak)
        center = x[peak_index]
        std_dev = np.std(x_peak)

        # setting bounds for single peak fit
        amp_variation = 0.5 * 10**np.floor(np.log10(np.abs(amplitude)))
        bounds_lower = [amplitude-amp_variation,center-0.1,std_dev-0.1] 
        bounds_upper = [amplitude+amp_variation,center+0.1,std_dev+0.1]
        bounds = scipy.optimize.Bounds(lb = bounds_lower, ub = bounds_upper)
        
        # Fit Gaussian
        popt, _ = curve_fit(gaussian, x_peak, y_peak, p0=[amplitude, center, std_dev], maxfev = int(1e8), bounds = bounds)
        popt_all.extend(popt)
        
    
        # setting bounds for cumulative fit
        amp_variation = 0.5 * 10**np.floor(np.log10(np.abs(popt[0])))
        bounds_lower_all.extend([popt[0]-amp_variation,popt[1]-0.1,popt[2]-0.1])
        bounds_upper_all.extend([popt[0]+amp_variation,popt[1]+0.1,popt[2]+0.1])
        
    
    # Set bounds for multigaussian fit
    bounds_all = scipy.optimize.Bounds(lb = bounds_lower_all, ub = bounds_upper_all)
   

    # Fit results to multi-gaussian function
    popt, _ = curve_fit(multi_gaussians, x, y_baseline_subtracted, p0 = popt_all, maxfev = int(1e8), bounds= bounds_all)
    multi_gaussian_fit = multi_gaussians(x, *popt)
        

    # final fit with baseline added
    peak_fit = multi_gaussian_fit + baseline_fit
    r_squared = linregress(peak_fit, y).rvalue**2


    return peak_fit, baseline_fit, popt_all, r_squared
    

########## AOI Analysis ##########
## Function to determine optimum paramters to extract Detector Area of Interest Spectrum
# * Inputs
#     1. filename: file path to hdf5 (.h5) file containing total scan data
#     2. min_energy: minimum of energy range of interest (keV)
#     3. elements: anticipated elements contained in the sample from background analysis using pyXRF
# * Outputs
#     1. detector_2D_map_fig: HTML figure containing 2D map of detector
#     2. fig1: HTML figure containing all relevant data/information processed that
#        can be used later to plot the results        
#     3. peak_fit_params: parameters used to define the gaussian fit of peaks in background subtracted partilce spectrum
def AOI_particle_analysis(filename, min_energy, sample_elements, background_elements, denoise = True, Sigray = False):
    ########## Load data filenin variable ##########
    with h5py.File(filename, 'r') as file:
        data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        if not Sigray:
            ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
            if not Sigray:
                ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
                dwell_time = attributes['param_dwell'] # seconds
                
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")
    
    # Position axes
    x_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),data.shape[1])
    y_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),data.shape[0])
    if Sigray:
        y_pos = y_pos[::-1]
    
    # normalize data by ion_chmaber_data(i0)
    if not Sigray:
        data = data/ion_chamber_data[:,:,np.newaxis]
        data = data/dwell_time

    # Use incident X-ray energy to define energy range of interest 
    # incident_wavelength = 1.2398e-9/incident_energy # convert incident energy to wavelength (hc/lambda)
    # compton_wavelength = 4.86e-12 + incident_wavelength # determine compton wavelength using maximum wavelength differential plus incident 
    # max_energy = 1.2398e-9/compton_wavelength  # convert compton wavelength to energy and set the maximum energy to about the compton peak 
    max_energy = incident_energy
    energy = 0.01*np.arange(data.shape[2])
    min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
    max_idx = min([i for i, v in enumerate(energy) if v >= incident_energy])


    # Total average spectrum
    avg_data = np.mean(data, axis = (0,1)) 
    avg_data = avg_data[min_idx:max_idx]
   
    
    ########## Plotting whole detector view to identify AOI ##########
    # determine max intensity of each pixel to get better particle resolution
    max_int = np.max(data, axis = 2, keepdims = True)
    temp = np.sum(max_int, axis = (2))
    
    
    # determine the bounds of the color map based on robust scaling similar to xarray robust=True
    vmin = np.quantile(temp, 0.02)
    vmax = np.quantile(temp, 1- 0.02)
    
    # plotting the map
    detector_2D_map_fig = go.Figure(data = go.Heatmap(z = temp, zmin = vmin, zmax = vmax, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
    detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                      title_x = 0.5,
                                      width = 500,
                                      height = 500,
                                      font = dict(size = 20),
                                      xaxis = dict(title = 'X-axis'),
                                      yaxis = dict(title = 'Y-axis'))
    if Sigray:
        detector_2D_map_fig.update_layout(yaxis=dict(autorange='reversed'))
    detector_2D_map_fig.show()

    ########## Handling bad pixels ##########
    bad_pixels = input("Smooth over bad pixels? (Yes or No):")
    if bad_pixels.lower() == "yes":
        # get number of values to extract
        nbad_pixels = input("Input integer value for number of bad pixels based on number unique xy coordinates showing distinctly lower intensity:")
        k = int(nbad_pixels) # number of values to be extracted 
        idx_flat = np.argpartition(temp.flatten(),k)[:k] # index of k lowest values 
        idx_2d = np.unravel_index(idx_flat,temp.shape)
        temp[idx_2d] = np.mean(temp) # new detecotr data without dead pixels 
        
         # determine the bounds of the color map based on robust scaling similar to xarray robust=True
        vmin = np.quantile(temp, 0.02)
        vmax = np.quantile(temp, 1- 0.02)


        # plot new data
        detector_2D_map_fig = go.Figure(data = go.Heatmap(z = temp, zmin = vmin, zmax = vmax, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
        detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                          title_x = 0.5,
                                          width = 500,
                                          height = 500,
                                          font = dict(size = 20),
                                          xaxis = dict(title = 'X-axis'),
                                          yaxis = dict(title = 'Y-axis'))
        if Sigray:
            detector_2D_map_fig.update_layout(yaxis=dict(autorange='reversed'))
        detector_2D_map_fig.show()

    # plotting detector mapping highlighting location of each sample_element
    if sample_elements:
        element_2D_map_fig = make_subplots(rows= len(sample_elements), cols=1, subplot_titles = sample_elements)
        for count,element in enumerate(sample_elements): # loop through each element in the sample
            xray_data = list(xdb.xray_lines(element).values()) # find the xray data for each sample element
            element_energy = []
            for _,line in enumerate(xray_data): # loop through each xray line in each sample element
                absorption_edge_id = line[2][:2] # output absoprtion edge name of the selected element in the i-th index
                absorption_edge_energy = list(xdb.xray_edge(element,absorption_edge_id))[0] # absorption edge energy in eV
                if line[1] >= 0.1 and absorption_edge_energy < max_energy*1000: # if the line has adequete intensity and below max_energy append to list
                    element_energy = np.append(element_energy,[line[0],line[1],line[2]])
                    
                    
        
            element_energy = element_energy.reshape((int(len(element_energy)/3),3)) # reshape to be n x 3 array
            element_energy = np.array([row for row in element_energy if row[2] == element_energy[0,2]]) # select only the first edge available
            max_idx_element = np.argmax(element_energy[:,1]) # find maximum intensity of all edge lines

            # setting up an energy range of interest surrounding the highest intensity line in the first edge available
            energy_range_minima = round((float(element_energy[max_idx_element,0]) - 500)/10) 
            energy_range_maxima = round((float(element_energy[max_idx_element,0]) + 500)/10)
            energy_range = slice(energy_range_minima, energy_range_maxima)

            # extracting this energy range from the hdf file data
            element_det_data = data[:, :, energy_range]
            element_det_data_max_int = np.max(element_det_data, axis = 2, keepdims = True)
            summed_element_det_data = np.sum(element_det_data_max_int, axis=(2))
            
            if bad_pixels.lower() == "yes":
                k = int(nbad_pixels) # number of values to be extracted 
                idx_flat = np.argpartition(summed_element_det_data.flatten(),k)[:k] # index of k lowest values 
                idx_2d = np.unravel_index(idx_flat,summed_element_det_data.shape)
                summed_element_det_data[idx_2d] = np.mean(summed_element_det_data) # new detector data without dead pixels 

            # plotting detector map highlighting element of interest's signal
            element_2D_map_fig.add_trace(go.Heatmap(z = summed_element_det_data), row = count + 1, col = 1 )
      

        element_2D_map_fig.update_traces(dict(showscale=False, 
                                              coloraxis=None, 
                                              colorscale='Viridis'), 
                                         selector={'type':'heatmap'})
        

        element_2D_map_fig.update_layout(width = 425, height = len(sample_elements)*500, 
                                         font = dict(size = 20),
                                         xaxis = dict(title = 'X-axis'),
                                         yaxis = dict(title = 'Y-axis'))
        if Sigray:
            element_2D_map_fig.update_layout(yaxis=dict(autorange='reversed'))
        element_2D_map_fig.show()
    


    ######### Selecting area of interest based on XRF mappings ##########
    # # y-direction
    user_input = input("Utilizing the detector map outputted, enter x values for area of interest (AOI) in slice format (e.g., '1:5'):")
    detector_ROI_columns = input_to_slice(user_input)
    detector_ROI_columns = slice(detector_ROI_columns.start+1, detector_ROI_columns.stop+1)
    
    
    # # x-direction
    user_input = input("Utilizing the detector map outputted, enter y values for area of interest (AOI) in slice format (e.g., '1:5'):")
    detector_ROI_rows = input_to_slice(user_input)
    detector_ROI_rows = slice(detector_ROI_rows.start,detector_ROI_rows.stop)
    
    
    AOI_data = data[detector_ROI_rows, detector_ROI_columns, :]
    y_int = y_pos[detector_ROI_columns]
    x_int = x_pos[detector_ROI_rows]
 
   
    # Avg spectrum in selected area
    AOI = np.mean(AOI_data, axis=(0,1))
    AOI = AOI[min_idx:max_idx]
    energy_int = energy[min_idx:max_idx]
    
    

    ######### Selecting background area based on PyXRF mappings ##########

    # # y-direction
    user_input = input("Subtract self-consistent detector background (yes or no)?:")
    if user_input.lower() == 'yes':
        AOI_shape = AOI_data.shape # get the shape of the area selected by the user above
        npoints = AOI_shape[0]* AOI_shape[1] # find the number of pixels/points in the area selected above
        
        # find the lowest intensities plotted in detector map
        lowest_intensities = np.argsort(temp.flatten())[:npoints] # sort the intensities to find the npoints lowest values
        low_int_idx = np.unravel_index(lowest_intensities, temp.shape) # extract 2d indices of lowest intensity

        bkg_data = data[low_int_idx[1][:,np.newaxis], low_int_idx[0][:,np.newaxis], :] # extracting bkg_Data based on lowest intensities
       
        # Avg background spectrum in selected area
        background = np.mean(bkg_data, axis=(0,1))
        background = background[min_idx:max_idx]       

        
        # Background subtracted AOI
        baseline = arpls(background) # Baseline of AOI spectrum
        AOI_bkg_sub = AOI - background
        AOI_bkg_sub[AOI_bkg_sub <= 0] = 0

        # add baseline to AOI spectrum
        AOI_bkg_sub = AOI_bkg_sub + baseline
    else:
        user_input = input("Utilizing the detector map outputted, enter x values for area containing background spectra in slice format (e.g., '1:5'). Input 'none' to analyze whole detector:")
        if user_input.lower() != 'none':
            detector_ROI_columns = input_to_slice(user_input)
            detector_ROI_columns = slice(detector_ROI_columns.start+1, detector_ROI_columns.stop+1)
            
            # # x-direction
            user_input = input("Utilizing the detector map outputted, enter y values for area containing background spectra in slice format (e.g., '1:5'):")
            detector_ROI_rows = input_to_slice(user_input)
            
    
            # identify background spectrum
            bkg_data = data[detector_ROI_rows, detector_ROI_columns, :]
         
         
           
            # Avg background spectrum in selected area
            background = np.mean(bkg_data, axis=(0,1))
            background = background[min_idx:max_idx]
            
    
            
            # Background subtracted AOI
            baseline = arpls(background) # Baseline of AOI spectrum
            AOI_bkg_sub = AOI - background
            AOI_bkg_sub[AOI_bkg_sub <= 0] = 0
    
            # add baseline to AOI spectrum
            AOI_bkg_sub = AOI_bkg_sub + baseline
            
        else:
            blank_filename = input("Input blank file data at same incident energy as sample data (if yes input variable containing file info, else enter no)?")
            if blank_filename.lower() == 'no':
                baseline = arpls(AOI)
                AOI_bkg_sub = AOI - baseline
            else: 
                with h5py.File(blank_filename, 'r') as file:
                    blank_data = file['xrfmap/detsum/counts'][:]
                    blank_ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
                    group_name = 'xrfmap/scan_metadata'
                    if group_name in file:
                        group = file[group_name]
                        attributes = dict(group.attrs)
                        blank_incident_energy = attributes['instrument_mono_incident_energy'] # keV
                        blank_ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
                        blank_dwell_time = attributes['param_dwell'] # seconds
                    else:
                        print(f"Group '{group_name}' not found in the HDF5 file.")
                
                # normalize data by ion_chmaber_data(i0)
                blank_data = blank_data/blank_ion_chamber_data[:,:,np.newaxis]
                blank_data = black_data/blank_dwell_time
    
                # Use incident X-ray energy to define energy range of interest 
                # incident_wavelength = 1.2398e-9/incident_energy # convert incident energy to wavelength (hc/lambda)
                # compton_wavelength = 4.86e-12 + incident_wavelength # determine compton wavelength using maximum wavelength differential plus incident 
                # max_energy = 1.2398e-9/compton_wavelength  # convert compton wavelength to energy and set the maximum energy to about the compton peak 
                min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
                max_idx = min([i for i, v in enumerate(energy) if v >= incident_energy])
    
    
                # Total average spectrum
                avg_blank_data = np.mean(blank_data, axis = (0,1)) 
                avg_blank_data = avg_blank_data[min_idx:max_idx]
                
                # Background subtracted AOI
                baseline = arpls(avg_blank_data) # Baseline of AOI spectrum
                AOI_bkg_sub = AOI - avg_blank_data
                AOI_bkg_sub[AOI_bkg_sub <= 0] = 0
    
                # add baseline to AOI spectrum
                AOI_bkg_sub = AOI_bkg_sub + baseline
            
    
    


    ########## Find peaks in data using parameter thresholds ##########
    prom = 0.0001
    tall = 0.0001
    dist = 10
    if denoise:
        y_smoothed = np.exp(denoise_and_smooth_data(energy_int, np.log(AOI_bkg_sub)))
    else:
        y_smoothed = AOI_bkg_sub
        
    peaks, properties = find_peaks(y_smoothed, prominence = prom, height = tall, distance = dist)

   
    
    labels = []
    for i in range(len(peaks)):
        labels.extend(['Peak '+str(i+1)])
    
    ########## Spectra Plotting ##########
    # Plot raw AOI Spectrum
    fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectrum'), layout_xaxis_range = [min_energy,max_energy])
    fig1.update_layout(title = 'raw AOI Spectrum for '+filename[-26:-13],
                       width = 1600,
                       height = 800,
                       font = dict(size = 20))
    
    # Plot Background subtracted AOI spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
    
    # Plot total averaged spectrum 
    fig1.add_trace(go.Scatter(x = energy_int, y = avg_data, mode = 'lines', name = 'Avg Spectrum'))
    
    if 'avg_blank_data' in vars():
        # Plot avg blank spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = avg_blank_data, mode = 'lines', name = 'Avg Blank Spectrum'))

        # Plot baseline spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))


    if 'background' in vars():
        # Plot background spectrum        
        fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectrum'))
        # Plot baseline spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))
    
    # Plot smoothed spectrum 
    fig1.add_trace(go.Scatter(x = energy_int, y = y_smoothed, mode = 'lines', name = 'Smoothed Spectrum'))
    
    # Plot points identified as peaks
    fig1.add_trace(go.Scatter(x = energy_int[peaks], y = AOI_bkg_sub[peaks], mode = 'markers+text', name = 'Peak fit', text = labels))
    

    
    # Plot formatting
    fig1.update_yaxes(title_text = 'Intensity (counts/s)', type = 'log', exponentformat = 'e')
    fig1.update_xaxes(title_text = 'Energy (keV)')
    fig1.update_traces(line={'width': 5})
    fig1.show()

    
    ########## Adjusting peak finding as needed ##########
    peak_props = input('Change peak thresholds for prominence, height, and/or distance (Yes or No)?')
    while True:
        if peak_props.lower() == 'no':
            ########## Identify elements ##########
            # identify fluorescent line energy that most closely matches the determined peaks
            tolerance = 1.25 # allowed difference in percent
            elements = background_elements + sample_elements
            matched_peaks, _ = identify_element_match(elements, energy_int[peaks]*1000, tolerance, incident_energy*1000)
            # Plotting vertical lines for matched peaks and labeled with element symbol
            for i in range(len(matched_peaks)):
                fig1.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])
            fig1.show()
            break
        if peak_props.lower() == 'yes':
            user_input = input("Enter new values for prominence (" + str(prom) + "), height(" + str(tall) + "), and distance(" + str(dist) + ") (comma-separated), 'no' to exit: ")
            if user_input.lower() == 'no':
                break
                
            try: 
                prom, tall, dist = map(float, user_input.split(','))
            except ValueError:
                print("Invalid input. Please enter integers separated by a comma or 'no' to exit.")
                continue
            
            # Find peaks in data
            peaks, properties = find_peaks(y_smoothed, prominence = prom, height = tall, distance = dist)
            
            # Label peaks
            labels = []
            for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])
            
            # Creating new figure for AOI Spectrum
            fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectrum'), layout_xaxis_range = [min_energy,max_energy])
            fig1.update_layout(title = 'AOI Spectrum for '+filename[-26:-13],
                               width = 1600,
                               height = 800,
                               font = dict(size = 20))
            
            # Plot Background subtracted AOI spectrum
            fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
                        
            # Plot total avg spectrum 
            fig1.add_trace(go.Scatter(x = energy_int, y = avg_data, mode = 'lines', name = 'Avg Spectrum'))
            
            if 'avg_blank_data' in vars():
                # Plot avg blank spectrum
                fig1.add_trace(go.Scatter(x = energy_int, y = avg_blank_data, mode = 'lines', name = 'Avg Blank Spectrum'))

                # Plot baseline spectrum
                fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))
            
            if 'background' in vars():
                # Plot background spectrum
                fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectrum'))

                # Plot baseline spectrum
                fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))
            
            # Plot smoothed spectrum 
            fig1.add_trace(go.Scatter(x = energy_int, y = y_smoothed, mode = 'lines', name = 'Smoothed Spectrum'))
            
            
            # Plot points identified as peaks
            fig1.add_trace(go.Scatter(x = energy_int[peaks], y = AOI_bkg_sub[peaks],mode = 'markers+text', name = 'Peak fit', text = labels))
            
           

            # Plot formatting
            fig1.update_yaxes(title_text = 'Intensity (counts/s)', type = 'log', exponentformat = 'e')
            fig1.update_xaxes(title_text = 'Energy (keV)')
            fig1.update_traces(line={'width': 5})

            ########## Identify elements ##########
            # identify fluorescent line energy that most closely matches the determined peaks
            tolerance = 1.25 # allowed difference in percent
            elements = background_elements + sample_elements
            matched_peaks, _ = identify_element_match(elements, energy_int[peaks]*1000, tolerance, incident_energy*1000)
            # Plotting vertical lines for matched peaks and labeled with element symbol
            for i in range(len(matched_peaks)):
                fig1.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])
            fig1.show()

    ## Remove peaks resulting from overfitting/noise in data 
    remove_peaks_q = input("Remove any peaks from consideration (yes or no)?")
    if remove_peaks_q.lower() == 'yes':
        user_input = input('Input comma-seperated list of peaks to be removed:')
        error_peaks_idx = [int(num)-1 for num in user_input.split(',')]
        # Create a boolean mask to select peaks to keep
        mask = np.ones(peaks.shape, dtype=bool)
        mask[error_peaks_idx] = False
        
        # Filter the array using the mask
        peaks = peaks[mask]
                        
            
    
    ########## Fit spectra and plot results ##########
    print('Beginning peak fitting')
    peak_fit, bkg_fit, peak_fit_params,r_squared = peak_fitting(energy_int, AOI_bkg_sub, peaks, dist)
    print('Peak fit r-squared value is:', r_squared)
    # # Find peaks in fitted data
    # peaks, properties = find_peaks(peak_fit-bkg_fit)
    
    # # Label peaks
    # labels = []
    # for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])
    

    ########## Final Plot ##########
    fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectrum'), layout_xaxis_range = [min_energy,max_energy])
    fig1.update_layout(title = 'AOI Spectrum for '+filename[-26:-13],
                       width = 1600,
                       height = 800,
                       font = dict(size = 20))
    
    # Plot Background subtracted AOI spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
                
    # Plot total avg spectrum 
    fig1.add_trace(go.Scatter(x = energy_int, y = avg_data, mode = 'lines', name = 'Avg Spectrum'))
    
    if 'avg_blank_data' in vars():
        # Plot avg blank spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = avg_blank_data, mode = 'lines', name = 'Avg Blank Spectrum'))

        # Plot baseline spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))
                
    if 'background' in vars():
        # Plot background spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectrum'))

        # Plot baseline spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))

    # Plot peak and background fits
    fig1.add_trace(go.Scatter(x = energy_int, y = peak_fit, mode = 'lines', name ='AOI Spectrum Fit'))
    fig1.add_trace(go.Scatter(x = energy_int, y = bkg_fit, mode = 'lines', name = 'AOI Spectrum Bkg Fit'))

    # Plot smoothed spectrum 
    fig1.add_trace(go.Scatter(x = energy_int, y = y_smoothed, mode = 'lines', name = 'Smoothed Spectrum'))

    # Plot points identified as peaks
    fig1.add_trace(go.Scatter(x = energy_int[peaks], y = AOI_bkg_sub[peaks],mode = 'markers+text', name = 'Peak fit', text = labels))


    
    # Plot formatting
    fig1.update_yaxes(title_text = 'Intensity (counts/s)', type = 'log', exponentformat = 'e')
    fig1.update_xaxes(title_text = 'Energy (keV)')
    fig1.update_traces(line={'width': 5})

    ########## Identify elements ##########
    # identify fluorescent line energy that most closely matches the determined peaks
    tolerance = 1.25 # allowed difference in percent
    matched_peaks, _ = identify_element_match(elements, energy_int[peaks]*1000, tolerance, incident_energy*1000)
    # Plotting vertical lines for matched peaks and labeled with element symbol
    for i in range(len(matched_peaks)):
        fig1.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])

    # show figure
    fig1.show()


    
    
    ########## XRF Map Plotting ##########
    # Defing energy range of interest from user input
    while True:
        user_input = input(str(len(peaks))+' peaks found. ''How many energy range(s) should be plotted in 2D? Enter 0 to exit')
        try: 
            ranges = int(user_input)
            break
        except ValueError:
            print("Invalid input. Please enter a single integer greater than 0 or enter '0' to exit.")
            continue

        
    energy_ranges = []
    for i in range(ranges):
        energy_range_str = input('Energy (keV*100) range ' + str(i+1) + ' to be plotted in 2D? (min:max+1)')
        energy_ranges.append(slice( *map(int, energy_range_str.split(':'))))

        # Plot 2D map of AOI 
        d_element = AOI_data[:, :, energy_ranges[i]]
        element_data = np.sum(d_element, axis=(2))

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        
        ax.set_title('Energies in range: ' + str(round(energy[energy_ranges[i]].min(),3)) + '-' + str(round(energy[energy_ranges[i]].max(),3)) + ' keV')
        if not Sigray:
            ax.set_ylabel("y ($\mu$m)")
            ax.set_xlabel("x ($\mu$m)")
        elif Sigray:
            ax.set_ylabel("y (mm)")
            ax.set_xlabel("x (mm)")

        im = ax.imshow(element_data, extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap='viridis')
        
        
        fig.colorbar(im, cax=cax, orientation = 'vertical')
        plt.show()
    
    # Making RGB figure
    if ranges == 3:
        element_1 = normalize(AOI_data[:, :, energy_ranges[0]])
        element_2 = normalize(AOI_data[:, :, energy_ranges[1]])
        element_3 = normalize(AOI_data[:, :, energy_ranges[2]])

        

        fig, ax = plt.subplots()

        # Plot each element with a different color map and alpha blending
        cmap1 = plt.get_cmap('Reds')
        cmap2 = plt.get_cmap('Blues')
        cmap3 = plt.get_cmap('Greens')

        # Plot each heatmap with a specific color and some transparency
        heatmap1 = ax.imshow(np.sum(element_1, axis = 2), extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap=cmap1)
        heatmap2 = ax.imshow(np.sum(element_2, axis = 2), extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap=cmap2)
        heatmap3 = ax.imshow(np.sum(element_3, axis = 2), extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap=cmap3)

        element_1_label = input('Enter symbol for element 1:')
        element_2_label = input("Enter symbol for element 2:")
        element_3_label = input("Enter symbol for element 3:")

        # Create a legend indicating the color associated with each dataset
        red_patch = mpatches.Patch(color='red', label= element_1_label + ' (Red)')
        blue_patch = mpatches.Patch(color='blue', label= element_2_label + ' (Blue)')
        green_patch = mpatches.Patch(color='green', label= element_3_label + ' (Green)')
        
        # Place the legend outside the plot
        plt.legend(handles=[red_patch, blue_patch, green_patch], loc='center left', bbox_to_anchor=(1, 0.5))
        
        if not Sigray:
            ax.set_ylabel("y ($\mu$m)")
            ax.set_xlabel("x ($\mu$m)")
        if Sigray:
            ax.set_ylabel("y (mm)")
            ax.set_xlabel("x (mm)")

        plt.show()       
        

    return detector_2D_map_fig, fig1



########## AOI Extraction ##########
## Function to Extract Detector Area of Interest Spectrum using previously determined parameters
# * Inputs
#     1. filename: file path to hdf5 (.h5) file containing total scan data
#     2. min_energy: minimum of energy range of interest (keV)
#     3. elements: anticipated elements contained in the sample from background analysis using pyXRF
#     4. AOI_x, AOI_y, BKG_x, BKG_y: x and y ranges of the particle of interest and equally sized background region
#        determined by the analyzing the file with AOI_particle_analysis
#     5. prom, height, dist: peak parameters determined by the analyzing the file with AOI_particle_analysis
#     6. bad_pixels: number of bad/dead pixels in the detector image
#     7. error_peaks: peaks to be ignored in the peak fitting that result from noise in the data. Adjusting the peak parameters
#        to remove these pekas results in loss of peak(s) of interest        
# * Outputs detector_data, fig1, peak_fit_params, x_pos, y_pos, matched_peaks
#     1. detector_data: data used to make detector plot
#     2. fig1: HTML figure containing all relevant data/information processed that
#        can be used later to plot the results        
#     3. peak_fit_params: parameters used to define the gaussian fit of peaks in background subtracted partilce spectrum
#     4. x_pos, y_pos: x and y position of the detector image location based on the sample stage
#     5. matched_peaks: peaks matched to an element known to be present
def AOI_extractor(filename, min_energy, elements, AOI_x, AOI_y, BKG_x, BKG_y, prom, height, dist, bad_pixels, error_peaks, blank_file = None, denoise = True, Sigray = False):
    ########## Load data filenin variable ##########
    with h5py.File(filename, 'r') as file:
        data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        if not Sigray:
            ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
            if not Sigray:
                ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
                dwell_time = attributes['param_dwell'] # seconds
                
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")
    
    # Position axes
    x_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),data.shape[1])
    y_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),data.shape[0])
    if Sigray:
        y_pos = y_pos[::-1]
    
    # normalize data by ion_chmaber_data(i0)
    if not Sigray:
        data = data/ion_chamber_data[:,:,np.newaxis]
        data = data/dwell_time
   

    ########## Use incident X-ray energy to define energy range of interest ##########
    # incident_wavelength = 1.2398e-9/incident_energy # convert incident energy to wavelength (hc/lambda)
    # compton_wavelength = 4.86e-12 + incident_wavelength # determine compton wavelength using maximum wavelength differential plus incident 
    # max_energy = 1.2398e-9/compton_wavelength  # convert compton wavelength to energy and set the maximum energy to about the compton peak 
    max_energy = incident_energy
    energy = 0.01*np.arange(data.shape[2])
    min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
    max_idx = min([i for i, v in enumerate(energy) if v >= max_energy])
    


    ########## Detector data ##########
    max_int = np.max(data, axis = 2, keepdims = True)
    detector_data = np.sum(max_int,axis = (2))
    
    # set scaling to be robust
    vmin = np.quantile(detector_data,0.02)
    vmax = np.quantile(detector_data, 1- 0.02)
    
    detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, zmin = vmin, zmax = vmax, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
    detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                      title_x = 0.5,
                                      width = 500,
                                      height = 500,
                                      font = dict(size = 20),
                                      xaxis = dict(title = 'X-axis'),
                                      yaxis = dict(title = 'Y-axis'))
    if Sigray:
        detector_2D_map_fig.update_layout(yaxis = dict(autorange = 'reversed'))
    detector_2D_map_fig.show()

    if bad_pixels > 0:
        k = bad_pixels # number of values to be extracted 
        idx_flat = np.argpartition(detector_data.flatten(),k)[:k] # index of k lowest values 
        idx_2d = np.unravel_index(idx_flat,detector_data.shape)
        detector_data[idx_2d] = np.mean(detector_data) # new detecotr data without dead pixels 
        
         # set scaling to be robust
        vmin = np.quantile(detector_data,0.02)
        vmax = np.quantile(detector_data, 1- 0.02)

        # plot new data
        detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, zmin = vmin, zmax = vmax, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
        detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                          title_x = 0.5,
                                          width = 500,
                                          height = 500,
                                          font = dict(size = 20),
                                          xaxis = dict(title = 'X-axis'),
                                          yaxis = dict(title = 'Y-axis'))
        if Sigray:
            detector_2D_map_fig.update_layout(yaxis = dict(autorange = 'reversed'))
        
        detector_2D_map_fig.show()
        
    
    
    
    ########## Total average spectrum ##########
    avg_data = np.mean(data, axis = (0,1))
    avg_data = avg_data[min_idx:max_idx]
    
    
    ######### Setting area of interest ##########
    AOI_x = slice(AOI_x.start+1,AOI_x.stop+1)
    AOI_data = data[AOI_y, AOI_x, :]
    
    
    # Avg spectrum in selected area
    AOI = np.mean(AOI_data, axis=(0,1))
    AOI = AOI[min_idx:max_idx]
    energy_int = energy[min_idx:max_idx]
    
    
    ########## Position axes ##########

    # AOI positions
    y_int = y_pos[AOI_y]
    x_int = x_pos[AOI_x]
    
    if BKG_x:
        ######### Setting background area ##########
        # identify background spectrum
        BKG_x = slice(BKG_x.start+1, BKG_x.stop+1)
        bkg_data = data[BKG_y, BKG_x, :]
        
        # Avg background spectrum in selected area
        background = np.mean(bkg_data, axis=(0,1))
        background = background[min_idx:max_idx]
        

        # Background subtracted AOI
        baseline = arpls(background) # Baseline of AOI spectrum
        AOI_bkg_sub = AOI - background
        AOI_bkg_sub[AOI_bkg_sub <= 0] = 0
        


        # add baseline to AOI spectrum
        AOI_bkg_sub = AOI_bkg_sub + baseline
    else:
        blank_filename = blank_file
        if blank_filename:
            with h5py.File(blank_filename, 'r') as file:
                blank_data = file['xrfmap/detsum/counts'][:]
                blank_ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
                group_name = 'xrfmap/scan_metadata'
                if group_name in file:
                    group = file[group_name]
                    attributes = dict(group.attrs)
                    blank_incident_energy = attributes['instrument_mono_incident_energy'] # keV
                    blank_ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
                    blank_dwell_time = attributes['param_dwell'] #seconds
                else:
                    print(f"Group '{group_name}' not found in the HDF5 file.")
            
            # normalize data by ion_chmaber_data(i0)
            blank_data = blank_data/blank_ion_chamber_data[:,:,np.newaxis]
            blank_data = blank_data/blank_dwell_time

            # Use incident X-ray energy to define energy range of interest 
            # incident_wavelength = 1.2398e-9/incident_energy # convert incident energy to wavelength (hc/lambda)
            # compton_wavelength = 4.86e-12 + incident_wavelength # determine compton wavelength using maximum wavelength differential plus incident 
            # max_energy = 1.2398e-9/compton_wavelength  # convert compton wavelength to energy and set the maximum energy to about the compton peak 
            min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
            max_idx = min([i for i, v in enumerate(energy) if v >= incident_energy])


            # Total average spectrum
            avg_blank_data = np.mean(blank_data, axis = (0,1)) 
            avg_blank_data = avg_blank_data[min_idx:max_idx]
            
            # Background subtracted AOI
            baseline = arpls(avg_blank_data) # Baseline of AOI spectrum
            AOI_bkg_sub = AOI - avg_blank_data
            AOI_bkg_sub[AOI_bkg_sub <= 0] = 0

            # add baseline to AOI spectrum
            AOI_bkg_sub = AOI_bkg_sub + baseline
        # subtract the lowest N points as background
        elif self_consistent_bkg:
            AOI_shape = AOI_data.shape # get the shape of the area selected by the user above
            npoints = AOI_shape[0]* AOI_shape[1] # find the number of pixels/points in the area selected above
            
            # find the lowest intensities plotted in detector map
            lowest_intensities = np.argsort(detector_data.flatten())[:npoints] # sort the intensities to find the npoints lowest values
            low_int_idx = np.unravel_index(lowest_intensities, detector_data.shape) # extract 2d indices of lowest intensity

            bkg_data = data[low_int_idx[1][:,np.newaxis], low_int_idx[0][:,np.newaxis], :] # extracting bkg_Data based on lowest intensities
           
            # Avg background spectrum in selected area
            background = np.mean(bkg_data, axis=(0,1))
            background = background[min_idx:max_idx]       

            
            # Background subtracted AOI
            baseline = arpls(background) # Baseline of AOI spectrum
            AOI_bkg_sub = AOI - background
            AOI_bkg_sub[AOI_bkg_sub <= 0] = 0

            # add baseline to AOI spectrum
            AOI_bkg_sub = AOI_bkg_sub + baseline
            
        else: 
            baseline = arpls(AOI)
            AOI_bkg_sub = AOI - baseline
           
    

    ########## Find peaks in data using parameter thresholds ##########
    if denoise:
        y_smoothed = np.exp(denoise_and_smooth_data(energy_int, np.log(AOI_bkg_sub)))
    else:
        y_smoothed = AOI_bkg_sub
    
    peaks, properties = find_peaks(y_smoothed, prominence = prom, height = height, distance = dist)
    # Label peaks
    labels = []
    for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])   
    
        
       
    ## Remove peaks resulting from overfitting/noise in data 
    if error_peaks:
        error_peaks_idx = [num - 1 for num in error_peaks]
        # Create a boolean mask to select peaks to keep
        mask = np.ones(peaks.shape, dtype=bool)
        mask[error_peaks_idx] = False        
        # Filter the array using the mask
        peaks = peaks[mask]
                        
    
    ########## Fit spectra and plot results ##########
    print('Beginning peak fitting')
    peak_fit, bkg_fit, peak_fit_params, r_squared = peak_fitting(energy_int, AOI_bkg_sub, peaks, dist)
    print('Peak fit r-squared value is:', r_squared)
    # # Find peaks in fitted data
    # peaks, properties = find_peaks(peak_fit-bkg_fit)
    
    # # Label peaks
    # labels = []
    # for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])
    
       


    ########## Final Plot ##########
    fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectrum'), layout_xaxis_range = [min_energy,max_energy])
    fig1.update_layout(title = 'AOI Spectrum for '+filename[-26:-13],
                       width = 1600,
                       height = 800,
                       font = dict(size = 20))
    
    # Plot Background subtracted AOI spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
                
    # Plot total avg spectrum 
    fig1.add_trace(go.Scatter(x = energy_int, y = avg_data, mode = 'lines', name = 'Avg Spectrum'))
    
    if 'avg_blank_data' in vars():
        # Plot avg blank spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = avg_blank_data, mode = 'lines', name = 'Avg Blank Spectrum'))

        # Plot baseline spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))
    
    if 'background' in vars():
        # Plot background spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectrum'))

        # Plot baseline spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectrum'))

    # Plot peak and background fits
    fig1.add_trace(go.Scatter(x = energy_int, y = peak_fit, mode = 'lines', name ='AOI Spectrum Fit'))
    fig1.add_trace(go.Scatter(x = energy_int, y = bkg_fit, mode = 'lines', name = 'AOI Spectrum Bkg Fit'))

    # # Plot points identified as peaks
    # fig1.add_trace(go.Scatter(x = energy_int[peaks], y = peak_fit[peaks],mode = 'markers+text', name = 'Peak fit', text = labels))


    # Plot formatting
    fig1.update_yaxes(title_text = 'Intensity (counts/s)', type = 'log', exponentformat = 'e')
    fig1.update_xaxes(title_text = 'Energy (keV)')
    fig1.update_traces(line={'width': 5})

    ########## Identify elements ##########
    # identify fluorescent line energy that most closely matches the determined peaks
    tolerance = 1.25 # allowed difference in percent
    matched_peaks, _ = identify_element_match(elements, energy_int[peaks]*1000, tolerance, incident_energy*1000)
    # Plotting vertical lines for matched peaks and labeled with element symbol
    for i in range(len(matched_peaks)):
        fig1.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])

    # show figure
    fig1.show()


    
    
    ########## XRF Map Plotting ##########
    # Defing energy range of interest from user input
    while True:
        user_input = input(str(len(peaks))+' peaks found. ''How many energy range(s) should be plotted in 2D? Enter 0 to exit')
        try: 
            ranges = int(user_input)
            break
        except ValueError:
            print("Invalid input. Please enter a single integer greater than 0 or enter '0' to exit.")
            continue

        
    energy_ranges = []
    for i in range(ranges):
        energy_range_str = input('Energy (keV*100) range ' + str(i+1) + ' to be plotted in 2D? (min:max+1)')
        energy_ranges.append(slice( *map(int, energy_range_str.split(':'))))

        # Plot 2D map of AOI 
        d_element = AOI_data[:, :, energy_ranges[i]]
        element_data = np.sum(d_element, axis=(2))

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        
        ax.set_title('Energies in range: ' + str(round(energy[energy_ranges[i]].min(),3)) + '-' + str(round(energy[energy_ranges[i]].max(),3)) + ' keV')
        if not Sigray:
            ax.set_ylabel("y ($\mu$m)")
            ax.set_xlabel("x ($\mu$m)")
        if Sigray:
            ax.set_ylabel("y (mm)")
            ax.set_xlabel("x (mm)")
            
        im = ax.imshow(element_data, extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap='viridis')
        
        
        fig.colorbar(im, cax=cax, orientation = 'vertical')
        plt.show()
    
    # Making RGB figure
    if ranges == 3:
        element_1 = normalize(AOI_data[:, :, energy_ranges[0]])
        element_2 = normalize(AOI_data[:, :, energy_ranges[1]])
        element_3 = normalize(AOI_data[:, :, energy_ranges[2]])

        

        fig, ax = plt.subplots()

        # Plot each element with a different color map and alpha blending
        cmap1 = plt.get_cmap('Reds')
        cmap2 = plt.get_cmap('Blues')
        cmap3 = plt.get_cmap('Greens')

        # Plot each heatmap with a specific color and some transparency
        heatmap1 = ax.imshow(np.sum(element_1, axis = 2), extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap=cmap1)
        heatmap2 = ax.imshow(np.sum(element_2, axis = 2), extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap=cmap2)
        heatmap3 = ax.imshow(np.sum(element_3, axis = 2), extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap=cmap3)

        element_1_label = input('Enter symbol for element 1:')
        element_2_label = input("Enter symbol for element 2:")
        element_3_label = input("Enter symbol for element 3:")

        # Create a legend indicating the color associated with each dataset
        red_patch = mpatches.Patch(color='red', label= element_1_label + ' (Red)')
        blue_patch = mpatches.Patch(color='blue', label= element_2_label + ' (Blue)')
        green_patch = mpatches.Patch(color='green', label= element_3_label + ' (Green)')
        
        # Place the legend outside the plot
        plt.legend(handles=[red_patch, blue_patch, green_patch], loc='center left', bbox_to_anchor=(1, 0.5))
        
        if not Sigray:
            ax.set_ylabel("y ($\mu$m)")
            ax.set_xlabel("x ($\mu$m)")
        elif Sigray:
            ax.set_ylabel("y (mm)")
            ax.set_xlabel("x (mm)")
            
        plt.show()

   
   
    return detector_data, fig1, x_pos, y_pos, matched_peaks, peak_fit_params




########## Extract detector image data of selected file ##########
# * Inputs
#     1. filename: file path to hdf5 (.h5) file containing total scan data
# * Outputs
#     1. detector_data
#     2. x_pos, y_pos: x and y positions extracted from hdf5 file
def extract_detector_data(filename, Sigray = False):
    ########## Load data file in variable ##########
    with h5py.File(filename, 'r') as file:
        data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
            if not Sigray:
                ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
                dwell_time = file['param_dwell'] # seconds
                
                
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")

    
    ########## Use incident X-ray energy to define energy range of interest ##########
    # incident_wavelength = 1.2398e-9/incident_energy # convert incident energy to wavelength (hc/lambda)
    # compton_wavelength = 4.86e-12 + incident_wavelength # determine compton wavelength using maximum wavelength differential plus incident 
    # max_energy = 1.2398e-9/compton_wavelength  # convert compton wavelength to energy and set the maximum energy to about the compton peak 
    max_energy = incident_energy
    energy = 0.01*np.arange(data.shape[2])
    min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
    max_idx = min([i for i, v in enumerate(energy) if v >= max_energy])

    ########## Position axes ##########
    # whole positions
    x_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),data.shape[1])
    y_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),data.shape[0])
    if Sigray:
        y_pos = y_pos[::-1]
   
    # normalize data by ion_chmaber_data(i0)
    if not Sigray:    
        data = data/ion_chamber_data[:,:,np.newaxis]
        data = data/dwell_time

    
    ########## Detector data ##########
    max_int = np.max(data, axis = 2, keepdims = True)
    detector_data = np.sum(max_int,axis = (2))
    detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
    detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                      title_x = 0.5,
                                      width = 500,
                                      height = 500,
                                      font = dict(size = 20),
                                      xaxis = dict(title = 'X-axis'),
                                      yaxis = dict(title = 'Y-axis'))
    if Sigray:
        detector_2D_map_fig.update_layout(yaxis = dict(autorange= 'reversed'))
    detector_2D_map_fig.show()
    
    ########## Handling bad pixels ##########
    user_input = input("Smooth over bad pixels? (Yes or No):")
    if user_input.lower() == "yes":
        # get number of values to extract
        user_input = input("Input integer value for number of bad pixels based on number unique xy coordinates showing distinctly lower intensity:")
        k = int(user_input) # number of values to be extracted 
        idx_flat = np.argpartition(detector_data.flatten(),k)[:k] # index of k lowest values 
        idx_2d = np.unravel_index(idx_flat,detector_data.shape)
        detector_data[idx_2d] = np.mean(detector_data) # new detecotr data without dead pixels 
    
        # plot new data
        detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
        detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                          title_x = 0.5,
                                          width = 500,
                                          height = 500,
                                          font = dict(size = 20),
                                          xaxis = dict(title = 'X-axis'),
                                          yaxis = dict(title = 'Y-axis'))
        if Sigray:
            detector_2D_map_fig.update_layout(yaxis = dict(autorange= 'reversed'))
        detector_2D_map_fig.show()
    
    return detector_data, x_pos, y_pos

########## Function to Extract information from XRF scan of Standard ##########
# * **Inputs**
#   1. standard_filename: filepath to Micromatter standard scan of interest
#   2. background_filename: filepath to background scan of Mylar blank provided by Micromatter
#   3. open_air_filename: scan of the beamline with nothing in beampath
#   3. element: list of string element of interest contained on standard scan provided in 'standard_filename'
#   4. area_rho: area density of element of interest in units of micrograms per cm squared provided by Micromatter
#   5. scan_area: square area covered by the XRF scan in units of micron squared
     

# * **Outputs**
#   1. fig: plotly figure showing the data manipulation and contains all the data shwon in the figure
#   2. cal_eq: calibration equation for calculating the mass relative to intensity. 

def standard_data_extractor(standard_filename, background_filename, open_air_filename, element, area_rho, scan_area, min_energy):  
    ########## Load data filenin variable ##########
    with h5py.File(standard_filename, 'r') as file:
        standard_data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
            ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
            dwell_time = attributes['param_dwell'] # seconds
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")
    
    
    # Use incident X-ray energy to define energy range of interest 
    # incident_wavelength = 1.2398e-9/incident_energy # convert incident energy to wavelength (hc/lambda)
    # compton_wavelength = 4.86e-12 + incident_wavelength # determine compton wavelength using maximum wavelength differential plus incident 
    # max_energy = 1.2398e-9/compton_wavelength  # convert compton wavelength to energy and set the maximum energy to about the compton peak 
    max_energy = incident_energy
    energy = 0.01*np.arange(standard_data.shape[2])
    max_idx = min([i for i, v in enumerate(energy) if v >= max_energy])
    min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])    
    
    
    # Position axes
    x_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),standard_data.shape[1])
    y_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),standard_data.shape[0])
    
    # normalize data by ion_chmaber_data(i0)
    standard_data = standard_data/ion_chamber_data[:,:,np.newaxis]
    standard_data = standard_data/dwell_time
    
    # Total avg spectrum
    standard_avg_data = np.mean(standard_data, axis = (0,1))
    standard_avg_data = standard_avg_data[min_idx:max_idx]
    energy_int = energy[min_idx:max_idx]
    
    
    
    
    ########## extract background data ##########
    with h5py.File(background_filename, 'r') as file:
        background_data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
            ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
            bkg_dwell_time = attributes['param_dwell'] #seconds
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")
    
    # Position axes
    x_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),background_data.shape[1])
    y_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),background_data.shape[0])
    
    # normalize data by ion_chmaber_data(i0)
    background_data = background_data/ion_chamber_data[:,:,np.newaxis]
    background_data = background_data/bkg_dwell_time
    
    # Total avg spectrum
    background_avg_data = np.mean(background_data, axis = (0,1))
    background_avg_data = background_avg_data[min_idx:max_idx]
    
    
    
    
    ########## Subtract background from standard #########
    standard_data = standard_avg_data - background_avg_data
    standard_data[standard_data <= 0] = 0
    
    # define baseline of background and refining baseline by iterating arpls() function
    baseline = arpls(background_avg_data)
    std_data_plus_baseline = standard_data + baseline
    
    
    ########## Identify Peaks ##########
    # find peaks
    y_smoothed = np.exp(denoise_and_smooth_data(energy_int, np.log(std_data_plus_baseline)))
    peaks, _ = find_peaks(std_data_plus_baseline, distance = 10)
    
    # Label peaks
    labels = []
    for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])
    
    ########## Find element of interest ##########
    # identify fluorescent line energy that most closely matches the determined peaks
    tolerance = 1.25 # allowed difference in percent
    matched_peaks, _ = identify_element_match(element, energy_int[peaks]*1000, tolerance, incident_energy*1000)
    
    # find peak belonging to element of interest
    element_int_peaks_standard = [row for row in matched_peaks if row[1] == element[0]] 
    
    # remove all peaks except those belonging to element of interest
    element_peak_idx = [ID[0]-1 for ID in element_int_peaks_standard]
    peaks = peaks[element_peak_idx]
            
        
    ########## Plot the results to ensure they make sense ##########
    fig = go.Figure(data = go.Scatter(x = energy_int, y = standard_avg_data, mode = 'lines', name = 'Standard Spectrum'), layout_xaxis_range = [min_energy,max_energy])
    fig.update_layout(title = 'Spectrum for ' + element[0] + ' Standard ' + '(' + standard_filename[-26:-13] + ')',
                       width = 1600,
                       height = 800,
                       font = dict(size = 20))
    
    # Plot Background spectrum
    fig.add_trace(go.Scatter(x = energy_int, y = background_avg_data, mode = 'lines', name = 'Background Spectrum'))
    
    # Plot bkg subtracted standard spectrum 
    fig.add_trace(go.Scatter(x = energy_int, y = standard_data, mode = 'lines', name = 'Background subtracted Standard Spectrum'))
    
    # plot baseline 
    fig.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline'))
    
    # plot standard + baseline
    fig.add_trace(go.Scatter(x = energy_int, y = std_data_plus_baseline, mode = 'lines', name = 'Bkg subtracted Standard + Baseline'))
    
    
    # plot peaks
    fig.add_trace(go.Scatter(x = energy_int[peaks], y = std_data_plus_baseline[peaks], mode = 'markers+text', name = 'peak fit', text = labels))
    
    # Plotting vertical lines for matched peaks and labeled with element symbol
    for i in range(len(matched_peaks)):
        fig.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])
    
    
    # Plot formatting
    fig.update_yaxes(title_text = 'Intensity (counts)', type = 'log', exponentformat = 'e')
    fig.update_xaxes(title_text = 'Energy (keV)')
    fig.update_traces(line={'width': 5})
    
    fig.show()
        
        
    
        
    ########## Making Calibration curve ##########
    # extracting peak idx 
    user_input = input('Input comma seperated peaks of interest (i.e. peaks that clearly align with element of interest fluorescence lines and are present in sample).')
    peaks_int = list(map(int,user_input.split(',')))
    peak_int_idx = [x-1 for x in peaks_int]

    # determining open-air contributions from empty beampath 
    with h5py.File(open_air_filename, 'r') as file:
        open_air_data = file['xrfmap/detsum/counts'][:]
        air_pos_data = file['xrfmap/positions/pos'][:]
        air_ion_chamber_data = file['xrfmap/scalers/val'][:,:,0]
        air_dwell_time = attributes['param_dwell'] #seconds
        

     # normalize data by ion_chmaber_data(i0)
    open_air_data = open_air_data/air_ion_chamber_data[:,:,np.newaxis]
    open_air_data = open_air_data/air_dwell_time

    open_air_avg_data = np.mean(open_air_data, axis = (0,1))
    open_air_avg_data = open_air_avg_data[min_idx:max_idx]

    ## Fit gaussian to region corresponding to element of interest
    params = peak_fitting(energy_int,open_air_avg_data, peaks[peak_int_idx],10)[2]
    standard_element_integral_params = peak_fitting(energy_int,std_data_plus_baseline, peaks[peak_int_idx], 10)[2]
    
    std_amp = standard_element_integral_params[::3]
    std_stddev = standard_element_integral_params[2::3]
    amp = params[::3]
    stddev = params[2::3]
    
    open_air_integral = np.zeros(len(peaks_int))
    std_integral = np.zeros(len(peaks_int))
    for i in range(len(peaks_int)):
        open_air_integral[i] = np.sqrt(2*np.pi) * amp[i] * stddev[i]
        std_integral[i] = np.sqrt(2*np.pi) * std_amp[i] * std_stddev[i]
        
    sum_open_air_integral = sum(open_air_integral)
    standard_element_integral_intensity = sum(std_integral) - sum_open_air_integral

    
    # Converting standard data
    scan_area = scan_area * 1e-8 # convert micron squared to cm squared
    element_mass = area_rho * scan_area * 1e6 # output in picograms
    
    print(element_mass,'pg of',element[0],'in area of standard captured')
    
    # determine calibration curve function
    cal_eq = np.poly1d(np.polyfit([sum_open_air_integral, standard_element_integral_intensity],[0, element_mass],1))
    
    # plotting calibration curve
    fig1, ax = plt.subplots()
    x = np.linspace(0,standard_element_integral_intensity)
    
    ax.plot(x, cal_eq(x))
    plt.xlabel('Integral Intensity (counts)', fontsize = 16)
    plt.ylabel('Mass (pg)', fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.title('Quantitative Calibration Curve \n for ' + element[0] + ' Standard', fontsize = 18)
    
    ########## Add calibration function to plot ##########
    # create a list with two empty handles (or more if needed)
    handles = [patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                     lw=0, alpha=0)] * 2
    
    # create the corresponding number of labels (= the text you want to display)
    labels = []
    labels.append("Calibration Function:")
    labels.append(str(cal_eq))
    
    # create the legend, supressing the blank space of the empty line symbol and the
    # padding between symbol and label by setting handlelenght and handletextpad
    legend_properties = {'weight':'bold'}
    plt.legend(handles, labels, loc='best', fontsize=16, 
              fancybox=True, framealpha=0.7, 
              handlelength=0, handletextpad=0,
              prop = legend_properties)
    plt.show()
    
    return fig, cal_eq, sum_open_air_integral


def quantitative_analysis_curve(intensity_data, element_area_density_data, plot= True):
    # determine calibration curve function
    cal_eq = np.poly1d(np.polyfit(intensity_data,element_area_density_data,1))
    
    # plotting calibration curve
    x = np.linspace(0,intensity_data.max())
    y = cal_eq(x)
    
    # caculate relative fit 
    r2 = r2_score(element_area_density_data,cal_eq(intensity_data))

    if plot:
        # ax.scatter(intensity_data, element_mass_data, color = 'red', label = 'Standard data')
        # ax.plot(x, y, label = 'Line of Best Fit')
        # ax.legend(loc='lower right')
        plt.scatter(intensity_data, element_area_density_data, color = 'red', label = 'Standard data')
        plt.plot(x, y, label = 'Line of Best Fit')
        plt.legend(loc='lower right')
        plt.xlabel('Integral Intensity (counts/s)', fontsize = 16)
        plt.ylabel('Mass (pg)', fontsize = 16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ticklabel_format(axis='x', style='sci', scilimits = (0,0))
        
        plt.title('Quantitative Analysis Curve', fontsize = 18)
        
        # create textbox containing calibration function and R2 value
        plt.text(0.05, 0.95, 'Calibration Function:'+str(cal_eq)+'\n$R^2$='+"{:.6f}".format(r2),
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8, linewidth=2),
                 verticalalignment='top')
        plt.show()

    return cal_eq, r2

