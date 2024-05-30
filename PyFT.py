#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_dt(time, plot=False, verbose=0):
    """
    Calculate the average time step in a dataset, ignoring large gaps in time.

    Parameters
    ----------
    time : numpy array
        Array of time points.
    plot : bool, optional
        If True, plot a histogram of the time differences. Default is False.
    verbose : int, optional
        Verbosity level. If greater than 0, print the threshold and the average time step. Default is 0.

    Returns
    -------
    (float)    The average time step, ignoring large gaps as determined by the 80th percentile of the time differences.
    """
    # Create a pandas Series
    time_series = pd.Series(time)

    # Calculate time differences
    time_diff = time_series.diff()

    if plot == True:
        # Plot histogram of time differences
        plt.hist(time_diff.dropna(), bins=10, edgecolor='k', alpha=0.7)
        plt.xlabel('Time Difference')
        plt.ylabel('Frequency')
        plt.title('Histogram of Time Differences')
        plt.show()

    # Determine threshold using a percentile (e.g., 95th percentile)
    threshold = time_diff.quantile(0.80)

    # Filter out large gaps
    filtered_time_diff = time_diff[time_diff <= threshold]

    # Calculate the average time step, ignoring NaN values
    dt = filtered_time_diff.mean()

    if verbose > 0:
        print("Threshold for large gaps:", threshold)
        print("Average time step:", dt)
    
    return dt


def generate_data_with_gaps(dt=0.05, T_start=0, T_end=200, w=2, a0=3, phase=0, yshift=10, noise=False, noise_level=0.2, gap_multiplier=4, lambdas=2.5, verbose=0):
    """
    Generate mock data with gaps.

    Parameters
    ----------
    dt : float, optional.            Time step between data points. Default is 0.05.
    T_start : float, optional.       Start time of the generated data. Default is 0.
    T_end : float, optional.         End time of the generated data. Default is 200.
    w : float, optional.             Frequency of the cosine function. Default is 2.
    a0 : float, optional.            Amplitude of the cosine function. Default is 3.
    phase : float, optional.         Phase shift of the cosine function. Default is 0.
    yshift : float, optional.        Vertical shift of the signal. Default is 10.
    noise : bool, optional.          If True, add Gaussian noise to the signal. Default is False.
    noise_level : float, optional.   Standard deviation of the Gaussian noise. Default is 0.2.
    gap_multiplier : int, optional.  Multiplier for creating gaps in the data. Default is 4.
    lambdas : float, optional.       Desired number of wavelengths to display in the data. Default is 2.5.
    verbose : int, optional.         Verbosity level. If greater than 0, print the path where the mock dataset is saved. Default is 0.

    Returns
    -------
    (pandas.DataFrame)    DataFrame containing the generated data with gaps.
    """
    import numpy as np
    import pandas as pd
    
    # Generate time array
    time = np.arange(T_start, T_end, dt)
    if noise == True:
        signal = a0 * np.cos(w * time + phase) + yshift + np.random.normal(scale=noise_level, size=len(time))
    else:
        signal = a0 * np.cos(w * time + phase) + yshift
    
    # Generate masks
    n = int(lambdas * np.pi / dt) # Number of data points to display 'lambdas' number of wavelengths (nearest integer)
    ones = np.ones(n, dtype=bool)
    masks = []
    for i in range(gap_multiplier):
        nans = np.full((i * n,), np.nan)
        pattern = np.concatenate((ones, nans))
        mask = np.tile(pattern, int(np.ceil(len(time) / len(pattern))))
        mask = mask[:len(time)]
        masks.append(time * mask)
        masks.append(signal * mask)
    
    # Stack time and signal arrays horizontally
    data = np.column_stack(masks)
    
    # Generate header dynamically
    header = ','.join([f'time_{i+1},signal_{i+1}' for i in range(len(masks) // 2)])
    
    # Save as CSV
    np.savetxt('mockdata_gaps.csv', data, delimiter=',', header=header, comments='')
    
    if verbose > 0:
        print('Created mock dataset at /mockdata_gaps.csv')
        
def generate_data(dt=0.05, T_start=0, T_end=100, noise_level_1=0.2, noise_level_2=0.2*10, verbose=0):
    """
    Generate random data with noise.

    Parameters
    ----------
    dt : float, optional. Time step between data points. Default is 0.05.
    T : float, optional.  End time of the generated data. Default is 100.
    noise_level_1 : float, optional.  Standard deviation of the Gaussian noise for signal_1. Default is 0.2.
    noise_level_2 : float, optional.  Standard deviation of the Gaussian noise for signal_2. Default is 2.0.

    Returns
    -------
    numpy.ndarray
        Array containing the generated data.

    Notes
    -----
    - The function generates random data with noise following a specific trend.
    - Two signals (signal_1 and signal_2) are generated, each with its own noise level.
    - Signal_1 follows the trend `a0*cos(w*t) + 8*cos(5*t)` with noise added.
    - Signal_2 follows the trend `a0*cos(w*t)` with noise added.
    - The generated data is returned as a numpy array.
    """
    
    # Generate time array
    time = np.arange(T_start, T_end, dt)

    # Parameters
    w = 2  # Frequency of the cosine function
    a0 = 3  # Amplitude of the cosine function

    # Generate data following the trend cos(w*t) with random noise
    signal_1 = a0 * np.cos(w * time) + np.random.normal(scale=noise_level_1, size=len(time)) + 8 * np.cos(5 * time) + np.random.normal(scale=noise_level_1 * 8, size=len(time))
    signal_2 = a0 * np.cos(w * time) + np.random.normal(scale=noise_level_2, size=len(time)) + 10

    # Uncertainty
    unc = 0.4 * (max(signal_1) - 1) / (T_end-T_start) * time + 1

    # Stack time and signal arrays horizontally
    data = np.column_stack((time, signal_1, signal_2, unc))
    np.savetxt('mockdata.csv', data, delimiter=',', header='time,signal_1,signal_2,unc', comments='')

    if verbose > 0:
        print('Created mock dataset at /mockdata.csv')

def pythag(a,b):
    c = np.sqrt(a**2 + b**2)
    return c


# In[ ]:




