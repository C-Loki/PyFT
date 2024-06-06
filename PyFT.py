#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyFT as pyft
import xarray as xr
import cftime

def calc_dt(time, plot=False, verbose=0):
    """
    Calculate the average time step in a dataset, ignoring large gaps in time.

    Parameters
    ----------
    time : numpy array.       Array of time points.
    plot : bool, optional.    If True, plot a histogram of the time differences. Default is False.
    verbose : int, optional.  Verbosity level. If greater than 0, print the threshold and the average time step. Default is 0.

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

def moving_average(data, time=None, window_size=None):
    try:
        if window_size == None:
            window_size = int(len(time)/25)
    except NameError:
        raise NameError("If not providing a window_size, please make sure to provide time_data using the 'time' option.")
    return pd.Series(data).rolling(window=window_size).mean().tolist()

def calc_noise(time, data, window_size=None, noise_type=None, degree=5):
    """
    Parameters
    ----------
    time : array-like.         Array of time points.
    data : array-like.         Array of data points.
    window_size: int, optional. Size of the moving average window for smoothing the data. Default is None.
    noise_type (str, optional): Type of noise to calculate. If 'white' or 'w', calculates white noise level; otherwise, calculates noise level in chunks. Default is None.
    degree (int, optional):     Degree of polynomial for fitting when noise_type is not 'white' or 'w'. Default is 5.         

    Returns
    ----------
    - noise (array-like): Returns an array representing noise level over time.
    """
    data_smoothed = moving_average(data, time=time, window_size=window_size)
    residuals = data - data_smoothed
    
    if noise_type.lower() in ['white', 'w']:
        noise = np.std(residuals[~np.isnan(residuals)])
        noise = np.full(len(time),noise)
        return noise
    else:
        # Calculate the number of chunks of 20 data points
        chunck_wid = 20
        num_chunks = len(residuals) // chunck_wid

        # For each chunck, calculate the std dev of the residuals.
        std_devs = []
        for i in range(num_chunks):
            # Calculate the start and end indices of the current chunk
            start_idx = i * chunck_wid
            end_idx = start_idx + chunck_wid

            # Extract the data points for the current chunk
            chunk = residuals[start_idx:end_idx]

            # Calculate the standard deviation of the chunk and append to the list
            std_devs.append(np.std(chunk))

        std_devs = np.repeat(std_devs, chunck_wid)

        # Degree of the polynomial fit
        degree = degree

        # Perform polynomial fit
        coefficients = np.polyfit(time[~np.isnan(std_devs)], std_devs[~np.isnan(std_devs)], degree)
        polynomial = np.poly1d(coefficients)
        noise = polynomial(time)
        return noise    # returns a range of numbers

def pythag(a,b):
    """
    Pythagoras's Theorem. This is mainly a test function.
    """
    c = np.sqrt(a**2 + b**2)
    return c

def process_df(data_path, time_column, column_of_interest, unc_column=None, calc_unc=False, verbose=0, detrend_type='linear'):
    """
    Process data from a CSV or netCDF file.

    Parameters
    ----------
    data_path : str
        Path to the data file.
    time_column : str
        Name of the column containing time values.
    column_of_interest : str
        Name of the column containing signal values.
    unc_column : str, optional
        Name of the column containing uncertainties.
    calc_unc : bool, optional
        Whether to calculate uncertainties.
    verbose : int, optional
        Verbosity level.
    detrend_type : str, optional
        Type of detrending to apply.

    Returns
    ----------
    ds : xarray.Dataset
        Processed data in an xarray Dataset.
    """
    from scipy import signal
    
    # Split the filename and extension
    filename, extension = data_path.rsplit('.', 1)
    
    if extension not in ['csv', 'nc']:
        raise ValueError(f"Unsupported file extension: {extension}. Currently only .csv or .nc are supported.")

    if extension == 'csv':
        # Read the CSV file into a DataFrame
        data = pd.read_csv(data_path)

        # Extract time and signal data, dropping NaN values
        time = data[time_column].dropna()
        signal_data = data[column_of_interest].dropna()

        # Try to extract uncertainties, if not present, set them to 1
        if calc_unc:
            unc = pyft.calc_noise(time, signal_data, window_size=None, noise_type='w', degree=5)
        else:
            try:
                unc = data[unc_column]
            except KeyError:
                if verbose > 0:
                    print("No uncertainties detected. No weighting will be used.")
                unc = np.ones(len(signal_data))

        # Calculate weights based on uncertainties
        weights = 1 / unc ** 2

        # Remove linear trend from signal data
        signal = signal.detrend(signal_data, type=detrend_type)

        # Create the xarray Dataset
        ds = xr.Dataset(
            {
                'signal': (['time'], signal),
                'weights': (['time'], weights),
            },
            coords={
                'time': time
            }
        )

    elif extension == 'nc':
        # Open the netCDF file
        data = xr.open_dataset(data_path)
        data = data.rename({'tos': 'signal'})

        # Try to extract uncertainties, if not present, set them to 1
        if calc_unc:
            unc = pyft.calc_noise(data['time'], data['signal'], window_size=None, noise_type='w', degree=5)
        else:
            try:
                unc = data[unc_column]
            except KeyError:
                if verbose > 0:
                    print("No uncertainties detected. No weighting will be used.")
                unc = np.ones(len(data['signal']))

        # Calculate weights based on uncertainties
        weights = 1 / unc ** 2

        data['weights'] = weights

        if 'signal' in data.coords:
            data = data.reset_coords('signal', drop=True)

        # Detrend the 'signal' variable
        detrended_signal = signal.detrend(data['signal'], type=detrend_type)

        # Reassign dimensions when reassigning
        data['signal'] = (data['signal'].dims, detrended_signal)

        ds = data

        # Check if time values are numeric, if not, change them to be numeric
        time_dtype = ds['time'].dtype
        if not np.issubdtype(time_dtype, np.number):
            time_data = ds['time']
            start_time = ds['time'][0]

            year = start_time.dt.year.values
            month = start_time.dt.month.values
            day = start_time.dt.day.values
            hour = start_time.dt.hour.values 
            minute = start_time.dt.minute.values
            second = start_time.dt.second.values

            # Get the numerical representation
            time_num = cftime.date2num(time_data, units=f'common_years since {year}-{month}-{day} {hour}:{minute}:{second}', calendar='noleap')
            ds = ds.assign_coords(time=time_num)

    return ds


def conversion_factor(from_unit, to_unit):
    """
    Convert a time value from one unit to another.

    Parameters
    ----------
    value : float
        The numeric value to be converted.
    from_unit : str
        The unit of the input value.
    to_unit : str
        The unit to convert the input value to.

    Returns
    ----------
    converted_value : float
        The converted value in the target unit.
    """
    from collections import defaultdict
    
    # Define conversion factors for each unit to seconds
    conversion_factors_to_seconds = {
        'ns': 1 / 1e9,
        'ms': 1 / 1e3,
        's': 1,
        'min': 60,
        'hour': 60 * 60,
        'day': 60 * 60 * 24,
        'week': 60 * 60 * 24 * 7,
        'month': 60 * 60 * 24 * 365.25/12,  # Approximate month duration (30.4375 days)
        'common_month': 60 * 60 * 24 * 365/12,   # Approximate month duration (30.4167 days)
        'year': 60 * 60 * 24 * 365.25,   # Approximate year duration (365.25 days)
        'common_year': 60 * 60 * 24 * 365   # Exact year duration (365 days)
    }

    # Define unit aliases in groups for clarity
    unit_groups = {
        'ns': ['ns', 'nanosecond', 'nanoseconds', 'nano'],
        'ms': ['ms', 'millisecond', 'milliseconds', 'milli'],
        's': ['s', 'second', 'seconds', 'sec'],
        'min': ['min', 'minute', 'minutes', 'mins'],
        'hour': ['h', 'hr', 'hour', 'hours', 'hrs'],
        'day': ['day', 'days', 'd'],
        'week': ['week', 'weeks', 'w', 'wks', 'wk'],
        'month': ['month', 'months', 'mon', 'mo'],
        'common_month': ['common month', 'common_month', 'cmonth', 'cmonths', 'cmon', 'cmons', 'cmo'],
        'year': ['year', 'years', 'y', 'yr', 'yrs'],
        'common_year': ['common year', 'common_year', 'cyr', 'cyrs']
    }

    # Create the unit_aliases dictionary using defaultdict
    unit_aliases = defaultdict(lambda: None, 
                               {alias: canonical for canonical, aliases in unit_groups.items() for alias in aliases})

    # Normalize the unit names
    from_unit_normalized = unit_aliases[from_unit]
    to_unit_normalized = unit_aliases[to_unit]

    # Handle case where the unit might not be found
    if from_unit_normalized is None:
        raise ValueError(f"Unknown unit: {from_unit}")
    if to_unit_normalized is None:
        raise ValueError(f"Unknown unit: {to_unit}")

    # Convert from seconds to the target unit using the normalized unit
    conversion_factor = conversion_factors_to_seconds[from_unit_normalized] / conversion_factors_to_seconds[to_unit_normalized]

    return conversion_factor

# In[ ]:




