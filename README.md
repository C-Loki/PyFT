# PyFT Power Spectrum Function

[![Build Status](https://img.shields.io/github/actions/workflow/status/C-Loki/PyFT/ci.yml)](https://github.com/C-Loki/PyFT/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/C-Loki/PyFT)](https://codecov.io/gh/C-Loki/PyFT)
[![License](https://img.shields.io/github/license/C-Loki/PyFT)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

## Short Description
The PyFT power_spectrum function is a Python tool designed to compute and plot the power spectrum of time series data. It uses Fourier transform techniques followed by Frandsen et al. 1995 Sect 4.2 to analyze the frequency components present in the data. For further details, refer to [Frandsen et al. 1995](https://ui.adsabs.harvard.edu/abs/1995A%26A...301..123F/abstract).

# Features
- Reads time series data from a CSV file.
- Detrends the data to remove any linear trends.
- Computes the Fourier transform of the data.
- Calculates and plots the power spectral density.
- Provides recommended frequency intervals for analysis.
- Computes the spectral window at a given frequency.

# Requirements
pandas
numpy
scipy
xarray
matplotlib
cftime


## How to Run
### Input Data
To run the PyFT power_spectrum function, you'll need to:

- Specify the path to your data file (`data_path`) as a string. The data file should be in CSV (.csv) or NetCDF (.nc) format.
- Define the column containing the time information (`time_column`) and the column or variable containing the signal (`column_of_interest`) as strings.

### Optional Parameters
You can initialize the following optional parameters when running the function:

- `df` (recommended): Frequency step in the same units as the input data. By default, it's calculated based on the data to allow oversampling of df=1/5dt, where dt represents the time-step in the data. This calculation accommodates data with gaps as well.
- `f_int` (recommended): Frequency interval over which to calculate the power spectrum, expressed in the same units as the input data. By default, this interval spans from 0 to 0.9fnyq, where fnyq is the Nyquist frequency. This default setting ensures that the user can observe all spectral information without redundancy beyond the Nyquist frequency. **Warning:** For large datasets, the default interval may become very large, which can overwhelm the computer's memory causing it to crash. Therefore, it is advisable to specify an `f_int` before running the function on large datasets.
- `time_units` (recommended): A list of length 2 containing the current and desired time units for conversion as strings. The first element represents the current time units, while the second element denotes the desired time units. By default, it is set to ['s', 'ms'], indicating that the time series data is assumed to be in seconds, and the output power spectrum will be in milliseconds^-1.
- `unc_column`: Column or variable name containing uncertainty values, as a string. If provided, these values will be used as weights to enhance the accuracy of the output power spectrum.
- `calc_unc`: Boolean parameter indicating whether to calculate uncertainties. When set to true, the uncertainties will be computed using the custom PyFT function calc_noise. This function employs a boxcar method to determine the data fluctuation over time by one standard deviation. These standard deviations will be used to weight the power spectrum data. **Note:** If `unc_column` is being used, `calc_unc` should not be enabled. If these two features are used together, only `unc_column` will be accepted.
- `verbose`: Level of verbosity (0: minimal output, 2: detailed output).
- `recommendations`: Whether to print recommended frequency parameters such as `df` and `f_int`.
- `plotting`: Plotting mode 
  - 0: No plots will be generated.
  - 1: Frequency domain will be plotted within the specified interval.
  - 2: Zoomed-in frequency domain will be plotted using the recommended interval.

### Using the Function
After setting the input parameters, call the `power_spectrum` function with the specified parameters. The function will process the data, compute the power spectrum, and generate plots according to the specified settings.
