# PyFT

# Overview
This Python script calculates and plots the power spectral density of time series data. It also computes the spectral window at a given frequency using sine and cosine functions. The script is designed to be flexible and provides recommendations for sampling frequency and frequency intervals.

# Features
- Reads time series data from a CSV file.
- Detrends the data to remove any linear trends.
- Computes the Fourier transform of the data.
- Calculates and plots the power spectral density.
- Provides recommended frequency intervals for analysis.
- Computes the spectral window at a given frequency.

# Requirements
Python 3.x
pandas
numpy
scipy
xarray
matplotlib

# User Inputs
data_path: Path to the CSV file containing the time series data. The file should have a 'time' column and at least one signal column.
column_of_interest: The column name of the signal to be analyzed.
df: Frequency step (default is recommended value).
f_int: Frequency interval (default is recommended value).
verbose: Verbosity level for logging messages.
recommendations: Boolean flag to print recommendations for frequency steps and intervals.

# How to Use
...

# Code Explaination
...