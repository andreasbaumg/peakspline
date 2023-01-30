# Peak-Spline

This package provides the _fit_peakspline_ function for fitting Gaussian-like
functions. These functions can have an arbitrary shape but must reach on the
left and right side of the peak values near zero.

The code provided is part of a larger package for the evaluation of
imaging spectrometer calibration data, which is currently not publicly
available.


## Installation

Download and install the package with

    git clone ...
    pip install peakspline


## Usage

Please refer to the [examples](./examples) folder.


## Limitations

The accuracy of the fit is limited by the uncertainty and the distance
of the sampling points. Since the spline passes through every sampling point,
outliers can have a significant impact on the results.


