from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from peakspline.gauss import gauss, gauss_integ_tot
from peakspline.peakspline import (
    _fit_center_max,
    _fit_center_median,
    _fit_resolution,
    _get_x_bounds,
    _guess_center,
    _guess_fwhm,
    _integral_tot,
    fit_peakspline,
    splrep,
)


@dataclass
class SplineData:
    xdata = np.arange(490., 510., 1.)
    xhr = np.linspace(xdata.min(), xdata.max(), 1000)
    center = 500.3
    fwhm = 3.6
    amp = 1.
    ydata = gauss(xdata, center, fwhm, amp, 0.)
    tck = splrep(xdata, ydata)  # type: ignore


@pytest.fixture(scope="module")
def sd():
    np.random.seed(1337)
    sd = SplineData()
    sd.ydata = np.random.normal(sd.ydata, 0.01 * sd.ydata)
    return sd


@dataclass
class SplineDataDoubleGauss:
    xdata = np.arange(490., 510., 1.)
    xhr = np.linspace(xdata.min(), xdata.max(), 1000)
    center = 500.3
    amp = 1.
    fwhm = 3.2363264
    fwhm_gauss = 2
    ydata = gauss(xdata, center - 1, fwhm_gauss, amp) + gauss(
        xdata, center + 1, fwhm_gauss, amp)
    tck = splrep(xdata, ydata)  # type: ignore


@pytest.fixture(scope="module")
def sd_double_gauss():
    np.random.seed(1337)
    sd = SplineDataDoubleGauss()
    sd.ydata = np.random.normal(sd.ydata, 0.01 * sd.ydata)
    return sd


def test_guess_fwhm(sd):
    assert _guess_fwhm(sd.xdata, sd.ydata) == pytest.approx(sd.fwhm, 2)


def test_guess_center(sd):
    assert _guess_center(sd.xdata, sd.ydata) == pytest.approx(
        sd.center, abs=np.diff(sd.xdata).max())


def test_get_bounds(sd):
    assert _get_x_bounds(sd.tck) == (sd.xdata.min(), sd.xdata.max())


def test_integral_tot(sd):
    assert _integral_tot(sd.tck) == pytest.approx(
        gauss_integ_tot(sd.fwhm, sd.amp), rel=2e-3)


def test_fit_center_median(sd):
    assert _fit_center_median(
        sd.xdata, sd.ydata, sd.tck,
        _integral_tot(sd.tck)) == pytest.approx(sd.center)


def test_fit_center_max(sd):
    assert _fit_center_max(sd.xdata, sd.ydata, sd.tck) == pytest.approx(
        sd.center, abs=2e-2)


def test_fit_resolution(sd):
    assert _fit_resolution(
        sd.xdata, sd.ydata, sd.tck, sd.center,
        _integral_tot(sd.tck)) == pytest.approx(
            sd.fwhm, abs=2e-3)


def test_fit_peakspline(sd):
    res = fit_peakspline(sd.xdata, sd.ydata)
    assert res.center == pytest.approx(sd.center, abs=2e-2)
    assert res.resolution == pytest.approx(sd.fwhm, abs=2.e-2)
    assert res.area == pytest.approx(gauss_integ_tot(sd.fwhm, sd.amp), rel=3e-3)


def test_fit_peakspline_double_gauss(sd_double_gauss):
    sd = sd_double_gauss
    res = fit_peakspline(sd.xdata, sd.ydata)
    assert res.center == pytest.approx(sd.center, abs=2e-2)
    assert res.resolution == pytest.approx(sd.fwhm, abs=2.e-2)
    assert res.area == pytest.approx(
        gauss_integ_tot(sd.fwhm_gauss, sd.amp) * 2, rel=3e-3)
