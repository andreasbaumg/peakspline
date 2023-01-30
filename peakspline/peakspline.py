from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev as _splev
from scipy.interpolate import splint
from scipy.interpolate import splrep as _splrep
from scipy.optimize import minimize, minimize_scalar

tcktype = tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], int]

# relative integral of a Gaussian function \int_(-FWHM/2)^(FWHM/2) G(x) d x
# where \int_(-inf)^(inf) G(x) d x = 1.
GAUSS_FWHM_INTEGRAL = 0.7609681085504878


def splrep(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    w: npt.NDArray[np.float_] | None = None,
) -> tcktype:
    """Wrapper function to get proper type hints for
    :py:func:`scipy.interpolate.splrep`.
    """
    return _splrep(x, y, w)  # type: ignore


def splev(a: float | npt.NDArray[np.float_],
          tck: tcktype) -> float | npt.NDArray[np.float_]:
    """Wrapper function to get proper type hints for
    :py:func:`scipy.interpolate.splev`.
    """
    return _splev(a, tck)  # type: ignore


@dataclass
class PeakSplineResults:
    center: float
    resolution: float
    area: float
    tck: tcktype

    def __call__(
            self, x: npt.NDArray[np.float_] | float
    ) -> npt.NDArray[np.float_] | float:
        return splev(x, self.tck)


def _guess_fwhm(
        xdata: npt.NDArray[np.float_], ydata: npt.NDArray[np.float_]) -> float:
    """Guesses the FWHM of a peak.

    Args:
        ydata: Position data of peak.
        ydata: Intensity data of peak.

    Returns: FWHM guess.
    """
    ymax = ydata.max()
    ymin = ydata.min()
    yhalf = (ymax - ymin) / 2. + ymin
    overhalf = np.where(ydata >= yhalf)[0]
    imin, imax = overhalf.min(), overhalf.max()
    return np.abs(xdata[imax] - xdata[imin])


def _guess_center(
        xdata: npt.NDArray[np.float_], ydata: npt.NDArray[np.float_]) -> float:
    """Guesses center of a peak.

    Args:
        ydata: Position data of peak.
        ydata: Intensity data of peak.

    Returns: Center guess.
    """
    return xdata[ydata.argmax()]


def _get_x_bounds(tck: tcktype) -> tuple[float, float]:
    """Returns min and max position values.

    Args:
        tck: Spline tck parameters.

    Returns: Min and max position values.
    """
    return tck[0][0], tck[0][-1]


def _fit_interal_from_center(
    center: float,
    integral: float,
    guess: float,
    tck: tcktype,
) -> float:
    """Fits the width symmetrically from `center` so that the integral from
    `center`- width/2 to `center`- width/2 is equal to `integral`.

    Args:
        center: Center position of the fit.
        integral: Target integral.
        guess: Guess of the width.
        tck: Spline tck parameters.

    Returns: Width of fitted integral.
    """
    def func(x: float):
        a = center - x
        b = center + x
        return np.abs(splint(a, b, tck) - integral)  # type: ignore

    b = _get_x_bounds(tck)
    b = (b[1] - b[0]) / 2.
    res = minimize(func, guess / 2, bounds=([0, b], ), method="TNC")
    return abs(res.x[0]) * 2


def _fit_resolution(
    xdata: npt.NDArray[np.float_],
    ydata: npt.NDArray[np.float_],
    tck: tcktype,
    center: float,
    integral_tot: float,
    integral_res: float = GAUSS_FWHM_INTEGRAL,
) -> float:
    """Fits the resolution symmetrically from `center` so that the integral from
    `center`- width/2 to `center`- width/2 is equal to `integral`.

    Args:
        center: Center position of the fit.
        integral: Target integral.
        tck: Spline tck parameters.
        center: Center of the peak.
        integral_tot: Total integral of the peak.
        integral_res: Relative integral value that is equivalent to the
            resolution. Must be 0<=integral_res<=1.

    Returns: Width of fitted integral.
    """
    integ = integral_tot * integral_res
    guess = _guess_fwhm(xdata, ydata)
    return _fit_interal_from_center(center, integ, guess, tck)


def _fit_center_median(
    xdata: npt.NDArray[np.float_],
    ydata: npt.NDArray[np.float_],
    tck: tcktype,
    integral_tot: float,
) -> float:
    """Fits the center of a peak so that the integral from -inf to center ==
    integral from center to inf.

    Args:
        ydata: Position data of peak.
        ydata: Intensity data of peak.
        tck: Spline tck parameters.
        integral_tot: Total integral of the peak.

    Returns: Center position
    """
    a = tck[0][0]  # type: float
    area = integral_tot / 2

    def func(b: float) -> float:
        return np.abs(splint(a, b, tck) - area)  # type: ignore

    return minimize(
        func,
        _guess_center(xdata, ydata),
        bounds=(_get_x_bounds(tck), ),
        method="TNC",
    ).x[0]


def _fit_center_max(
    xdata: npt.NDArray[np.float_],
    ydata: npt.NDArray[np.float_],
    tck: tcktype,
) -> float:
    """Fits the center of a peak, so that the center is at the maximum value of
    the peak.

    Args:
        ydata: Position data of peak.
        ydata: Intensity data of peak.
        tck: Spline tck parameters.

    Returns: Center position
    """
    def func(x):
        # return negative value since the minimum is searched
        return -splev(x, tck)

    pmax = np.argmax(ydata)
    # max of spline must be next to max of ydata
    il, ir = np.clip((pmax - 1, pmax + 1), 0, len(xdata) - 1)
    return minimize_scalar(
        func, bounds=(xdata[il], xdata[ir]), method='bounded').x


def _integral_tot(tck: tcktype) -> float:
    """Calculates the total integral of a spline from the first to the last
    knot.

    Args:
        tck: Spline tck parameters.

    Returns: integral from first to last knot.
    """
    a, b = _get_x_bounds(tck)
    return splint(a, b, tck)  # type: ignore


def fit_peakspline(
    xdata: npt.NDArray[np.float_],
    ydata: npt.NDArray[np.float_],
    weights: npt.NDArray[np.float_] | None = None,
    mode: str = "median",
    integral_res: float = GAUSS_FWHM_INTEGRAL,
) -> PeakSplineResults:
    """Fits a cubic spline to peak data and determines center, resolution and
    the area under the curve.

    Args:
        ydata: Position data of peak.
        ydata: Intensity data of peak.
        weights: Weights of the data.
        threshold: Values below the threshold are used for detection of response
            edge. As threshold is a relative value referenced to 1, values must
            be between 0 and 1.
        nmin: Minimum amount of values which must below the threshold. Half of
            nmin must be on the left and right hand side of the peak.
        mode: If mode is `median` (default) the median is used to determine the
            center position. If mode is `max` the maximum position is used as
            the center position.
        integral_res: Relative integral value that is equivalent to the
            resolution. Must be 0<=integral_res<=1. Default value is the
            integral over the FWHM of a Gaussian curve.

    Returns: Optimized peak parameters.
    """
    idx = np.argsort(xdata)
    xdata = np.take(xdata, idx)
    ydata = np.take(ydata, idx)
    if weights is not None:
        weights = np.take(weights, idx)
    tck = splrep(xdata, ydata, w=weights)
    integ_tot = _integral_tot(tck)
    if mode == "median":
        center = _fit_center_median(xdata, ydata, tck, integ_tot)
    elif mode == "max":
        center = _fit_center_max(xdata, ydata, tck)
    else:
        raise ValueError("mode must be one of ['median', 'max']")
    resolution = _fit_resolution(
        xdata, ydata, tck, center, integ_tot, integral_res)
    return PeakSplineResults(center, resolution, integ_tot, tck)
