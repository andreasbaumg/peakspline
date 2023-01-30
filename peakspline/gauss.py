from __future__ import annotations

import numpy as np
import numpy.typing as npt

FWHM2SIGMAFACTOR = (2 * np.sqrt(2 * np.log(2)))
GAUSSINTFACTOR = 2 * np.sqrt(np.pi / 2.) / FWHM2SIGMAFACTOR


def gauss_fwhm2sigma(
        fwhm: npt.NDArray[np.float_] | float) -> npt.NDArray[np.float_] | float:
    return fwhm / FWHM2SIGMAFACTOR


def gauss(
    x: npt.NDArray[np.float_] | float,
    center: npt.NDArray[np.float_] | float,
    fwhm: npt.NDArray[np.float_] | float,
    amp: npt.NDArray[np.float_] | float,
    offset: npt.NDArray[np.float_] | float = 0.,
) -> npt.NDArray[np.float_] | float:
    sigma = gauss_fwhm2sigma(fwhm)
    return amp * np.exp(-((x - center)**2 / (2 * sigma**2))) + offset


def gauss_integ_tot(
    fwhm: npt.NDArray[np.float_] | float,
    amp: npt.NDArray[np.float_] | float = 1,
) -> npt.NDArray[np.float_] | float:
    """Integral of Gaussian function.

    Args:
        fwhm: FWHM of Gaussian function.
        amp: Amplitude of Gaussian function.

    Returns: Integral of Gaussian function.
    """
    return amp * GAUSSINTFACTOR * fwhm
