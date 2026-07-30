import os

import numpy as np
import pytest
from astropy.io import fits
from mpdaf.tests.utils import get_data_file

from zap import process


def test_process(tmp_path):
    """just run zap on a cube to see if it runs until the end ..."""
    testfile = get_data_file("sdetect", "minicube.fits")
    outfile = str(tmp_path / "OUTPUT.fits")
    process(testfile, outcubefits=outfile)
    assert os.path.exists(outfile)


def test_zero_nan_slices(tmp_path):
    testfile = get_data_file("sdetect", "minicube.fits")
    with fits.open(testfile) as hdul:
        hdul[1].data[:10] = np.nan
        hdul.writeto(tmp_path / "minicube.fits")

    testfile = str(tmp_path / "minicube.fits")
    outfile = str(tmp_path / "OUTPUT.fits")

    process(testfile, outcubefits=outfile, zero_nan_slices=True)
    assert os.path.exists(outfile)

    with (
        pytest.warns(RuntimeWarning, match="Mean of empty slice"),
        pytest.raises(ValueError, match="Found array with 0 sample"),
    ):
        process(testfile, outcubefits=outfile, zero_nan_slices=False, overwrite=True)
