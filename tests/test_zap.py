import os

from mpdaf.tests.utils import get_data_file

from zap import process


def test_process(tmp_path):
    """just run zap on a cube to see if it runs until the end ..."""
    testfile = get_data_file("sdetect", "minicube.fits")
    outfile = str(tmp_path / "OUTPUT.fits")
    process(testfile, outcubefits=outfile)
    assert os.path.exists(outfile)
