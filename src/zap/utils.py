import logging
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

__all__ = ['mask_nan_edges']


def mask_nan_edges(cube, outfile=None, plot=False, threshold=50,
                   extname='DATA'):
    """Mask the edges of a cube, using the number of nans in a spaxel.

    At the edges of MUSE cubes, spaxels can contain many NaNs in the spectral
    axis. Because of this, ZAP will not subtract the sky for these spaxels,
    which will lead to spaxels with higher flux. This function allows to mask
    these spaxels, using a threshold on the percentage of NaN values in the
    spectral axis.

    Parameters
    ----------
    cube : ndarray or str
        Cube data or FITS filename.
    outfile : str
        Path of the output (masked) cube. This can only be used if the input
        ``cube`` is a filename, as the DATA extension will be replaced by the
        masked cube.
    plot: bool
        Show images of the different steps (default: False).
    threshold: float
        Percentage of NaNs above which the spaxel will be masked.
    extname : str
        Extension name for the data, defaults to DATA.

    Returns
    -------
    mask, cube : ndarray, ndarray
        The computed mask and the masked cube.

    """
    logger = logging.getLogger(__name__)

    if isinstance(cube, str):
        data = fits.getdata(cube, extname=extname)
    else:
        data = cube

    nans = (100 / data.shape[0]) * np.sum(np.isnan(data), axis=0)
    mask = nans > threshold
    labels, nlabels = ndi.label(mask)

    if nlabels > 0:
        area = [np.sum(mask[labels == l]) for l in range(1, nlabels + 1)]
        mask = labels == (np.argmax(area) + 1)
        logger.info('%i label(s), selected one contain %i pixels', nlabels,
                    np.sum(mask))
    else:
        logger.warning('Nothing masked')
        return mask, data

    if plot:
        im = np.nanmean(data, axis=0)

    if outfile is not None:
        if not isinstance(cube, str):
            raise ValueError('cannot save the file if the output was given '
                             'as ndarray')

        logger.info("Save cube to %s", outfile)
        with fits.open(cube) as hdul:
            hdul[extname].data = data
            hdul.writeto(outfile, overwrite=True)

    if plot:
        import matplotlib.pyplot as plt
        from astropy.stats import sigma_clip

        imorig = im.copy()
        im[mask] = np.nan
        clipped = sigma_clip(np.ma.masked_invalid(im), sigma=5, maxiters=3)
        mean = clipped.mean()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        vmin, vmax = (mean - 1), (mean + 1)
        ax1.imshow(imorig, vmin=vmin, vmax=vmax, origin='lower')
        ax1.set_title('Input image')
        ax2.imshow(im, vmin=vmin, vmax=vmax, origin='lower')
        ax2.set_title('Masked image')
        ax3.imshow(np.isnan(im) ^ np.isnan(imorig), cmap='binary',
                   origin='lower')
        ax3.set_title('Masked pixels')

    return mask, data
