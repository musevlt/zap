# ZAP - Zurich Atmosphere Purge
#
# Copyright (c) 2014-2016 Kurt Soto
# Copyright (c) 2015-2019 Simon Conseil
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import astropy.units as u
import logging
import numpy as np
import os
import scipy.ndimage as ndi
import sys
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from functools import wraps
from multiprocessing import cpu_count, Manager, Process
from scipy.stats import sigmaclip
from sklearn.decomposition import PCA
from time import time

from zap._version import __version__

__all__ = ['process', 'SVDoutput', 'nancleanfits', 'contsubfits', 'Zap',
           'SKYSEG']

# Limits of the segments in Angstroms. Zap now uses by default only one
# segment, based on the cube wavelength's min and max.  See below for the
# old values.
SKYSEG = []

# These are the old limits from the original zap. Keeping them for reference,
# and may be useful for further testing.
# [0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000]

# range where the NaD notch filters absorbs significant flux
NOTCH_FILTER_RANGES = {
    'WFM-AO-E': [5755, 6008],
    'WFM-AO-N': [5805, 5966],
    'NFM-AO-N': [5780, 6050],
}

# List of allowed values for cftype (continuum filter)
CFTYPE_OPTIONS = ('median', 'fit', 'none')

# Number of available CPUs
NCPU = cpu_count()

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


# ================= Top Level Functions =================

def process(cubefits, outcubefits='DATACUBE_ZAP.fits', clean=True,
            zlevel='median', cftype='median', cfwidthSVD=300, cfwidthSP=300,
            nevals=[], extSVD=None, skycubefits=None, mask=None,
            interactive=False, ncpu=None, pca_class=None, n_components=None,
            overwrite=False, varcurvefits=None):
    """ Performs the entire ZAP sky subtraction algorithm.

    This is the main ZAP function. It works on an input FITS file and
    optionally writes the product to an output FITS file.

    Parameters
    ----------
    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    outcubefits : str
        Output FITS file, based on the input one to propagate all header
        information and other extensions. Default to `DATACUBE_ZAP.fits`.
    clean : bool
        If True (default value), the NaN values are cleaned. Spaxels with more
        then 25% of NaN values are removed, the others are replaced with an
        interpolation from the neighbors. The NaN values are reinserted into
        the final datacube. If set to False, any spaxel with a NaN value will
        be ignored.
    zlevel : str
        Method for the zeroth order sky removal: `none`, `sigclip` or `median`
        (default).
    cftype : {'median', 'fit', 'none'}
        Method for the continuum filter.
    cfwidthSVD : int or float
        Window size for the continuum filter, for the SVD computation.
        Default to 300.
    cfwidthSP : int or float
        Window size for the continuum filter used to remove the continuum
        features for calculating the eigenvalues per spectrum. Smaller values
        better trace the sources. An optimal range of is typically
        20 - 50 pixels. Default to 300.
    nevals : list
        Allow to specify the number of eigenspectra used for each segment.
        Provide either a single value that will be used for all of the
        segments, or a list of 11 values that will be used for each of the
        segments.
    extSVD : Zap object
        Can be a ``Zap`` object output from :func:`~zap.SVDoutput`.
        If given, the SVD from this object will be used, otherwise the SVD is
        computed. So this allows to compute the SVD on an other field or with
        different settings.
    skycubefits : str
        Path for the optional output of the sky that is subtracted from the
        cube. This is simply the input cube minus the output cube.
    mask : str
        A 2D fits image to exclude regions that may contaminate the zlevel or
        eigenspectra. This image should be constructed from the datacube itself
        to match the dimensionality. Sky regions should be marked as 0, and
        astronomical sources should be identified with an integer greater than
        or equal to 1. Default to None.
    interactive : bool
        If True, a :class:`~zap.Zap` object containing all information on
        the ZAP process is returned, and can be used to explore the
        eigenspectra and recompute the output (with the
        :meth:`~zap.Zap.reprocess` method). In this case, the output files
        are not saved (`outcubefits` and `skycubefits` are ignored). Default
        to False.
    varcurvefits : str
        Path for the optional output of the explained variance curves.

    """
    logger.info('Running ZAP %s !', __version__)
    t0 = time()
    if not isinstance(cubefits, str):
        raise TypeError('The process method only accepts a single datacube '
                        'filename.')

    # check if outcubefits/skycubefits exists before beginning
    if not overwrite:
        def _check_file_exists(filename):
            if filename is not None and os.path.exists(filename):
                raise IOError('Output file "{0}" exists'.format(filename))

        _check_file_exists(outcubefits)
        _check_file_exists(skycubefits)

    if ncpu is not None:
        global NCPU
        NCPU = ncpu

    if extSVD is not None and mask is not None:
        raise ValueError('extSVD and mask parameters are incompatible: if mask'
                         ' must be used, then the SVD has to be recomputed')

    if mask is not None or (extSVD is None and cfwidthSVD != cfwidthSP):
        # Compute the SVD separately, only if a mask is given, or if the
        # cfwidth values differ and extSVD is not given. Otherwise, the SVD
        # will be computed in the _run method, which allows to avoid running
        # twice the zlevel and continuumfilter steps.
        extSVD = SVDoutput(cubefits, clean=clean, zlevel=zlevel,
                           cftype=cftype, cfwidth=cfwidthSVD, mask=mask)

    zobj = Zap(cubefits, pca_class=pca_class, n_components=n_components)
    zobj._run(clean=clean, zlevel=zlevel, cfwidth=cfwidthSP, cftype=cftype,
              nevals=nevals, extSVD=extSVD)

    if interactive:
        # Return the zobj object without saving files
        return zobj

    if skycubefits is not None:
        zobj.writeskycube(skycubefits=skycubefits, overwrite=overwrite)

    if varcurvefits is not None:
        zobj.writevarcurve(varcurvefits=varcurvefits, overwrite=overwrite)

    zobj.mergefits(outcubefits, overwrite=overwrite)
    logger.info('Zapped! (took %.2f sec.)', time() - t0)


def SVDoutput(cubefits, clean=True, zlevel='median', cftype='median',
              cfwidth=300, mask=None, ncpu=None, pca_class=None,
              n_components=None):
    """Performs the SVD decomposition of a datacube.

    This allows to use the SVD for a different datacube. It used to allow to
    save the SVD to a file but this is no more possible. Instead it returns
    a ``Zap`` which can be given to the :func:`~zap.process` function.

    Parameters
    ----------
    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    clean : bool
        If True (default value), the NaN values are cleaned. Spaxels with more
        then 25% of NaN values are removed, the others are replaced with an
        interpolation from the neighbors.
    zlevel : str
        Method for the zeroth order sky removal: `none`, `sigclip` or `median`
        (default).
    cftype : {'median', 'fit', 'none'}
        Method for the continuum filter.
    cfwidth : int or float
        Window size for the continuum filter, default to 300.
    mask : str
        Path of a FITS file containing a mask (1 for objects, 0 for sky).

    """
    logger.info('Processing %s to compute the SVD', cubefits)

    if ncpu is not None:
        global NCPU
        NCPU = ncpu

    zobj = Zap(cubefits, pca_class=pca_class, n_components=n_components)
    zobj._prepare(clean=clean, zlevel=zlevel, cftype=cftype,
                  cfwidth=cfwidth, mask=mask)
    zobj._msvd()
    return zobj


def contsubfits(cubefits, outfits='CONTSUB_CUBE.fits', ncpu=None,
                cftype='median', cfwidth=300, clean_nan=True, zlevel='median',
                overwrite=False):
    """A standalone implementation of the continuum removal."""
    if ncpu is not None:
        global NCPU
        NCPU = ncpu

    zobj = Zap(cubefits)
    zobj._prepare(clean=clean_nan, zlevel=zlevel, cftype=cftype,
                  cfwidth=cfwidth)
    cube = zobj.make_contcube()

    outhead = _newheader(zobj)
    outhdu = fits.PrimaryHDU(data=cube, header=outhead)
    outhdu.writeto(outfits, overwrite=overwrite)
    logger.info('Continuum cube file saved to %s', outfits)


def nancleanfits(cubefits, outfn='NANCLEAN_CUBE.fits', rejectratio=0.25,
                 boxsz=1, overwrite=False):
    """Interpolates NaN values from the nearest neighbors.

    Parameters
    ----------
    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    outfn : str
        Output FITS file. Default to ``NANCLEAN_CUBE.fits``.
    rejectratio : float
        Defines a cutoff for the ratio of NAN to total pixels in a spaxel
        before the spaxel is avoided completely. Default to 0.25
    boxsz : int
        Defines the number of pixels around the offending NaN pixel.
        Default to 1, which looks for the 26 nearest neighbors which
        is a 3x3x3 cube.

    """
    with fits.open(cubefits) as hdu:
        hdu[1].data = _nanclean(hdu[1].data, rejectratio=rejectratio,
                                boxsz=boxsz)[0]
        hdu.writeto(outfn, overwrite=overwrite)


def timeit(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        logger.info('%s - Time: %.2f sec.', func.__name__, time() - t0)
        return res
    return wrapped


# ================= Main class =================

class Zap(object):

    """ Main class to run each of the steps of ZAP.

    Attributes
    ----------

    cleancube : numpy.ndarray
        The final datacube after removing all of the residual features.
    contarray : numpy.ndarray
        A 2D array containing the subtracted continuum per spaxel.
    cube : numpy.ndarray
        The original cube with the zlevel subtraction performed per spaxel.
    laxis : numpy.ndarray
        A 1d array containing the wavelength solution generated from the header
        parameters.
    wcs : astropy.wcs.WCS
        WCS object with the wavelength solution.
    lranges : list
        A list of the wavelength bin limits used in segmenting the sepctrum
        for SVD.
    nancube : numpy.ndarray
        A 3d boolean datacube containing True in voxels where a NaN value was
        replaced with an interpolation.
    nevals : numpy.ndarray
        A 1d array containing the number of eigenvalues used per segment to
        reconstruct the residuals.
    normstack : numpy.ndarray
        A normalized version of the datacube decunstructed into a 2d array.
    pranges : numpy.ndarray
        The pixel indices of the bounding regions for each spectral segment.
    recon : numpy.ndarray
        A 2d array containing the reconstructed emission line residuals.
    run_clean : bool
        Boolean that indicates that the NaN cleaning method was used.
    run_zlevel : bool
        Boolean indicating that the zero level correction was used.
    stack : numpy.ndarray
        The datacube deconstructed into a 2d array for use in the the SVD.
    variancearray : numpy.ndarray
        A list of length nsegments containing variances calculated per spaxel
        used for normalization
    y,x : numpy.ndarray
        The position in the cube of the spaxels that are in the 2d
        deconstructed stack
    zlsky : numpy.ndarray
        A 1d array containing the result of the zero level subtraction

    """

    def __init__(self, cubefits, pca_class=None, n_components=None):
        self.cubefits = cubefits
        self.ins_mode = None

        with fits.open(cubefits) as hdul:
            self.instrument = hdul[0].header.get('INSTRUME')
            if self.instrument == 'MUSE':
                self.ins_mode = hdul[0].header.get('HIERARCH ESO INS MODE')
                self.cube = hdul[1].data
                self.header = hdul[1].header
            elif self.instrument == 'KCWI':
                self.cube = hdul[0].data
                self.header = hdul[0].header
            elif self.instrument == 'FOCAS':
                self.cube = hdul[0].data
                self.header = hdul[0].header
            else:
                raise ValueError('unsupported instrument %s' % self.instrument)

        # Workaround for floating points errors in wcs computation: if cunit is
        # specified, wcslib will convert in meters instead of angstroms, so we
        # remove cunit before creating the wcs object
        header = self.header.copy()
        unit = u.Unit(header.pop('CUNIT3'))
        self.wcs = WCS(header).sub([3])

        # Create Lambda axis
        wlaxis = np.arange(self.cube.shape[0])
        self.laxis = self.wcs.all_pix2world(wlaxis, 0)[0]
        if unit != u.angstrom:
            # Make sure lambda is in angstroms
            self.laxis = (self.laxis * unit).to(u.angstrom).value

        # Change laser region into zeros if AO
        if self.ins_mode in NOTCH_FILTER_RANGES:
            logger.info('Cleaning laser region for AO, mode=%s, limits=%s',
                        self.ins_mode, NOTCH_FILTER_RANGES[self.ins_mode])
            self.notch_limits = self.wcs.all_world2pix(
                NOTCH_FILTER_RANGES[self.ins_mode], 0)[0].astype(int)
            lmin, lmax = self.notch_limits
            self.cube[lmin:lmax + 1] = 0.0
        else:
            self.notch_limits = None

        # NaN Cleaning
        self.run_clean = False
        self.nancube = None
        self._boxsz = 1
        self._rejectratio = 0.25

        # Mask file
        self.maskfile = None

        # zlevel parameters
        self.run_zlevel = False
        self.zlsky = np.zeros_like(self.laxis)

        # Extraction results
        self.stack = None
        self.y = None
        self.x = None

        # Normalization Maps
        self.contarray = None
        self.variancearray = None
        self.normstack = None

        # identify the spectral range of the dataset
        laxmin = min(self.laxis)
        laxmax = max(self.laxis)

        # List of segmentation limits in the optical
        skyseg = np.array(SKYSEG)
        skyseg = skyseg[(skyseg > laxmin) & (skyseg < laxmax)]

        # segment limit in angstroms
        self.lranges = (np.vstack([np.append(laxmin - 10, skyseg),
                                   np.append(skyseg, laxmax + 10)])).T

        # segment limit in pixels
        laxis = self.laxis
        lranges = self.lranges
        pranges = []
        for i in range(len(lranges)):
            paxis = wlaxis[(laxis > lranges[i, 0]) & (laxis <= lranges[i, 1])]
            pranges.append((np.min(paxis), np.max(paxis) + 1))
        self.pranges = np.array(pranges)

        # eigenspace Subset
        if pca_class is not None:
            logger.info('Using %s', pca_class)
            self.pca_class = pca_class
        else:
            self.pca_class = PCA

        # Reconstruction of sky features
        self.n_components = n_components
        self.recon = None
        self.cleancube = None

    @timeit
    def _prepare(self, clean=True, zlevel='median', cftype='median',
                 cfwidth=300, extzlevel=None, mask=None):
        # clean up the nan values
        if clean:
            self._nanclean()

        # if mask is supplied, apply it
        if mask is not None:
            self._applymask(mask)

        # Extract the spectra that we will be working with
        self._extract()

        # remove the median along the spectral axis
        if extzlevel is None:
            if zlevel.lower() != 'none':
                self._zlevel(calctype=zlevel)
        else:
            self._externalzlevel(extzlevel)

        # remove the continuum level - this is multiprocessed to speed it up
        self._continuumfilter(cfwidth=cfwidth, cftype=cftype)

        # normalize the variance in the segments.
        self._normalize_variance()

    def _run(self, clean=True, zlevel='median', cftype='median',
             cfwidth=300, nevals=[], extSVD=None):
        """ Perform all steps to ZAP a datacube:

        - NaN re/masking,
        - deconstruction into "stacks",
        - zerolevel subraction,
        - continuum removal,
        - normalization,
        - singular value decomposition,
        - eigenvector selection,
        - residual reconstruction and subtraction,
        - data cube reconstruction.

        """
        self._prepare(clean=clean, zlevel=zlevel, cftype=cftype,
                      cfwidth=cfwidth, extzlevel=extSVD)

        # do the multiprocessed SVD calculation
        if extSVD is None:
            self._msvd()
        else:
            self.models = extSVD.models

        self.components = [m.components_.copy() for m in self.models]

        # choose some fraction of eigenspectra or some finite number of
        # eigenspectra
        if nevals == []:
            self.optimize()
            self.chooseevals(nevals=self.nevals)
        else:
            self.chooseevals(nevals=nevals)

        # reconstruct the sky residuals using the subset of eigenspace
        self.reconstruct()

        # stuff the new spectra back into the cube
        self.remold()

    def _nanclean(self):
        """
        Detects NaN values in cube and removes them by replacing them with an
        interpolation of the nearest neighbors in the data cube. The positions
        in the cube are retained in nancube for later remasking.
        """
        self.cube, self.nancube = _nanclean(
            self.cube, rejectratio=self._rejectratio, boxsz=self._boxsz)
        self.run_clean = True

    @timeit
    def _extract(self):
        """Deconstruct the datacube into a 2d array.

        Since spatial information is not required, and the linear algebra
        routines require 2d arrays. The operation rejects any spaxel with even
        a single NaN value, since this would cause the linear algebra routines
        to crash.

        Adds the x and y data of these positions into the Zap class

        """
        # make a map of spaxels with NaNs
        badmap = (np.logical_not(np.isfinite(self.cube))).sum(axis=0)
        # get positions of those with no NaNs
        self.y, self.x = np.where(badmap == 0)
        # extract those positions into a 2d array
        self.stack = self.cube[:, self.y, self.x]
        logger.info('Extract to 2D, %d valid spaxels (%d%%)', len(self.x),
                    len(self.x) / np.prod(self.cube.shape[1:]) * 100)

    def _externalzlevel(self, extSVD):
        """Remove the zero level from the extSVD file."""
        logger.debug('Using external zlevel from %s', extSVD)
        if isinstance(extSVD, Zap):
            self.zlsky = np.array(extSVD.zlsky, copy=True)
            self.run_zlevel = extSVD.run_zlevel
        else:
            self.zlsky = fits.getdata(extSVD, 0)
            self.run_zlevel = 'extSVD'
        self.stack -= self.zlsky[:, np.newaxis]

    @timeit
    def _zlevel(self, calctype='median'):
        """
        Removes a 'zero' level from each spectral plane. Spatial information is
        not required, so it operates on the extracted stack.

        Operates on stack, leaving it with this level removed and adds the data
        'zlsky' to the class. zlsky is a spectrum of the zero levels.

        This zero level is currently calculated with a median.

        Experimental operations -

        - exclude top quartile
        - run in an iterative sigma clipped mode

        """
        self.run_zlevel = calctype
        if calctype == 'none':
            logger.info('Skipping zlevel subtraction')
            return

        if calctype == 'median':
            logger.info('Median zlevel subtraction')
            func = _imedian
        elif calctype == 'sigclip':
            logger.info('Iterative Sigma Clipping zlevel subtraction')
            func = _isigclip
        else:
            raise ValueError('Unknow zlevel type, must be none, median, or '
                             'sigclip')

        self.zlsky = np.hstack(parallel_map(func, self.stack, NCPU, axis=0))
        self.stack -= self.zlsky[:, np.newaxis]

    @timeit
    def _continuumfilter(self, cfwidth=300, cftype='median'):
        """A multiprocessed implementation of the continuum removal.

        This process distributes the data to many processes that then
        reassemble the data.  Uses two filters, a small scale (less than the
        line spread function) uniform filter, and a large scale median filter
        to capture the structure of a variety of continuum shapes.

        added to class
        contarray - the removed continuua
        normstack - "normalized" version of the stack with the continuua
            removed

        """
        if cftype not in CFTYPE_OPTIONS:
            raise ValueError("cftype must be median, fit or none, got {}"
                             .format(cftype))
        logger.info('Applying Continuum Filter, cftype=%s', cftype)
        self._cftype = cftype
        self._cfwidth = cfwidth

        # remove continuum features
        if cftype == 'none':
            self.normstack = self.stack.copy()
        else:
            if cftype == 'fit' and self.instrument != 'MUSE':
                warnings.warn('the continuum fit method is currently adapted '
                              'to MUSE and should not be used for other '
                              'instruments', UserWarning)

            self.contarray = _continuumfilter(self.stack, cftype,
                                              cfwidth=cfwidth,
                                              notch_limits=self.notch_limits)
            self.normstack = self.stack - self.contarray

    def _normalize_variance(self):
        """Normalize the variance in the segments."""
        logger.debug('Normalizing variances')
        # self.variancearray = np.std(self.stack, axis=1)
        # self.normstack /= self.variancearray[:, np.newaxis]

        nseg = len(self.pranges)
        self.variancearray = var = np.zeros((nseg, self.stack.shape[1]))
        for i in range(nseg):
            pmin, pmax = self.pranges[i]
            var[i, :] = np.var(self.normstack[pmin:pmax, :], axis=0)
            self.normstack[pmin:pmax, :] /= var[i, :]

    @timeit
    def _msvd(self):
        """Multiprocessed singular value decomposition.

        Takes the normalized, spectral segments and distributes them
        to the individual svd methods.

        """
        logger.info('Calculating SVD on %d segments (%s)', len(self.pranges),
                    self.pranges)
        indices = [x[0] for x in self.pranges[1:]]
        # normstack = self.stack - self.contarray
        Xarr = np.array_split(self.normstack.T, indices, axis=1)

        self.models = []
        for i, x in enumerate(Xarr):
            if self.n_components is not None:
                ncomp = max(x.shape[1] * self.n_components, 60)
                logger.info('Segment %d, computing %d eigenvectors out of %d',
                            i, ncomp, x.shape[1])
            else:
                ncomp = None

            self.models.append(self.pca_class(n_components=ncomp).fit(x))

    def chooseevals(self, nevals=[]):
        """Choose the number of eigenspectra/evals to use for reconstruction.

        User supplies the number of eigen spectra to be used (neval) or the
        percentage of the eigenspectra that were calculated (peval) from each
        spectral segment to be used.

        The user can either provide a single value to be used for all segments,
        or provide an array that defines neval or peval per segment.

        """
        nranges = len(self.pranges)
        nevals = np.atleast_1d(nevals)

        # deal with an input list
        if len(nevals) > 1:
            if len(nevals) != nranges:
                nevals = np.array([nevals[0]])
                logger.info('Chosen eigenspectra array does not correspond to '
                            'number of segments')
            else:
                logger.info('Choosing %s eigenspectra for segments', nevals)

        if len(nevals) == 1:
            logger.info('Choosing %s eigenspectra for all segments', nevals)
            nevals = np.zeros(nranges, dtype=int) + nevals

        if nevals.ndim == 1:
            start = np.zeros(nranges, dtype=int)
            end = nevals
        else:
            start, end = nevals.T

        self.nevals = nevals
        for i, model in enumerate(self.models):
            model.components_ = self.components[i][start[i]:end[i]]

    @timeit
    def reconstruct(self):
        """Reconstruct the residuals from a given set of eigenspectra and
        eigenvalues
        """
        logger.info('Reconstructing Sky Residuals')
        indices = [x[0] for x in self.pranges[1:]]
        # normstack = self.stack - self.contarray
        Xarr = np.array_split(self.normstack.T, indices, axis=1)
        Xnew = [model.transform(x)
                for model, x in zip(self.models, Xarr)]
        Xinv = [model.inverse_transform(x)
                for model, x in zip(self.models, Xnew)]
        self.recon = np.concatenate([x.T * self.variancearray[i, :]
                                     for i, x in enumerate(Xinv)])
        # self.recon = np.concatenate([x.T for x in Xinv])
        # self.recon *= self.variancearray[:, np.newaxis]

    def make_cube_from_stack(self, stack, with_nans=False):
        """Stuff the stack back into a cube."""
        cube = self.cube.copy()
        cube[:, self.y, self.x] = stack
        if with_nans:
            cube[self.nancube] = np.nan
        if self.ins_mode in NOTCH_FILTER_RANGES:
            lmin, lmax = self.notch_limits
            cube[lmin:lmax + 1] = np.nan
        return cube

    def remold(self):
        """ Subtracts the reconstructed residuals and places the cleaned
        spectra into the duplicated datacube.
        """
        logger.info('Applying correction and reshaping data product')
        self.cleancube = self.make_cube_from_stack(self.stack - self.recon,
                                                   with_nans=self.run_clean)

    def reprocess(self, nevals=[]):
        """ A method that redoes the eigenvalue selection, reconstruction, and
        remolding of the data.
        """
        self.chooseevals(nevals=nevals)
        self.reconstruct()
        self.remold()

    def optimize(self):
        """Compute the optimal number of components needed to characterize
        the residuals.

        This function calculates the variance per segment with an increasing
        number of eigenspectra/eigenvalues. It then deterimines the point at
        which the second derivative of this variance curve reaches zero. When
        this occurs, the linear reduction in variance is attributable to the
        removal of astronomical features rather than emission line residuals.

        """
        logger.info('Compute number of components')
        ncomp = []
        for model in self.models:
            var = model.explained_variance_
            deriv, mn1, std1 = _compute_deriv(var)
            cross = np.append([False], deriv >= (mn1 - std1))
            ncomp.append(np.where(cross)[0][0])

        self.nevals = np.array(ncomp)

    def make_contcube(self):
        """ Remold the continuum array so it can be investigated.

        Takes the continuum stack and returns it into a familiar cube form.
        """
        contcube = self.cube.copy() * np.nan
        contcube[:, self.y, self.x] = self.contarray
        return contcube

    def _applymask(self, mask):
        """Apply a mask to the input data to provide a cleaner basis set.

        mask is >1 for objects, 0 for sky so that people can use sextractor.
        The file is read with ``astropy.io.fits.getdata`` which first tries to
        read the primary extension, then the first extension is no data was
        found before.

        """
        logger.info('Applying Mask for SVD Calculation from %s', mask)
        self.maskfile = mask
        mask = fits.getdata(mask).astype(bool)
        nmasked = np.count_nonzero(mask)
        logger.info('Masking %d pixels (%d%%)', nmasked,
                    nmasked / np.prod(mask.shape) * 100)
        self.cube[:, mask] = np.nan

    def writecube(self, outcubefits='DATACUBE_ZAP.fits', overwrite=False):
        """Write the processed datacube to an individual fits file."""
        outhead = _newheader(self)
        outhdu = fits.PrimaryHDU(data=self.cleancube, header=outhead)
        outhdu.writeto(outcubefits, overwrite=overwrite)
        logger.info('Cube file saved to %s', outcubefits)

    def writeskycube(self, skycubefits='SKYCUBE_ZAP.fits', overwrite=False):
        """Write the processed datacube to an individual fits file."""
        outcube = self.cube - self.cleancube
        outhead = _newheader(self)
        outhdu = fits.PrimaryHDU(data=outcube, header=outhead)
        outhdu.writeto(skycubefits, overwrite=overwrite)
        logger.info('Sky cube file saved to %s', skycubefits)

    def writevarcurve(self, varcurvefits='VARCURVE_ZAP.fits', overwrite=False):
        """Write the explained variance curves to an individual fits file."""
        from astropy.table import Table
        table = Table([m.explained_variance_ for m in self.models])
        hdu = fits.table_to_hdu(table)
        _newheader(self, hdu.header)
        hdu.writeto(varcurvefits, overwrite=overwrite)
        logger.info('Variance curve file saved to %s', varcurvefits)

    def mergefits(self, outcubefits, overwrite=False):
        """Merge the ZAP cube into the full muse datacube and write."""
        # make sure it has the right extension
        outcubefits = outcubefits.split('.fits')[0] + '.fits'
        with fits.open(self.cubefits) as hdu:
            if self.instrument == 'MUSE':
                hdu[1].header = _newheader(self)
                hdu[1].data = self.cleancube
            elif self.instrument == 'KCWI':
                hdu[0].header = _newheader(self)
                hdu[0].data = self.cleancube
            elif self.instrument == 'FOCAS':
                hdu[0].header = _newheader(self)
                hdu[0].data = self.cleancube
            else:
                raise ValueError('unsupported instrument %s' % self.instrument)

            hdu.writeto(outcubefits, overwrite=overwrite)
        logger.info('Cube file saved to %s', outcubefits)

    def plotvarcurve(self, i=0, ax=None):
        var = self.models[i].explained_variance_
        deriv, mn1, std1 = _compute_deriv(var)

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=[10, 15])

        ax1, ax2, ax3 = ax
        ax1.plot(var, linewidth=3)
        ax1.plot([self.nevals[i], self.nevals[i]], [min(var), max(var)])
        ax1.set_ylabel('Variance')

        ax2.plot(np.arange(deriv.size), deriv)
        ax2.hlines([mn1, mn1 - std1], 0, len(deriv), colors=('k', '0.5'))
        ax2.plot([self.nevals[i] - 1, self.nevals[i] - 1],
                 [min(deriv), max(deriv)])
        ax2.set_ylabel('d/dn Var')

        deriv2 = np.diff(deriv)
        ax3.plot(np.arange(deriv2.size), np.abs(deriv2))
        ax3.plot([self.nevals[i] - 2, self.nevals[i] - 2],
                 [min(deriv2), max(deriv2)])
        ax3.set_ylabel('(d^2/dn^2) Var')
        # ax3.set_xlabel('Number of Components')

        ax1.set_title('Segment {0}, {1} - {2} Angstroms'.format(
            i, self.lranges[i][0], self.lranges[i][1]))

    def plotvarcurves(self):
        nseg = len(self.models)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nseg, 3, figsize=(16, nseg * 2),
                                 tight_layout=True)
        for i in range(nseg):
            self.plotvarcurve(i=i, ax=axes[i])


# ================= Helper Functions =================

def worker(f, i, chunk, out_q, err_q, kwargs):
    try:
        result = f(i, chunk, **kwargs)
    except Exception as e:
        err_q.put(e)
        return

    # output the result and task ID to output queue
    out_q.put((i, result))


def parallel_map(func, arr, indices, **kwargs):
    logger.debug('Running function %s with %s chunks', func.__name__, indices)
    axis = kwargs.pop('axis', None)
    if isinstance(indices, (int, np.integer)) and indices == 1:
        return [func(0, arr, **kwargs)]

    manager = Manager()
    out_q = manager.Queue()
    err_q = manager.Queue()
    jobs = []
    chunks = np.array_split(arr, indices, axis=axis)
    if 'split_arrays' in kwargs:
        split_arrays = [np.array_split(a, indices, axis=axis)
                        for a in kwargs.pop('split_arrays')]
    else:
        split_arrays = None

    for i, chunk in enumerate(chunks):
        if split_arrays:
            kwargs['split_arrays'] = [s[i] for s in split_arrays]
        p = Process(target=worker, args=(func, i, chunk, out_q, err_q, kwargs))
        jobs.append(p)
        p.start()

    # gather the results
    for proc in jobs:
        proc.join()

    if not err_q.empty():
        # kill all on any exception from any one slave
        raise err_q.get()

    # Processes finish in arbitrary order. Process IDs double
    # as index in the resultant array.
    results = [None] * len(jobs)
    while not out_q.empty():
        idx, result = out_q.get()
        results[idx] = result

    return results


def _compute_deriv(arr, nsigma=5):
    """Compute statistics on the derivatives"""
    npix = int(0.25 * arr.shape[0])
    deriv = np.diff(arr[:npix])
    ind = int(.15 * deriv.size)
    mn1 = deriv[ind:].mean()
    std1 = deriv[ind:].std() * nsigma
    return deriv, mn1, std1


def _continuumfilter(stack, cftype, cfwidth=300, notch_limits=None):
    if cftype == 'fit':
        x = np.arange(stack.shape[0])

        # Excluding the very red part for the fit. This is Muse-specific,
        # but anyway for another instrument this method should probably
        # not be used as is.
        w = np.ones(stack.shape[0])
        w[3600:] = 0

        if notch_limits is not None:
            # Exclude the notch filter region
            lmin, lmax = notch_limits
            w[lmin:lmax + 1] = 0

        res = np.polynomial.polynomial.polyfit(x, stack, deg=5, w=w)
        ret = np.polynomial.polynomial.polyval(x, res, tensor=True)
        return ret.T

    if cftype == 'median':
        func = _icfmedian
    else:
        raise ValueError('unknown cftype option')

    logger.info('Using cfwidth=%d', cfwidth)

    if notch_limits is not None:
        # To manage the notch filter which is filled with zeros, we process the
        # stack in two halves, before and after the filter.
        c = np.zeros_like(stack)
        ctmp = parallel_map(func, stack[:notch_limits[0]], NCPU, axis=1,
                            cfwidth=cfwidth)
        c[:notch_limits[0]] = np.concatenate(ctmp, axis=1)
        ctmp = parallel_map(func, stack[notch_limits[1]:], NCPU, axis=1,
                            cfwidth=cfwidth)
        c[notch_limits[1]:] = np.concatenate(ctmp, axis=1)
    else:
        c = parallel_map(func, stack, NCPU, axis=1, cfwidth=cfwidth)
        c = np.concatenate(c, axis=1)

    return c


def _icfmedian(i, stack, cfwidth=None):
    ufilt = 3  # set this to help with extreme over/under corrections
    return ndi.median_filter(
        ndi.uniform_filter(stack, (ufilt, 1)), (cfwidth, 1))


def rolling_window(a, window):  # function for striding to help speed up
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _newheader(zobj, header=None):
    """Put the pertinent zap parameters into the header"""
    header = header or zobj.header.copy()
    header['COMMENT'] = 'These data have been ZAPped!'
    header.append(('ZAPvers', __version__, 'ZAP version'), end=True)
    # zlevel removal performed
    header.append(('ZAPzlvl', zobj.run_zlevel, 'ZAP zero level correction'))
    # Nanclean performed
    header['ZAPclean'] = (zobj.run_clean,
                          'ZAP NaN cleaning performed for calculation')
    # Continuum Filtering
    header['ZAPcftyp'] = (zobj._cftype, 'ZAP continuum filter type')
    header['ZAPcfwid'] = (zobj._cfwidth, 'ZAP continuum filter size')

    # number of segments
    nseg = len(zobj.pranges)
    header['ZAPnseg'] = (nseg, 'Number of segments used for ZAP SVD')

    # per segment variables
    if hasattr(zobj, 'nevals'):
        for i in range(nseg):
            header['ZAPseg{}'.format(i)] = (
                '{}:{}'.format(zobj.pranges[i][0], zobj.pranges[i][1] - 1),
                'spectrum segment (pixels)')
            header['ZAPnev{}'.format(i)] = (zobj.nevals[i],
                                            'number of eigenvals/spectra used')

    return header


def _isigclip(i, istack):
    mn = []
    for col in istack:
        clipped, bot, top = sigmaclip(col, low=3, high=3)
        mn.append(clipped.mean())
    return np.array(mn)


def _imedian(i, istack):
    return np.median(istack, axis=1)


@timeit
def _nanclean(cube, rejectratio=0.25, boxsz=1):
    """
    Detects NaN values in cube and removes them by replacing them with an
    interpolation of the nearest neighbors in the data cube. The positions in
    the cube are retained in nancube for later remasking.

    """
    logger.info('Cleaning NaN values in the cube')
    cleancube = cube.copy()
    badcube = np.logical_not(np.isfinite(cleancube))        # find NaNs
    badmap = badcube.sum(axis=0)  # map of total nans in a spaxel

    # choose some maximum number of bad pixels in the spaxel and extract
    # positions
    badmask = badmap > (rejectratio * cleancube.shape[0])
    logger.info('Rejected %d spaxels with more than %.1f%% NaN pixels',
                np.count_nonzero(badmask), rejectratio * 100)

    # make cube mask of bad spaxels
    badcube &= (~badmask[np.newaxis, :, :])
    z, y, x = np.where(badcube)

    neighbor = np.zeros((z.size, (2 * boxsz + 1)**3))
    icounter = 0
    logger.info("Fixing %d remaining NaN pixels", len(z))

    # loop over samplecubes
    nz, ny, nx = cleancube.shape
    for j in range(-boxsz, boxsz + 1, 1):
        for k in range(-boxsz, boxsz + 1, 1):
            for l in range(-boxsz, boxsz + 1, 1):
                iz, iy, ix = z + l, y + k, x + j
                outsider = ((ix <= 0) | (ix >= nx - 1) |
                            (iy <= 0) | (iy >= ny - 1) |
                            (iz <= 0) | (iz >= nz - 1))
                ins = ~outsider
                neighbor[ins, icounter] = cleancube[iz[ins], iy[ins], ix[ins]]
                neighbor[outsider, icounter] = np.nan
                icounter = icounter + 1

    cleancube[z, y, x] = np.nanmean(neighbor, axis=1)
    return cleancube, badcube
