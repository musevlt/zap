================================
Welcome to ZAP's documentation!
================================

Tired of sky subtraction residuals? ZAP them!

ZAP (the *Zurich Atmosphere Purge*) is a high precision sky subtraction tool
which can be used as complete sky subtraction solution, or as an enhancement to
previously sky-subtracted MUSE integral field spectroscopic data.  The method
uses PCA to isolate the residual sky subtraction features and remove them from
the observed datacube. Though the operation of ZAP is not dependent on perfect
flat-fielding of the data in a MUSE exposure, better results are obtained when
these corrections are made ahead of time. Future development will include
expansion to more instruments.

.. note::

    Version 2.0 is compatible with the WFM-AO mode, and also brings major
    improvements for the sky subtraction. Check below the details in the
    :ref:`changelog` section as well as the dicussion on the
    :ref:`eigenvectors-number`.

    Version 2.1 brings compatibility with the NFM-AO mode.

The paper describing the original method can be found here:
http://adsabs.harvard.edu/abs/2016MNRAS.458.3210S

Please cite ZAP as::

    \bibitem[Soto et al.(2016)]{2016MNRAS.458.3210S}
        Soto, K.~T., Lilly, S.~J., Bacon, R., Richard, J., \& Conseil, S.
        2016, \mnras, 458, 3210

.. contents::

Installation
============

ZAP requires the following packages:

* Numpy
* Astropy
* SciPy (0.18.1 or later is recommend as a SVD convergence issue was found with
  an older version)
* Scikit-learn

Many linear algebra operations are performed in ZAP, so it can be beneficial to
use an alternative BLAS package. In the Anaconda distribution, the default BLAS
comes with Numpy linked to MKL, which can amount to a 20% speedup of ZAP.

The last stable release of ZAP can be installed simply with pip::

    pip install zap

Or into the user path with::

    pip install --user zap

Usage
=====

In its most hands-off form, ZAP can take an input FITS datacube, operate on it,
and output a final FITS datacube. The main function to do this is
`zap.process`::

    import zap
    zap.process('INPUT.fits', outcubefits='OUTPUT.fits')

Care should be taken, however, since this case assumes a sparse field, and
better results can be obtained by applying masks.

There are a number of options that can be passed to the code which we describe
below.

Sparse Field Case
-----------------

This case specifically refers to the case where the sky can be measured in the
sky frame itself, using::

    zap.process('INPUT.fits', outcubefits='OUTPUT.fits')

In both cases, the code will create a resulting processed datacube named
``DATACUBE_ZAP.fits`` in the current directory. While this can work well in the
case of very faint sources, masks can improve the results.

For the sparse field case, a mask file can be included, which is a 2D FITS
image matching the spatial dimensions of the input datacube. Masks are defined
to be >= 1 on astronomical sources and 0 at the position of the sky. Set this
parameter with the ``mask`` keyword ::

    zap.process('INPUT.fits', outcubefits='OUTPUT.fits', mask='mask.fits')

Filled Field Case
-----------------

This approach also can address the saturated field case and is robust in the
case of strong emission lines, in this case the input is an offset sky
observation. To achieve this, we calculate the SVD on an external sky frame
using the function `zap.SVDoutput`.

An example of running the code in this way is as follows::

    extSVD = zap.SVDoutput('Offset_Field_CUBE.fits', mask='mask.fits')
    zap.process('Source_cube.fits', outcubefits='OUTPUT.fits', extSVD=extSVD)

The integration time of this frame does not need to be the same as the object
exposure, but rather just a 2-3 minute exposure.

.. _eigenvectors-number:

Optimal number of eigenvectors
------------------------------

The major difficulty to get a high quality sky subtraction is to find the
optimal number of eigenvalues to use. ZAP provides an automated way for this,
trying to find the inflexion point of the variance curve. This is one way to do
it, but there is no right answer to this issue. A higher number of eigenvalues
used for the reconstruction will give a better sky subtraction, but with the
risk of subtracting signal from strong emission lines.

The first thing one can do to optimize the PCA quality is to use a good mask, to
avoid incorporating signal from astronomical sources in the eigenvectors. Then
it is highly recommended to have a look at the explained variance curves (which
can be saved with the ``varcurvefits`` parameter) and the selected number of
eigenvalues (saved in the FITS headers in ``ZAPNEV*``). It is also possible to
use the interactive mode (see below) to try different number of eigenvectors.
This number can be specified manually with the ``neval`` parameter.

Strong values at edges
----------------------

Because of atmospheric refraction the cube edges are different depending on the
wavelength, which means that the spectra at the edges contain many NaN values.
ZAP filters out these spaxels (when the spectra have more than 25% of NaN
values), because it cannot process incomplete spectra. So these spectra are not
sky-subtracted and appear with a stronger flux in the output cube or image.

The `zap.mask_nan_edges` function allows to mask these spectra, detecting the
ones with too many NaNs, and replacing them with NaNs.

Command Line Interface
======================

ZAP can also be used from the command line::

    python -m zap INPUT_CUBE.fits

More information use of the command line interface can be found with the
command ::

    python -m zap -h


Interactive mode
================

ZAP can be used interactively from the Python console::

    import zap
    zobj = zap.process('INPUT.fits', interactive=True)

The run method operates on the datacube, and retains all of the data and methods
necessary to process a final data cube in a Python class named `~zap.Zap`. You
can elect to investigate the data product via the `~zap.Zap` object, and even
reprocess the cube with a different number of eigenspectra per region.
A workflow may go as follows:

.. code-block:: python

  import zap
  from matplotlib import pyplot as plt

  # allow ZAP to run the optimize routine
  zobj = zap.process('INPUT.fits', interactive=True)

  # plot the variance curves and the selection of the number of eigenspectra used
  zobj.plotvarcurve()

  # plot a spectrum extracted from the original cube
  plt.figure()
  plt.plot(zobj.cube[:,50:100,50:100].sum(axis=(1,2)), 'b', alpha=0.3)

  # plot a spectrum of the cleaned ZAP dataproduct
  plt.plot(zobj.cleancube[:,50:100,50:100].sum(axis=(1,2)), 'g')

  # choose just the first 3 spectra for all segments
  zobj.reprocess(nevals=3)

  # plot a spectrum extracted from the original cube
  plt.plot(zobj.cube[:,50:100,50:100].sum(axis=(1,2)), 'b', alpha=0.3)

  # plot a spectrum of the cleaned ZAP dataproduct
  plt.plot(zobj.cleancube[:,50:100,50:100].sum(axis=(1,2))), 'g')

  # choose some number of modes by hand
  zobj.reprocess(nevals=[2,5,2,4,6,7,9,8,5,3,5])

  # plot a spectrum
  plt.plot(zobj.cleancube[:,50:100,50:100].sum(axis=(1,2))), 'k')

  # Use the optimization algorithm to identify the best number of modes per segment
  zobj.optimize()

  # compare to the previous versions
  plt.plot(zobj.cleancube[:,50:100,50:100].sum(axis=(1,2))), 'r')

  # identify a pixel in the dispersion axis that shows a residual feature in
  # the original
  plt.figure()
  plt.matshow(zobj.cube[2903,:,:])

  # compare this to the zap dataproduct
  plt.figure()
  plt.matshow(zobj.cleancube[2903,:,:])

  # write the processed cube as a single extension FITS
  zobj.writecube('DATACUBE_ZAP.fits')

  # or merge the zap datacube into the original input datacube, replacing the
  # data extension
  zobj.mergefits(outcubefits='DATACUBE_FINAL_ZAP.fits')

.. _changelog:

Changelog
=========

.. include:: ../CHANGELOG

API
===

.. autofunction:: zap.process

.. autofunction:: zap.SVDoutput

.. autofunction:: zap.nancleanfits

.. autofunction:: zap.contsubfits

.. autofunction:: zap.mask_nan_edges

.. autoclass:: zap.Zap
   :members:
