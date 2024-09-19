kirby.data
====================

.. contents:: Contents
    :local:


Data Objects
------------
.. currentmodule:: kirby.data

.. list-table::
   :widths: 25 75

   * - :py:class:`ArrayDict`
     - A class representing an array dictionary.
   * - :py:class:`IrregularTimeSeries`
     - A class representing an irregular time series.
   * - :py:class:`RegularTimeSeries`
     - A class representing a regular time series.
   * - :py:class:`Interval`
     - A class representing an interval.
   * - :py:class:`Data`
     - A class representing data objects.

.. autoclass:: ArrayDict
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: IrregularTimeSeries
    :inherited-members:
    :show-inheritance:
    :undoc-members:


.. autoclass:: RegularTimeSeries
    :inherited-members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: Interval
    :inherited-members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: Data
    :inherited-members:
    :show-inheritance:
    :undoc-members:


Dataset Builder
---------------

.. currentmodule:: kirby.data.dataset_builder

.. autoclass:: DatasetBuilder
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: SessionContextManager
    :members:
    :show-inheritance:
    :undoc-members:


Dataset
-------

.. currentmodule:: kirby.data.dataset

.. autoclass:: Dataset
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: SessionFileInfo
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: DatasetIndex
    :members:
    :show-inheritance:
    :undoc-members:

Collate
-------
.. currentmodule:: kirby.data.collate

.. list-table::
   :widths: 25 75

   * - :py:class:`collate`
     - An extended collate function that handles padding and chaining.
   * - :py:class:`pad`
     - A wrapper to call when padding.
   * - :py:class:`track_mask`
     - A wrapper to call to track the padding mask during padding.
   * - :py:class:`pad8`
     - A wrapper to call when padding, but length is rounded up to the nearest multiple of 8. 
   * - :py:class:`track_mask8`
     - A wrapper to call to track the padding mask during padding with :py:class:`pad8`.
   * - :py:class:`chain`
     - A wrapper to call when chaining.
   * - :py:class:`track_batch`
     - A wrapper to call to track the batch index during chaining.


.. autofunction:: collate

.. autofunction:: pad

.. autofunction:: track_mask

.. autofunction:: pad8

.. autofunction:: trach_mask8

.. autofunction:: chain

.. autofunction:: track_batch

Samplers
--------
.. currentmodule:: kirby.data.sampler

.. list-table::
   :widths: 25 75

   * - :py:class:`SequentialFixedWindowSampler`
     - A Sequential sampler, that samples a fixed-length window from data.
   * - :py:class:`RandomFixedWindowSampler`
     - A Random sampler, that samples a fixed-length window from data.


.. autoclass:: SequentialFixedWindowSampler
  :members:
  :show-inheritance:
  :undoc-members:

.. autoclass:: RandomFixedWindowSampler
  :members:
  :show-inheritance:
  :undoc-members:
