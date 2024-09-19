kirby.transforms
====================

.. currentmodule:: kirby.transforms

.. list-table::
   :widths: 25 75

   * - :py:class:`Compose`
     - Compose several transforms together.
   * - :py:class:`UnitDropout`
     - Randomly drop units from the `data.units` and `data.spikes`.
   * - :py:class:`RandomTimeScaling`
     - Randomly scales the time axis.
   * - :py:class:`RandomOutputSampler`
     - Randomly drops output samples.


.. autoclass:: Compose
    :members:
    :show-inheritance:
    :undoc-members:


.. autoclass:: TriangleDistribution
    :members:
    :show-inheritance:
    :undoc-members:


.. autoclass:: UnitDropout
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: RandomTimeScaling
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: RandomOutputSampler
    :members:
    :show-inheritance:
    :undoc-members:
