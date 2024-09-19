kirby.nn
====================

.. contents:: Contents
    :local:


Embeddings Layers
-----------------
.. currentmodule:: kirby.nn

.. list-table::
   :widths: 25 75

   * - :py:class:`Embedding`
     - A simple embedding layer.
   * - :py:class:`InfiniteVocabEmbedding`
     - An extendable embedding layer + tokenizer.

.. autoclass:: Embedding
    :members:
    :undoc-members:

.. autoclass:: InfiniteVocabEmbedding
    :members:
    :undoc-members:
    :exclude-members: forward, extra_repr, initialize_parameters


Rotary modules
--------------
.. currentmodule:: kirby.nn

.. list-table::
   :widths: 25 75

   * - :py:class:`RotaryEmbedding`
     - Rotary embedding layer.
   * - :py:class:`RotaryCrossAttention`
     - Rotary cross-attention layer.
   * - :py:class:`RotarySelfAttention`
     - Rotary self-attention layer.
   * - :py:class:`PerceiverRotary`
     - PerceiverIO with rotary embeddings.

.. autoclass:: RotaryEmbedding
    :members:
    :undoc-members:

.. autofunction:: apply_rotary_pos_emb

.. autoclass:: RotaryCrossAttention
    :members:
    :undoc-members: 

.. autoclass:: RotarySelfAttention
    :members:
    :undoc-members:

.. autoclass:: PerceiverRotary
    :members:
    :undoc-members:
    

Readout Layers
--------------

.. currentmodule:: kirby.nn

.. list-table::
   :widths: 25 75

   * - :py:func:`compute_loss_or_metric`
     - Helper function to compute various losses and metrics.
   * - :py:class:`MultitaskReadout`
     - A multi-task readout module.
   * - :py:func:`prepare_for_multitask_readout`
     - Tokenizer function for :py:class:`MultitaskReadout`.
   * - :py:func:`extract_request_keys_from_decoder_registry`
     - Helper function to extract request keys from decoder registry.

.. autofunction:: compute_loss_or_metric

.. autoclass:: MultitaskReadout
    :members:
    :show-inheritance:
    :undoc-members:

.. autofunction:: prepare_for_multitask_readout

.. autofunction:: extract_request_keys_from_decoder_registry