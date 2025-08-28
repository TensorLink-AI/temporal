temporal.utils.utils
====================

.. py:module:: temporal.utils.utils


Functions
---------

.. autoapisummary::

   temporal.utils.utils.mock_builder
   temporal.utils.utils.test_patch_transform_block_init
   temporal.utils.utils.test_patch_transform_block_forward_split_first
   temporal.utils.utils.test_patch_transform_block_forward_merge_first
   temporal.utils.utils.test_patch_transform_block_invalid_order
   temporal.utils.utils.test_patch_transform_block_merge_first_invalid_expansion
   temporal.utils.utils.test_patch_transform_block_non_divisible_d_model_split_first


Module Contents
---------------

.. py:function:: mock_builder()

   Creates a mock ModuleBuilder with a basic config.


.. py:function:: test_patch_transform_block_init(mock_builder, order)

   Tests the initialization of PatchTransformBlock for both orders.


.. py:function:: test_patch_transform_block_forward_split_first(mock_builder)

   Tests the forward pass of PatchTransformBlock with order='split_first'.


.. py:function:: test_patch_transform_block_forward_merge_first(mock_builder)

   Tests the forward pass of PatchTransformBlock with order='merge_first'.


.. py:function:: test_patch_transform_block_invalid_order(mock_builder)

   Tests that PatchTransformBlock raises an error for an invalid order.


.. py:function:: test_patch_transform_block_merge_first_invalid_expansion(mock_builder)

   Tests that 'merge_first' order raises an error with expansion_factor != 2.


.. py:function:: test_patch_transform_block_non_divisible_d_model_split_first(mock_builder)

   Tests that 'split_first' order raises an error if d_model is not divisible by expansion_factor.


