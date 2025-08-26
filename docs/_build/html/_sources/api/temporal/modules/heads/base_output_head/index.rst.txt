temporal.modules.heads.base_output_head
=======================================

.. py:module:: temporal.modules.heads.base_output_head


Classes
-------

.. autoapisummary::

   temporal.modules.heads.base_output_head.BaseOutputHead


Module Contents
---------------

.. py:class:: BaseOutputHead(*args, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   An abstract base class for all model output heads.

   This class defines the common interface that all output head modules must
   adhere to. An output head is responsible for taking the final hidden state
   from the model's backbone and transforming it into the desired output format
   (e.g., a point forecast, a probability distribution).

   It also defines a method for retrieving the appropriate loss function
   to be used with the head's output.


   .. py:method:: forward(hidden_state: torch.Tensor) -> torch.Tensor
      :abstractmethod:


      Processes the model's final hidden state to produce the output.

      This method must be implemented by all subclasses.

      :param hidden_state: The final hidden state from the model's
                           backbone, typically of shape `[batch_size, seq_len, d_model]`.
      :type hidden_state: torch.Tensor

      :returns:

                The model's final output, with its shape and meaning
                    determined by the specific head implementation.
      :rtype: torch.Tensor



   .. py:method:: get_loss_fn() -> Optional[Callable]
      :abstractmethod:


      Returns the default loss function associated with this head.

      This method should be implemented by subclasses to provide a suitable
      loss function for the type of output they produce. For example, a
      point forecast head might return Mean Squared Error, while a
      probabilistic head might return Negative Log-Likelihood.

      :returns: A callable loss function, or None if the head
                does not have a default loss.
      :rtype: Optional[Callable]



