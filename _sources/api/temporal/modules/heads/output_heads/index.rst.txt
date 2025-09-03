temporal.modules.heads.output_heads
===================================

.. py:module:: temporal.modules.heads.output_heads


Classes
-------

.. autoapisummary::

   temporal.modules.heads.output_heads.LinearOutputHead
   temporal.modules.heads.output_heads.GaussianHead
   temporal.modules.heads.output_heads.QuantileRegressionOutputHead
   temporal.modules.heads.output_heads.DistPredHead
   temporal.modules.heads.output_heads.MixtureOutputHead
   temporal.modules.heads.output_heads.StudentTHead


Module Contents
---------------

.. py:class:: LinearOutputHead(hidden_size: int, output_size: int = 1, **kwargs)

   Bases: :py:obj:`temporal.modules.heads.base_output_head.BaseOutputHead`


   Simple linear projection head for point forecasts.


   .. py:attribute:: proj


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Processes the model's final hidden state to produce the output.

      This method must be implemented by all subclasses.

      :param hidden_state: The final hidden state from the model's
                           backbone, typically of shape `[batch_size, seq_len, d_model]`.
      :type hidden_state: torch.Tensor

      :returns:

                The model's final output, with its shape and meaning
                    determined by the specific head implementation.
      :rtype: torch.Tensor



   .. py:method:: sample(y_hat: torch.Tensor, *, temperature: float = 1.2, jitter_std: float = 0.05, min_abs_jitter: float = 0.001, eta_blend_mean: float = 0.85) -> torch.Tensor


   .. py:method:: get_loss_fn() -> Optional[Callable]

      Returns the default loss function associated with this head.

      This method should be implemented by subclasses to provide a suitable
      loss function for the type of output they produce. For example, a
      point forecast head might return Mean Squared Error, while a
      probabilistic head might return Negative Log-Likelihood.

      :returns: A callable loss function, or None if the head
                does not have a default loss.
      :rtype: Optional[Callable]



.. py:class:: GaussianHead(hidden_size: int, output_size: int = 1, *, min_log_sigma: float = -7.0, max_log_sigma: float = 5.0, sigma_floor: float = 0.0001, init_log_sigma: Optional[float] = None, allow_lazy_infer: bool = True, **kwargs)

   Bases: :py:obj:`temporal.modules.heads.base_output_head.BaseOutputHead`


   Gaussian output head.
   - forward: returns concatenated [mu, log_sigma] with shape [B, T, 2F]
   - predict("mean"|"median"): returns [B, T, F]
   - sample(...): reparam sampling, returns [B, T, F] (or [B, 1, F] if T==1)
   - sample_quantiles(qs): returns [B, T, F, Q]


   .. py:attribute:: feature_size
      :value: 1



   .. py:attribute:: min_log_sigma


   .. py:attribute:: max_log_sigma


   .. py:attribute:: sigma_floor


   .. py:attribute:: allow_lazy_infer
      :value: True



   .. py:attribute:: proj


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      x: [B, T, hidden_size]
      returns: [B, T, 2F]  (concat of mu and log_sigma)



   .. py:method:: predict(params: torch.Tensor, method: str = 'mean') -> torch.Tensor


   .. py:method:: sample(params: torch.Tensor, *, method: str = 'reparam', temperature: float = 1.0, min_std: Optional[float] = None, eta_blend_mean: float = 0.0, generator: Optional[torch.Generator] = None) -> torch.Tensor


   .. py:method:: sample_quantiles(params: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor


.. py:class:: QuantileRegressionOutputHead(hidden_size: int, output_size: int, num_quantiles: int, feature_size: int = 1, **kwargs)

   Bases: :py:obj:`temporal.modules.heads.base_output_head.BaseOutputHead`


   Multi-quantile regression head.


   .. py:attribute:: num_quantiles


   .. py:attribute:: feature_size
      :value: 1



   .. py:attribute:: proj


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Processes the model's final hidden state to produce the output.

      This method must be implemented by all subclasses.

      :param hidden_state: The final hidden state from the model's
                           backbone, typically of shape `[batch_size, seq_len, d_model]`.
      :type hidden_state: torch.Tensor

      :returns:

                The model's final output, with its shape and meaning
                    determined by the specific head implementation.
      :rtype: torch.Tensor



   .. py:method:: sample(q_pred: torch.Tensor, *, temperature: float = 1.0, jitter_u: float = 0.05, enforce_monotone: bool = True, tail_extrapolation: str = 'hold', eta_blend_median: float = 0.3) -> torch.Tensor


   .. py:method:: predict(x: torch.Tensor, method: str = 'median') -> torch.Tensor


   .. py:method:: get_loss_fn() -> Optional[Callable]

      Returns the default loss function associated with this head.

      This method should be implemented by subclasses to provide a suitable
      loss function for the type of output they produce. For example, a
      point forecast head might return Mean Squared Error, while a
      probabilistic head might return Negative Log-Likelihood.

      :returns: A callable loss function, or None if the head
                does not have a default loss.
      :rtype: Optional[Callable]



.. py:class:: DistPredHead(hidden_size: int, output_size: int, **kwargs)

   Bases: :py:obj:`temporal.modules.heads.base_output_head.BaseOutputHead`


   DistPred outputs K candidate paths per feature; we sample a path per step.

   forward(...) returns a dict:
       {
         "paths": [B, T, C, K]   (or [B, T, 1, K] if univariate),
         "path_logits": [B, T, K]   # optional, if with_path_logits=True
       }

   .sample(...) supports sticky or mixture selection and returns:
       (y_next, new_state)
       where y_next is [B, 1, C] and new_state holds sticky path indices.


   .. py:attribute:: num_outputs


   .. py:attribute:: feature_size


   .. py:attribute:: proj


   .. py:attribute:: use_tanh


   .. py:attribute:: tanh_scale


   .. py:attribute:: with_path_logits


   .. py:attribute:: state_key_default


   .. py:method:: forward(x: torch.Tensor) -> Dict[str, torch.Tensor]

      x: [B, T, H] -> returns dict with:
        paths: [B, T, C, K]
        path_logits: [B, T, K] (optional)



   .. py:method:: predict(x: Union[torch.Tensor, Dict[str, torch.Tensor]], method: Union[str, float, int] = 'median') -> torch.Tensor

      Collapse [B,T,C,K] (or [B,T,K]) over K. Accepts dict from forward().
      Returns [B,T,C] (or [B,T,1] for univariate).



   .. py:method:: sample_quantiles(x: Union[torch.Tensor, Dict[str, torch.Tensor]], quantile_levels: List[float]) -> torch.Tensor

      Empirical quantiles from ensemble.
      Accepts tensor or dict from forward().
      Returns: [B,T,C,Q] (univariate -> C=1)



   .. py:method:: sample(head_out: Dict[str, torch.Tensor], *, state: Optional[Dict[str, torch.Tensor]] = None, temperature: float = 1.0, top_p: Optional[float] = 0.9, stickiness: float = 0.9, reselection_hazard: Optional[float] = 0.02, mode: str = 'sticky', mixture_sharpness: float = 1.0, dirichlet_alpha: Optional[float] = None, jitter_rel_std: float = 0.0, jitter_min_abs: float = 0.001, state_key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]

      One AR step. Expects T=1 in head_out tensors.
      :returns: [B,1,C]
                new_state: dict with updated path indices
      :rtype: y_next



.. py:class:: MixtureOutputHead(hidden_size: int, components: List[str], **kwargs)

   Bases: :py:obj:`temporal.modules.heads.base_output_head.BaseOutputHead`


   Univariate Mixture Density head (heterogeneous components supported).

   forward(...) -> dict with keys:
       {
         'components': List[str]             # canonical names, e.g. ['normal','student_t']
         'mixture_logits': [B,T,M]           # M = number of components
         '<param_keys>': tensors per component param (see DIST_OUTPUT_KEYS)
       }

   predict(...)          -> mixture mean [B,T,1]
   sample(...)           -> one-step sample [B,1,1]
   sample_quantiles(...) -> MC quantiles   [B,T,1,Q]


   .. py:attribute:: DIST_PARAM_COUNTS


   .. py:attribute:: DIST_OUTPUT_KEYS


   .. py:attribute:: hidden_size


   .. py:attribute:: components_raw


   .. py:attribute:: components


   .. py:attribute:: num_components


   .. py:attribute:: feature_size


   .. py:attribute:: param_indices
      :type:  Dict[str, Tuple[int, int]]


   .. py:attribute:: mixture_logits_indices


   .. py:attribute:: output_projection


   .. py:method:: forward(x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[str]]]

      x: [B, T, H] -> dict with canonical component names and flat params mapped to keys.



   .. py:method:: predict(params: Dict[str, torch.Tensor], method: str = 'mean') -> torch.Tensor

      Mixture mean (univariate) -> [B,T,1].



   .. py:method:: sample_quantiles(params: Dict[str, torch.Tensor], quantile_levels: List[float], *, num_mc: int = 256, temperature: float = 1.0, top_p: Optional[float] = None, min_std: float = 0.0001) -> torch.Tensor

      Monte Carlo mixture quantiles (univariate MDN).
      Returns: [B, T, 1, Q]



   .. py:method:: sample(params: Dict[str, Union[torch.Tensor, List[str]]], *, temperature: float = 1.0, top_p: Optional[float] = 0.9, min_std: float = 0.0001, eta_blend_mean: float = 0.3) -> torch.Tensor

      Sample a single step for AR feedback. Returns [B,1,1].



   .. py:method:: get_loss_fn() -> Optional[Callable]

      Returns the default loss function associated with this head.

      This method should be implemented by subclasses to provide a suitable
      loss function for the type of output they produce. For example, a
      point forecast head might return Mean Squared Error, while a
      probabilistic head might return Negative Log-Likelihood.

      :returns: A callable loss function, or None if the head
                does not have a default loss.
      :rtype: Optional[Callable]



.. py:class:: StudentTHead(hidden_size: int, output_size: int = 1, *, min_log_scale: float = -7.0, max_log_scale: float = 5.0, min_log_df: float = -2.0, max_log_df: float = 6.0, sigma_floor: float = 0.0001, df_floor: float = 1.001, init_log_scale: Optional[float] = None, init_log_df: Optional[float] = None, **kwargs)

   Bases: :py:obj:`temporal.modules.heads.base_output_head.BaseOutputHead`


   Student's t output head (no torch.special.betainc / StudentT.icdf dependency).

   forward(x): concat params [mu, log_scale, log_df] -> [B, T, 3F]
   predict("mean"|"median") -> [B, T, F]
   sample(...)              -> [B, T, F]
   sample_quantiles(qs)     -> [B, T, F, Q] (via vectorized bisection)


   .. py:attribute:: feature_size
      :value: 1



   .. py:attribute:: proj


   .. py:attribute:: min_log_scale


   .. py:attribute:: max_log_scale


   .. py:attribute:: min_log_df


   .. py:attribute:: max_log_df


   .. py:attribute:: sigma_floor


   .. py:attribute:: df_floor


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Processes the model's final hidden state to produce the output.

      This method must be implemented by all subclasses.

      :param hidden_state: The final hidden state from the model's
                           backbone, typically of shape `[batch_size, seq_len, d_model]`.
      :type hidden_state: torch.Tensor

      :returns:

                The model's final output, with its shape and meaning
                    determined by the specific head implementation.
      :rtype: torch.Tensor



   .. py:method:: predict(params: torch.Tensor, method: str = 'mean') -> torch.Tensor


   .. py:method:: sample(params: torch.Tensor, *, method: str = 'reparam', temperature: float = 1.0, eta_blend_mean: float = 0.0, generator: Optional[torch.Generator] = None) -> torch.Tensor


   .. py:method:: sample_quantiles(params: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor


   .. py:method:: get_loss_fn() -> Optional[Callable]

      Returns the default loss function associated with this head.

      This method should be implemented by subclasses to provide a suitable
      loss function for the type of output they produce. For example, a
      point forecast head might return Mean Squared Error, while a
      probabilistic head might return Negative Log-Likelihood.

      :returns: A callable loss function, or None if the head
                does not have a default loss.
      :rtype: Optional[Callable]



