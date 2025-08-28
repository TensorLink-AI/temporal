temporal.utils.hf_accessors
===========================

.. py:module:: temporal.utils.hf_accessors


Functions
---------

.. autoapisummary::

   temporal.utils.hf_accessors.save_hf
   temporal.utils.hf_accessors.load_hf


Module Contents
---------------

.. py:function:: save_hf(model: torch.nn.Module, config, save_directory: str, safe: bool = False, repo_id: Optional[str] = None, commit_message: Optional[str] = 'Save model using custom save_hf', private: bool = False, token: Optional[str] = None, push_to_hub: bool = False)

   Saves a model and its configuration to a directory, with optional Hub upload.

   This function saves a model's state dictionary and its configuration file
   in a format that is compatible with the Hugging Face ecosystem. It can
   also create a new repository on the Hub and upload the saved files.

   :param model: The PyTorch model to save.
   :type model: torch.nn.Module
   :param config: The configuration object for the model. It should have a
                  `to_dict()` method.
   :param save_directory: The local directory where the model and config
                          will be saved.
   :type save_directory: str
   :param safe: If True, saves the model weights using `safetensors`.
                Otherwise, uses `torch.save` (pickle format). Defaults to False.
   :type safe: bool
   :param repo_id: The ID of the repository on the Hugging Face
                   Hub (e.g., 'username/repo_name'). Required if `push_to_hub` is True.
   :type repo_id: Optional[str]
   :param commit_message: The commit message for the upload.
   :type commit_message: Optional[str]
   :param private: Whether to create the repository as private if it
                   doesn't exist. This is ignored if the repo already exists.
   :type private: bool
   :param token: Your Hugging Face API token. If None, it uses
                 the token from the environment or login cache.
   :type token: Optional[str]
   :param push_to_hub: If True, the function will push the `save_directory`
                       to the specified `repo_id` on the Hub.
   :type push_to_hub: bool

   :raises ImportError: If `safetensors` or `huggingface_hub` is required but not installed.
   :raises ValueError: If `push_to_hub` is True but `repo_id` is not provided.


.. py:function:: load_hf(model_name_or_path: str, model_cls: Type[torch.nn.Module], config_cls, safe: bool = False, map_location='cpu', cache_dir: Optional[str] = None, force_download: bool = False, token: Optional[str] = None, **model_kwargs)

   Loads a model and configuration from a local path or the Hugging Face Hub.

   This function can load models saved with `save_hf`. It automatically handles
   downloading files from the Hub if the `model_name_or_path` is a repository ID.

   :param model_name_or_path: The path to a local directory or a repository
                              ID on the Hugging Face Hub.
   :type model_name_or_path: str
   :param model_cls: The class of the model to instantiate.
                     It is expected to have an `__init__(self, config, **kwargs)` signature.
   :type model_cls: Type[torch.nn.Module]
   :param config_cls: The configuration class for the model. It should have a
                      `from_pretrained` or `from_dict` method.
   :param safe: If True, prioritizes loading `model.safetensors`.
   :type safe: bool
   :param map_location: The device to load the model weights onto (e.g., 'cpu', 'cuda:0').
   :type map_location: str
   :param cache_dir: The directory for caching downloaded Hub files.
   :type cache_dir: Optional[str]
   :param force_download: If True, forces a re-download from the Hub.
   :type force_download: bool
   :param token: Your Hugging Face API token for private repos.
   :type token: Optional[str]
   :param \*\*model_kwargs: Additional keyword arguments to pass to the model's constructor.

   :returns: An instance of `model_cls` with the loaded weights.
   :rtype: torch.nn.Module


