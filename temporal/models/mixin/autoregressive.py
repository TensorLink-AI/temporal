import torch
from typing import Any, Dict, List, Optional, Union

# assume these live somewhere in your codebase
from temporal.models.mixin.autoregressive_patch import AutoregressivePatchMixin
from temporal.models.mixin.autoregressive_stepwise import AutoregressiveStepwiseMixin


import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union

class AutoregressiveDispatchMixin:
    """
    Dispatch between patch‐based and stepwise generation based on config.
    """

    def generate(
        self,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        If the model's value_embedding_config.type == "patch", delegate to
        AutoregressivePatchMixin.generate; otherwise to AutoregressiveStepwiseMixin.generate.
        """
        # assume `self.config.value_embedding_config.type` exists
        is_patch = getattr(self.config.value_embedding_config, "type", None) == "patch"

        if is_patch:
            # call the patch‐based generator
            return super(AutoregressiveDispatchMixin, self).generate(*args, **kwargs)
        else:
            # call the stepwise generator
            return super(AutoregressiveStepwiseMixin, self).generate(*args, **kwargs)
