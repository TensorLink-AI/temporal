"""
Auto-import sub-packages so that *all* register() calls run once
when `import temporal.modules` is executed.
"""

from importlib import import_module

_subpkgs = [
    "attentions",
    "blocks",
    "decoders",
    "embedders",
    "encoders",
    "feedforward",
    "heads",
    "losses",
    "norm",
    "quantizers"
]

for _name in _subpkgs:
    import_module(f"{__name__}.{_name}")

__all__ = _subpkgs  # so `from temporal.modules import *` pulls them in
