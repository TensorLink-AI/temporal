# ---- Embedding modules --------------------------------------------
from .embedding                   import TimeSeriesValueEmbedding
# from .embedding                   import PositionalEmbedding # Commented out - was removed/commented in embedding.py
from .embedding                   import SinusoidalPositionalEmbedding
from .embedding                   import TimeSeriesPatchEmbedding
from .embedding                   import TimeSeriesGlobalEmbedding
from .embedding                   import RotaryPositionalEmbedding
from .embedding                   import LearnedAbsolutePositionalEmbedding
from .embedding                   import ShawRelativePositionalBias
from .embedding                   import FourierFeatureEmbedding
from .embedding                   import Time2VecEmbedding
from .embedding                   import ALiBiPositionalBias
from .embedding                   import BucketedRelativeBias
from .embedding                   import ConvolutionalPositionalEmbedding
from .embedding                   import TimeDeltaEmbedding
from .embedding                   import StackedPositionalEmbedding # Add the new stacked embedding
from .embedding                   import WaveletPositionalEmbedding # Add the new stacked embedding
from .embedding                   import S4PositionalEmbedding # Add the new stacked embedding

__all__ = [
    "TimeSeriesValueEmbedding",
    # "PositionalEmbedding", # Commented out
    "SinusoidalPositionalEmbedding",
    "TimeSeriesPatchEmbedding",
    "TimeSeriesGlobalEmbedding",
    "RotaryPositionalEmbedding",
    "LearnedAbsolutePositionalEmbedding",
    "ShawRelativePositionalBias",
    "FourierFeatureEmbedding",
    "Time2VecEmbedding",
    "ALiBiPositionalBias",
    "BucketedRelativeBias",
    "ConvolutionalPositionalEmbedding",
    "TimeDeltaEmbedding",
    "StackedPositionalEmbedding", # Add the new stacked embedding class name
    "S4PositionalEmbedding".
    "WaveletPositionalEmbedding"
]
