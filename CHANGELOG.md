# 📦 Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.0] - 2025-03-27
### Added
- Core encoder-decoder Transformer with AR generation
- Modular `HeadAggregator` with support for mean, gated, attention, fusion, stacked, weighted_mean
- KernelSynth-based time series data generation
- Hybrid loss support for stacking multiple loss types
- Adaptive context length (configurable)
- `from_pretrained_temporal()` model loader
- Initial CLI and config validation

### Fixed
- Improved forward compatibility for non-quantile losses
- Numerical stability in attention weights

### Notes
- This is the initial release of the `temporal` framework. 🚀
