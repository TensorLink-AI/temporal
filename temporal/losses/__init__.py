# This __init__.py defines the temporal.losses package.
# It does not explicitly import modules to avoid circular dependencies
# and allows other modules to import specific loss functions directly from their source.

# This file might be empty or contain package-level metadata/configurations.
# For now, it will be kept minimal.

# Example of what it might contain if it were to expose specific items from submodules:
# from .crps_loss_ensemble import crps_ensemble
# from .modules.losses.losses import MyLossClass

# The current approach is to allow direct imports from submodules or
# rely on the ModuleBuilder for dynamic instantiation.
