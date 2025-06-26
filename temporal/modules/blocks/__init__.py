
# temporal/modules/blocks/__init__.py

# Import all block modules here so that the @register_block decorator is triggered
# and the block types are available in the central registry.

from . import block_hierarchical
from . import block_multiscale
from . import effitime_block
from . import block_adaptive # Add this line to register the new block
from . import block_s4 # Add this line to register the new block
