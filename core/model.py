"""Backward-compat shim — all classes moved to submodules."""
import warnings
warnings.warn("core.model is deprecated, import from submodules directly", DeprecationWarning, stacklevel=2)

from .embedding import ZeckendorfEmbedding, PartitionedEmbedding, LmHead, PartitionedHead
from .bind import migrate_bind_state_dict, BottleneckBind
from .mirror import GroupedCognitiveMirror
from .mlp import GroupedMLP
from .block import WideBindBlock
from .stack import WideBindStack, AdaptiveController, MirrorLRScheduler
from .vsa_utils import dct_basis, zeckendorf_codes, fib_sigmoid_init, sparse_block_codes, vsa_prefix_scan
