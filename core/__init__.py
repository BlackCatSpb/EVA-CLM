from .config import WideBindConfig, EVAConfig
from .vsa_utils import dct_basis, zeckendorf_codes, sparse_block_codes, vsa_prefix_scan
from .embedding import ZeckendorfEmbedding, PartitionedEmbedding, LmHead, PartitionedHead, SigmoidCodedHead, CognitiveCodedHead
from .bind import BottleneckBind, SpiralBind, TrajectorySpiralBind
from .mirror import GroupedCognitiveMirror
from .mlp import GroupedMLP
from .block import EVABlock
from .stack import EVAStack, AdaptiveController, MirrorLRScheduler
from .live_inference import LiveInference, MirrorMonitor
from .logit_cache import LogitCache, LogitAttention, LogitCacheAttention
from .logit_cache_v2 import PerScaleCacheAttention

# Backward compat
CognitiveMirror = GroupedCognitiveMirror
