import importlib.metadata as _metadata

__version__ = _metadata.version("cs336_basics")

# expose submodules
from . import (
    data_loader,
    adamW,
    learning_rate_schedule,
    cross_entropy_loss,
    gradient_clip,
    softmax,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    SwiGLU,
    mha,
    scaled_dot_product_attention,
    Tokenizer,
    Transformer,
    TransformerLM,
    training_loop,
)

__all__ = [
    "__version__",
    "data_loader",
    "adamW",
    "learning_rate_schedule",
    "cross_entropy_loss",
    "gradient_clip",
    "softmax",
    "Embedding",
    "Linear",
    "RMSNorm",
    "RoPE",
    "SwiGLU",
    "mha",
    "scaled_dot_product_attention",
    "Tokenizer",
    "Transformer",
    "TransformerLM",
    "training_loop",
]
