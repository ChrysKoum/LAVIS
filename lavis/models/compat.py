"""
Compatibility helpers for modern transformers versions.
"""

from __future__ import annotations
from typing import Callable, Any
import logging


def resolve_apply_chunking_to_forward():
    """
    Returns a callable apply_chunking_to_forward compatible across transformers versions.

    Tries:
      - transformers.pytorch_utils.apply_chunking_to_forward  (newer)
      - transformers.modeling_utils.apply_chunking_to_forward (older)
    If missing, returns a local fallback implementation.
    """
    try:
        from transformers.pytorch_utils import apply_chunking_to_forward as fn
        return fn
    except Exception:
        pass

    try:
        from transformers.modeling_utils import apply_chunking_to_forward as fn
        return fn
    except Exception:
        pass

    logging.warning("transformers.apply_chunking_to_forward not found; using local fallback.")

    # Minimal fallback (works for typical HF usage pattern)
    def _fallback_apply_chunking_to_forward(
        forward_fn: Callable[..., Any],
        chunk_size: int,
        chunk_dim: int,
        *input_tensors: Any,
    ):
        import torch
        
        if chunk_size <= 0:
            return forward_fn(*input_tensors)

        tensor_shape = input_tensors[0].shape
        dim_size = tensor_shape[chunk_dim]
        if dim_size % chunk_size != 0:
            raise ValueError(f"chunk_size ({chunk_size}) must divide dimension ({dim_size})")

        num_chunks = dim_size // chunk_size
        outputs = []
        for i in range(num_chunks):
            sl = [slice(None)] * len(tensor_shape)
            sl[chunk_dim] = slice(i * chunk_size, (i + 1) * chunk_size)
            chunk_inputs = [t[tuple(sl)] for t in input_tensors]
            outputs.append(forward_fn(*chunk_inputs))

        out0 = outputs[0]
        if isinstance(out0, tuple):
            merged = []
            for k in range(len(out0)):
                merged.append(torch.cat([o[k] for o in outputs], dim=chunk_dim))
            return tuple(merged)

        return torch.cat(outputs, dim=chunk_dim)

    return _fallback_apply_chunking_to_forward


def resolve_find_pruneable_heads_and_indices():
    """
    Returns find_pruneable_heads_and_indices compatible across transformers versions.
    """
    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices as fn
        return fn
    except Exception:
        pass

    try:
        from transformers.modeling_utils import find_pruneable_heads_and_indices as fn
        return fn
    except Exception:
        pass

    logging.warning("transformers.find_pruneable_heads_and_indices not found; using local fallback.")

    def _fallback(heads, n_heads, head_size, already_pruned_heads):
        import torch
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    return _fallback


def resolve_prune_linear_layer():
    """
    Returns prune_linear_layer compatible across transformers versions.
    """
    try:
        from transformers.pytorch_utils import prune_linear_layer as fn
        return fn
    except Exception:
        pass

    try:
        from transformers.modeling_utils import prune_linear_layer as fn
        return fn
    except Exception:
        pass

    logging.warning("transformers.prune_linear_layer not found; using local fallback.")

    def _fallback(layer, index, dim=0):
        import torch
        import torch.nn as nn
        
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    return _fallback


# Resolve at import time (these calls don't import heavy modules)
apply_chunking_to_forward = resolve_apply_chunking_to_forward()
find_pruneable_heads_and_indices = resolve_find_pruneable_heads_and_indices()
prune_linear_layer = resolve_prune_linear_layer()
