"""
Compatibility shims for transformers API changes.

This module provides fallback implementations for functions that have moved
between transformers versions, allowing LAVIS to work with both old and new
versions of transformers.
"""

# apply_chunking_to_forward was moved from modeling_utils to pytorch_utils in newer transformers
try:
    from transformers.pytorch_utils import apply_chunking_to_forward
except ImportError:
    try:
        from transformers.modeling_utils import apply_chunking_to_forward
    except ImportError:
        # Fallback implementation based on transformers source
        # https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py
        import torch
        
        def apply_chunking_to_forward(
            forward_fn,
            chunk_size: int,
            chunk_dim: int,
            *input_tensors,
        ):
            """
            Applies chunking to a forward function to reduce memory usage.
            
            This function divides the input tensors into chunks along the specified
            dimension, applies the forward function to each chunk, and concatenates
            the results.
            
            Args:
                forward_fn: The forward function to apply
                chunk_size: Size of each chunk (0 means no chunking)
                chunk_dim: The dimension along which to chunk
                *input_tensors: Input tensors to chunk
                
            Returns:
                Concatenated output from applying forward_fn to each chunk
            """
            assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

            # inspect.signature is not available on torch.jit.script compiled functions
            if chunk_size <= 0:
                return forward_fn(*input_tensors)

            tensor_shape = input_tensors[0].shape[chunk_dim]
            for input_tensor in input_tensors:
                if input_tensor.shape[chunk_dim] != tensor_shape:
                    raise ValueError(
                        f"All input tensors must have the same shape along chunk_dim {chunk_dim}, "
                        f"but got {input_tensor.shape[chunk_dim]} and {tensor_shape}"
                    )

            if tensor_shape % chunk_size != 0:
                raise ValueError(
                    f"The dimension to be chunked ({tensor_shape}) is not a multiple "
                    f"of the chunk size ({chunk_size})"
                )

            num_chunks = tensor_shape // chunk_size

            # chunk input tensors
            input_tensors_chunks = tuple(
                input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors
            )
            # apply forward fn to every chunk
            output_chunks = tuple(
                forward_fn(*input_tensors_chunk)
                for input_tensors_chunk in zip(*input_tensors_chunks)
            )
            # concatenate output chunks
            return torch.cat(output_chunks, dim=chunk_dim)


# find_pruneable_heads_and_indices - also used in med.py
try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    try:
        from transformers.modeling_utils import find_pruneable_heads_and_indices
    except ImportError:
        import torch
        
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            """
            Finds the heads and their indices taking `already_pruned_heads` into account.
            """
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index


# prune_linear_layer - also used in med.py
try:
    from transformers.pytorch_utils import prune_linear_layer
except ImportError:
    try:
        from transformers.modeling_utils import prune_linear_layer
    except ImportError:
        import torch
        import torch.nn as nn
        
        def prune_linear_layer(layer, index, dim=0):
            """
            Prune a linear layer to keep only entries in index.
            """
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
