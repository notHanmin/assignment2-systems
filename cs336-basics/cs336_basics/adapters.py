from __future__ import annotations

import os
from typing import IO, Any, BinaryIO, Optional
from collections.abc import Iterable, Iterator, Callable
from jaxtyping import Float, Int
from collections import Counter
import regex as re
from multiprocessing import Pool
import numpy.typing as npt
import numpy as np
import torch
import json
from torch import Tensor, nn
from .common import gpt2_bytes_to_unicode
import math
import einops

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        variance = 2 / (out_features + in_features)
        std = math.sqrt(variance)
        nn.init.trunc_normal_(self.W, 0.0, std, -3*std, 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oi, ... i -> ... o", self.W, x)

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    return torch.einsum("oi, ... i -> ... o", weights, in_features)
    #raise NotImplementedError


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embeddings_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embeddings_dim, device=device, dtype=dtype))

        variance = 2 / (num_embeddings + embeddings_dim)
        std = math.sqrt(variance)
        nn.init.trunc_normal_(self.weight, 0.0, std, -3*std, 3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    return weights[token_ids]
    #raise NotImplementedError

class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        dff_approx = 8/3 * d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = torch.einsum("oi, ... i -> ... o", self.w1.W, x)
        w3x = torch.einsum("oi, ... i -> ... o", self.w3.W, x)
        silu = w1x * torch.sigmoid(w1x)
        product = silu * w3x
        return torch.einsum("io, ... o -> ...i", self.w2.W, product)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    w1x = torch.einsum("oi, ... i -> ... o", w1_weight, in_features)
    w3x = torch.einsum("oi, ... i -> ... o", w3_weight, in_features)
    silu = w1x * torch.sigmoid(w1x)
    product = silu * w3x
    return torch.einsum("io, ... o -> ...i", w2_weight, product)
    #raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    qk = torch.einsum("... qd, ... kd -> ... qk", Q, K)
    masked_qk = torch.where(mask, qk, -torch.inf)
    sm = run_softmax(masked_qk/math.sqrt(Q.shape[-1]), -1)
    return torch.einsum("... qk, ... kd -> ... qd", sm, V)
    
    #raise NotImplementedError

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, theta: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        
        self.W_q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        var = 1/d_model
        std = math.sqrt(var)

        nn.init.trunc_normal_(self.W_q, 0.0, std, -3*std, 3*std)
        nn.init.trunc_normal_(self.W_k, 0.0, std, -3*std, 3*std)
        nn.init.trunc_normal_(self.W_v, 0.0, std, -3*std, 3*std)
        nn.init.trunc_normal_(self.W_o, 0.0, std, -3*std, 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Q = torch.einsum("ki, ...si -> ... sk", self.W_q, x)
        K = torch.einsum("ki, ...si -> ... sk", self.W_k, x)
        V = torch.einsum("ki, ...si -> ... sk", self.W_v, x)

        multiQ = einops.rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        multiK = einops.rearrange(K, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        multiV = einops.rearrange(V, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        
        seq_len = x.shape[-2]
        d_k = self.d_model // self.num_heads
        device = x.device
        max_seq_len = multiQ.shape[-2]
        batch_size = multiQ.shape[0]
        positions = torch.arange(seq_len, device=device)
        token_positions = einops.repeat(positions, 'seq_len ->batch seq_len', batch = batch_size)

        rope_module = RoPE(self.theta, d_k, max_seq_len, device)
        multiQ_rope = rope_module.forward(multiQ, token_positions)
        multiK_rope = rope_module.forward(multiK, token_positions)
        
        
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        multihead_attention = run_scaled_dot_product_attention(multiQ_rope, multiK_rope, multiV, causal_mask)
        combined_multihead_attention = einops.rearrange(multihead_attention, "... h seq_len d_v -> ... seq_len (h d_v)")

        WoMultihead = torch.einsum("mv, ... sv -> ... sm", self.W_o, combined_multihead_attention)
        return WoMultihead

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    seq_len = in_features.shape[-2]

    Q = torch.einsum("ki, ...si -> ... sk", q_proj_weight, in_features)
    K = torch.einsum("ki, ...si -> ... sk", k_proj_weight, in_features)
    V = torch.einsum("ki, ...si -> ... sk", v_proj_weight, in_features)

    multiQ = einops.rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h=num_heads)
    multiK = einops.rearrange(K, "... seq_len (h d) -> ... h seq_len d", h=num_heads)
    multiV = einops.rearrange(V, "... seq_len (h d) -> ... h seq_len d", h=num_heads)
    
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=in_features.device, dtype=torch.bool))
    multihead_attention = run_scaled_dot_product_attention(multiQ, multiK, multiV, causal_mask)
    combined_multihead_attention = einops.rearrange(multihead_attention, "... h seq_len d_v -> ... seq_len (h d_v)")

    WoMultihead = torch.einsum("mv, ... sv -> ... sm", o_proj_weight, combined_multihead_attention)
    return WoMultihead
    #raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    Q = torch.einsum("oi, ...si -> ...so", q_proj_weight, in_features)
    K = torch.einsum("oi, ...si -> ...so", k_proj_weight, in_features)
    V = torch.einsum("oi, ...si -> ...so", v_proj_weight, in_features)

    multiQ = einops.rearrange(Q, "... s (h d) -> ... h s d", h=num_heads)
    multiK = einops.rearrange(K, "... s (h d) -> ... h s d", h=num_heads)
    multiV = einops.rearrange(V, "... s (h d) -> ... h s d", h=num_heads)

    seq_len = in_features.shape[-2]
    d_k = d_model // num_heads
    device = in_features.device

    rope_module = RoPE(theta, d_k, max_seq_len, device)
    multiQ_rope = rope_module.forward(multiQ, token_positions)
    multiK_rope = rope_module.forward(multiK, token_positions)

    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    multihead_attention = run_scaled_dot_product_attention(multiQ_rope, multiK_rope, multiV, causal_mask)

    attention_combined = einops.rearrange(multihead_attention, "... h s d -> ... s (h d)")

    final_output = torch.einsum("oi, ...si -> ...so", o_proj_weight, attention_combined)
    
    return final_output


    #raise NotImplementedError

class RoPE(nn.Module):
    """
    The RoPE module, responsible for creating, caching, and applying
    rotary positional embeddings.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initializes the RoPE module.

        Args:
            theta (float): The base for the frequency calculation.
            d_k (int): The dimension of the query/key vectors.
            max_seq_len (int): The maximum sequence length to pre-compute for.
            device: The device to store the cache on.
        """
        super().__init__()
        
        multiples = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        exponents = multiples / d_k
        freqs = 1.0 / (theta ** exponents)

        # Create position indices [0, 1, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # Calculate the angles (m * theta_k) for all position/frequency pairs
        # Shape: (max_seq_len, d_k / 2)
        angles = torch.einsum("p, f -> p f", positions, freqs)

        # Duplicate each value to align with the (d_k) feature dimension.
        # Shape becomes (max_seq_len, d_k)
        cos_vals, sin_vals = torch.cos(angles), torch.sin(angles)
        #cos_cache = einops.repeat(cos_vals, 'len d_half -> len (d_half r)', r=2)
        #sin_cache = einops.repeat(sin_vals, 'len d_half -> len (d_half r)', r=2)
        # Register the caches as non-persistent buffers. They are part of the
        # module's state but are not parameters and won't be saved in the state_dict.
        self.register_buffer('cos_cache', cos_vals, persistent=False)
        self.register_buffer('sin_cache', sin_vals, persistent=False)
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:
        """
        Applies RoPE to a given input tensor using the pre-computed cache.

        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor.
            token_positions (Int[Tensor, "... sequence_length"]): Token positions for the input.

        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPE applied.
        """
        # Use the token_positions to look up the correct sin/cos values from the cache.
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        
        # Split the input into two halves for the pairwise rotation.
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Since the cos/sin caches are interleaved (e.g., [c0, c0, c1, c1, ...]),
        # we only need the values for each pair. Slicing gives us the correct shape.
        #cos_pairs = cos[..., 0::2]
        #sin_pairs = sin[..., 0::2]

        # Apply the rotation
        rearranged_cos = einops.rearrange(cos, "batch seq_len d_model -> batch 1 seq_len d_model")
        rearranged_sin = einops.rearrange(sin, "batch seq_len d_model -> batch 1 seq_len d_model")
        rotated_x1 = x1 * rearranged_cos - x2 * rearranged_sin
        rotated_x2 = x1 * rearranged_sin + x2 * rearranged_cos
        
        # Combine the rotated halves back into a single tensor.
        rotated_x = torch.empty_like(x)
        rotated_x[..., 0::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2
        
        return rotated_x

# The adapter function that the test suite will call.
# This function should be in the file that your test adapter imports from.
def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor):
    """
    This function is the adapter for the test suite.
    It instantiates the RoPE module and runs the forward pass.
    """
    # The test might run on CPU, so we get the device from the input tensor
    device = in_query_or_key.device
    
    # Instantiate your module with the parameters from the test
    rope_module = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
    
    # Run the forward pass and return the result
    return rope_module.forward(in_query_or_key, token_positions)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    eps = 1e-5
    norm1 = run_rmsnorm(d_model, eps, weights['ln1.weight'], in_features)
    seq_len = in_features.shape[-2]
    device = in_features.device
    batch_size = in_features.shape[0]
    positions = torch.arange(seq_len, device=device)
    token_positions = einops.repeat(positions, 'seq_len ->batch seq_len', batch = batch_size)

    attention = run_multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights['attn.output_proj.weight'],
        norm1,
        token_positions
    )
    res = in_features + attention

    norm2 = run_rmsnorm(d_model, eps, weights['ln2.weight'], res)
    ff = run_swiglu(
        d_model, d_ff,
        weights['ffn.w1.weight'],
        weights['ffn.w2.weight'],
        weights['ffn.w3.weight'],
        norm2
    )

    return res + ff

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: int,
                 device=None,
                 dtype=None
                 ):
        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size, d_model, device, dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model, 
                    num_heads=num_heads, 
                    d_ff=d_ff, 
                    theta=rope_theta, # Make sure to pass all necessary args
                    device=device, 
                    dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)

        for block in self.layers:
            x = block(x)

        x = self.final(x)
        logits = self.lm_head(x)

        return logits

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    device = in_indices.device

    eps = 1e-5
    embedding = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices)

    for i in range(num_layers):
        layer_weights = {
            k.replace(f"layers.{i}.", ""): v 
            for k, v in weights.items() 
            if k.startswith(f"layers.{i}.")
        }

        embedding = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=embedding,
        )
        
    norm = run_rmsnorm(d_model, eps, weights['ln_final.weight'], embedding)
    linear = run_linear(d_model, vocab_size, weights['lm_head.weight'], norm)
    return linear
    #raise NotImplementedError

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = einops.reduce(x ** 2, '... d_model -> ... 1', 'mean')
        rms_a = torch.sqrt(mean_square + self.eps)
        return (x / rms_a) * self.g


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    mean_square = einops.reduce(in_features ** 2, '... d_model -> ... 1', 'mean')
    rms_a = torch.sqrt(mean_square + eps)
    return (in_features / rms_a) * weights

    #raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset)
    end = n - context_length - 1
    start_indices = np.random.randint(0, end + 1, size = batch_size)

    input_seq = [dataset[i : i+context_length] for i in start_indices]
    target_seq = [dataset[i+1 : i+1+context_length] for i in start_indices]

    x = np.stack(input_seq)
    y = np.stack(target_seq)

    x_tensor = torch.from_numpy(x).to(torch.long).to(device)
    y_tensor = torch.from_numpy(y).to(torch.long).to(device)
    return x_tensor, y_tensor
    #raise NotImplementedError



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_values = torch.max(in_features, dim=dim, keepdim=True).values
    
    # Subtract max for numerical stability
    shifted = in_features - max_values
    
    # Apply exponential
    exp_values = torch.exp(shifted)
    
    # Sum along the specified dimension, keeping dimensions for broadcasting
    sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)
    
    # Normalize to get softmax
    return exp_values / sum_exp

    #raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_values = torch.max(inputs, dim=1, keepdim=True).values
    stable_values = inputs - max_values
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_values), dim=-1))
    targets_extra = einops.rearrange(targets, "batch_size -> batch_size 1")
    target_logits = torch.gather(stable_values, -1, targets_extra)
    return (log_sum_exp - target_logits).mean()
    #raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(torch.sum(grad.pow(2)) for grad in grads))
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(clip_coef)
    return parameters
    #raise NotImplementedError

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas = betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["betas"][0]
            b2 = group["betas"][1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 1
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                step = state["step"]
                state["step"] += 1
                
                grad = p.grad.data
                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                bias_correction1 = 1 - b1 ** step
                bias_correction2 = 1 - b2 ** step

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = (v.sqrt()).add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)
                p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss

def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW
    #raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate+(1+math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))/2 * (max_learning_rate - min_learning_rate)
    #raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)
    #raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
    #raise NotImplementedError

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_tokens_pattern = "(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")"
        self.special_tokens_regex = re.compile(special_tokens_pattern)
        self.pre_tokenize_regex = re.compile(PAT)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding = 'utf-8') as f:
            gpt2_vocab = json.load(f)
        with open(merges_filepath, 'r', encoding = 'utf-8') as f:
            bpe_merges_from_file = f.read().split('\n')[1:-1]

        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        vocab = {
            index: bytes([gpt2_byte_decoder[char] for char in token])
            for token, index in gpt2_vocab.items()
        }

        merges = []
        for merge_rule in bpe_merges_from_file:
            p1, p2 = merge_rule.split()
            b1 = bytes([gpt2_byte_decoder[char] for char in p1])
            b2 = bytes([gpt2_byte_decoder[char] for char in p2])
            merges.append((b1, b2))

        if special_tokens:
            for token in special_tokens:
                byte_encoded = token.encode("utf-8")
                if byte_encoded not in vocab.values():
                    vocab[len(vocab)] = byte_encoded

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def apply_bpe(self, part_bytes: bytes) -> list[int]:
        if not part_bytes:
            return []

        parts = [bytes([b]) for b in part_bytes]

        while True:
            best_pair_to_merge = None
            min_rank = float('inf')

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                if pair in self.merge_ranks and self.merge_ranks[pair] < min_rank:
                    min_rank = self.merge_ranks[pair]
                    best_pair_to_merge = pair
            
            if best_pair_to_merge is None:
                break

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair_to_merge:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        
        # Convert the final byte parts to token IDs
        return [self.byte_to_id[p] for p in parts]

    def encode(self, text: str) -> list[int]:

        if not self.special_tokens:
            tokenized_ids = []
            pre_tokenized_parts = self.pre_tokenize_regex.findall(text)
            for part in pre_tokenized_parts:
                part_bytes = part.encode("utf-8")
                ids = self.apply_bpe(part_bytes)
                tokenized_ids.extend(ids)
            return tokenized_ids
    
        tokenized_ids = []
        chunks = self.special_tokens_regex.split(text)

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                tokenized_ids.append(self.byte_to_id[chunk.encode("utf-8")])
            else:
                pre_tokenized_parts = self.pre_tokenize_regex.findall(chunk)

                for part in pre_tokenized_parts:
                    part_bytes = part.encode("utf-8")
                    ids = self.apply_bpe(part_bytes)
                    tokenized_ids.extend(ids)

        return tokenized_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            chunks = self.special_tokens_regex.split(text_chunk)
            for chunk in chunks:
                if not chunk:
                    continue
                if chunk in self.special_tokens:

                    yield self.byte_to_id[chunk.encode("utf-8")]
                else:
                    pre_tokenized_parts = self.pre_tokenize_regex.findall(chunk)
                    for part in pre_tokenized_parts:
                        part_bytes = part.encode("utf-8")
                        ids = self.apply_bpe(part_bytes)

                        for single_id in ids:
                            yield single_id

    def decode(self, ids: list[int]) -> str:
        output = b"".join(self.vocab[id] for id in ids)
        return output.decode("utf-8", errors="replace")

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(args):
    file_path, start, end, special_tokens = args

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

    delimiter = "|".join(re.escape(token) for token in special_tokens)
    subchunks_str = re.split(delimiter, chunk_str)

    chunk_counter = Counter()
    regex_compiled = re.compile(PAT)
    for subchunk_str in subchunks_str:
        matches = regex_compiled.finditer(subchunk_str)
        chunk_counter.update(match.group(0) for match in matches)

    return chunk_counter

def parallel_pretokenize(
        file_path: str,
        num_processes: int,
        special_tokens: list[str]
) -> Counter:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

    chunk_args = [
        (file_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=num_processes) as pool:
        chunk_counters = pool.map(pretokenize, chunk_args)

    total_counter = Counter()
    for counter in chunk_counters:
        total_counter.update(counter)

    return total_counter

'''
## Usage
data_path = "data/TinyStoriesV2-GPT4-valid.txt"
num_processes = os.cpu_count()
if num_processes is None:
    num_processes = 1

special_tokens_list = ["<|endoftext|>"]
pretoken_counts = parallel_pretokenize(data_path, num_processes, special_tokens_list)

count = pretoken_counts.get("<|endoftext|>", 0)
print(f"Occurrences of ' and': {count}")

top_10_occurrences = pretoken_counts.most_common(10)

print("--- Top 10 Occurrences ---")
for item, count in top_10_occurrences:
    item_encoded = item.encode("utf-8")
    print(f"{item_encoded}: {count}")
'''

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = os.cpu_count()
    if num_processes is None:
        num_processes = 1
    
    pretoken_counts = parallel_pretokenize(str(input_path), num_processes,  special_tokens)

    vocab = {i : bytes([i]) for i in range(256)}
    next_id = 256
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1
    
    pretoken_counts_bytes = {tuple(key.encode("utf-8")): value for key, value in pretoken_counts.items()}
    
    # Merge
    merges = []
    merge_steps = vocab_size - 256 - len(special_tokens)
    for i in range(merge_steps):
        # Build counter for every bytes pair within each pretoken
        pair_count = Counter()
        for pretoken_ids, count in pretoken_counts_bytes.items():
            for i in range(len(pretoken_ids) - 1):
                pair = (pretoken_ids[i], pretoken_ids[i + 1])
                pair_count[pair] += count

        # No more pairs to merge
        if not pair_count:
            break
        # Find pair to merge
        best_pair = max(pair_count, key=lambda k: (pair_count[k], vocab[k[0]], vocab[k[1]]))

        # Update vocabulary
        token1, token2 = vocab[best_pair[0]], vocab[best_pair[1]]
        merges.append((token1, token2))
        vocab[next_id] = token1 + token2       
        next_id += 1

        new_pretoken_counts_bytes = {}
        for pretoken_ids, count in pretoken_counts_bytes.items():
            new_ids= []
            index = 0
            while index < len(pretoken_ids):
                if index < len(pretoken_ids) - 1 and (pretoken_ids[index], pretoken_ids[index + 1]) == best_pair:
                    new_ids.append(next_id - 1)
                    index += 2
                else:
                    new_ids.append(pretoken_ids[index])
                    index += 1
            new_pretoken_counts_bytes[tuple(new_ids)] = count
        pretoken_counts_bytes = new_pretoken_counts_bytes

    return vocab, merges
    #raise NotImplementedError
