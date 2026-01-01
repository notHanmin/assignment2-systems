from __future__ import annotations

from typing import Type

import torch
import math
import einops
import triton
import triton.language as tl

def FlashAttentionForward(Q, K, V, Br, Bc):
    batch_size = Q.shape[0]
    N = Q.shape[1]
    d = Q.shape[2]
    Tr = math.ceil(N/Br)    
    Tc = math.ceil(N/Bc)

    O = torch.zeros(batch_size, Tr, Br, d, device = Q.device, dtype = Q.dtype)
    L = torch.zeros(batch_size, Tr, Br, device = Q.device, dtype = Q.dtype)
    l = torch.zeros(batch_size, Br, device = Q.device, dtype = Q.dtype)
    m = torch.full((batch_size, Br), float('-inf'), device = Q.device, dtype = Q.dtype)

    for i in range(Tr):
        Qi = Q[:, i*Br:(i+1)*Br, :]
        Oi = torch.zeros(batch_size, Br, d, device = Q.device, dtype = Q.dtype)
        li = torch.zeros(batch_size, Br, device = Q.device, dtype = Q.dtype)
        mi = torch.full((batch_size, Br), float('-inf'), device = Q.device, dtype = Q.dtype)

        for j in range(Tc):
            Kj = K[:, j*Bc:(j+1)*Bc, :]
            Vj = V[:, j*Bc:(j+1)*Bc, :]
            Sij = einops.einsum(Qi, Kj, 'batch_size Br d, batch_size Bc d -> batch_size Br Bc') / math.sqrt(d)
            mij = torch.maximum(mi, torch.max(Sij, 2)[0])
            mij_aligned = einops.rearrange(mij, 'batch_size Br -> batch_size Br 1')
            Pij = torch.exp(Sij - mij_aligned)
            lij = torch.exp(mi - mij) * li + torch.sum(Pij, dim=2)
            Oij = torch.exp(einops.rearrange(mi - mij, 'batch_size Br -> batch_size Br 1')) * Oi + einops.einsum(Pij, Vj, 'batch_size Br Bc, batch_size Bc d -> batch_size Br d')
            Oi = Oij
            li = lij
            mi = mij
        
        Oi = 1 / einops.rearrange(li, 'batch_size Br -> batch_size Br 1') * Oi
        li = mi + torch.log(li)

        O[:, i, :, :] = Oi
        L[:, i, :] = li

    O = einops.rearrange(O, 'batch_size Tr Br d -> batch_size (Tr Br) d')
    L = einops.rearrange(L, 'batch_size Tr Br -> batch_size (Tr Br)')
    O = O[:, :N, :]
    L = L[:, :N]
    return O, L

def FlashAttentionBackward(Q, K, V, O, L, dO, Br, Bc):
    return dQ, dK, dV

class FlashAttention2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal):
            Br, Bc = 32, 32
            O, L = FlashAttentionForward(Q, K, V, Br, Bc)
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.Br, ctx.Bc, ctx.is_causal = Br, Bc, is_causal
        
            return O

        @staticmethod
        def backward(ctx, dO):
            Q, K, V, O, L = ctx.saved_tensors
            Bc = ctx.Bc
            Br = ctx.Br
            dQ, dK, dV = FlashAttentionBackward(Q, K, V, O, L, dO, Bc, Br)
            
            return dQ, dK, dV, None

@triton.jit
def FlashAttentionForwardTriton(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        col_indices = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
        row_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE

        mask = col_indices < N_KEYS
        mask_causal = row_indices[:, None] >= col_indices[None, :]
        mask_combined = mask[None, :] & mask_causal
        #Sij = tl.where(mask[None, :], Sij, float("-inf"))
        Sij = tl.where(mask_combined if is_causal else mask, Sij, float("-inf"))
        
        mij = tl.maximum(mi, tl.max(Sij, 1))
        Pij = tl.exp(Sij - mij[:, None])
        lij = tl.exp(mi - mij) * li + tl.sum(Pij, 1)
        Oij = tl.exp(mi-mij)[:, None] * oi + tl.dot(Pij.to(Vj.type.element_ty), Vj)

        oi=Oij
        li = lij
        mi = mij
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    oi = 1 / li[:, None] * oi
    li = mi + tl.log(li)

    tl.store(O_block_ptr, oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, li, boundary_check=(0,))        

class FlashAttention2Triton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal):
            batch_size, N_QUERIES, D = Q.shape
            N_KEYS = K.shape[1]
            O = torch.zeros((batch_size, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
            L = torch.zeros((batch_size, N_QUERIES), device=Q.device, dtype=torch.float32)
            Br, Bc = 32, 32

            ctx.save_for_backward(Q, K, V, O, L)
            ctx.Br, ctx.Bc, ctx.is_causal = Br, Bc, is_causal
        
            FlashAttentionForwardTriton[(N_QUERIES + Br - 1) // Br, batch_size](
                Q, K, V, O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                N_QUERIES, N_KEYS,
                1/math.sqrt(D),
                D,
                Br,
                Bc,
                is_causal
                )
            return O

        @staticmethod
        def backward(ctx, dO):
            Q, K, V, O, L = ctx.saved_tensors
            Bc = ctx.Bc
            Br = ctx.Br
            dQ, dK, dV = FlashAttentionBackward(Q, K, V, O, L, dO, Bc, Br)
            
            return dQ, dK, dV, None

def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttention2
    #raise NotImplementedError


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttention2Triton
    #raise NotImplementedError


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
