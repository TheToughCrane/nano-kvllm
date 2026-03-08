import torch
from torch import nn
import triton
import triton.language as tl
import os
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.layers.compress_utils import MyCompressCompact,store_qkvcache

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        vllm_config
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

        self.kv_compress_enabled = vllm_config.kv_compress_enabled
        self.kv_compress_N = vllm_config.kv_compress_N
        self.kv_compress_S = vllm_config.kv_compress_S
        self.kv_compress_R = vllm_config.kv_compress_R
        if self.kv_compress_enabled:
            self.q_cache = torch.tensor([])


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, Layer):


        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            """if not compress kv cache, we do not need to store query cache"""
            if not self.kv_compress_enabled:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                """most kv cache compression methods need to store a part of query cache,i.e.,lastest 10 queries"""
                """for simplicity, i stored all tokens's queries, i will optimize this when available"""
                store_qkvcache(k, v, k_cache, v_cache, context.slot_mapping,q,self.q_cache)
 

        if not context.is_prefill and self.kv_compress_enabled:
            """we only compress kv cache in decode phase"""
            MyCompressCompact(self.q_cache,k_cache,v_cache,Layer, k_cache.size(1), self.kv_compress_S, self.kv_compress_R)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            
        else:    
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
