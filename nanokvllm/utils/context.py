from dataclasses import dataclass
import torch
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    
    compression_events: list | None = None   # newly add: record the meat data of seq to be compressed（only return by rank0）
    compress_need_mask: list | None = None   # newly add: python list or cpu BoolTensor
    compress_any: bool = False 

    is_compress_step: bool = False
    compress_selected_batch_indices: list | None = None
    compress_selected_seq_ids: list[int] | None = None
    compress_base_context_lens: torch.Tensor | None = None  
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, 
                       compression_events=[],    
                       compress_need_mask=None,
                       compress_any=False,
                        is_compress_step=False,
                        compress_selected_batch_indices=None,
                        compress_selected_seq_ids=None,
                        compress_base_context_lens=None)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()