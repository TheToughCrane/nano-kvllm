import torch
from torch import nn
import triton
import triton.language as tl
from KvChat.utils.context import get_context
from KvChat.layers.CompressMethod import SnapKV, StreamingLLM, StrideKV



def MyCompressCompact(query_window_manager, k_cache: torch.Tensor, v_cache: torch.Tensor,
                      layer_id: int, block_size: int, S: int, R: int,query_window_size:int,
                      num_layers:int,context = None):
    
    """
    KV cache compression for decode phase (Query Window 2.0 version).

    Inputs:
      - query_window_manager: stores recent W queries per sequence, allocated on demand
      - k_cache: [num_blocks_k, block_size, num_kv_heads, head_dim]
      - v_cache: [num_blocks_k, block_size, num_kv_heads, head_dim]
      - layer_id: current layer id
      - block_size: KV block size
      - S: compression trigger threshold
      - R: number of retained tokens after compression

    Behavior:
      - only compress in decode
      - query input is taken from query_window_manager rather than paged q_cache
      - KV compact remains in-place
      - append compression_events for rank0 metadata update
    """
    """need_mask is a Python list[bool] with length == batch_size.
    True means the corresponding sequence in this decode batch needs compression.
    """
    need_mask = context.compress_need_mask
    if not any(need_mask):
        return False
    device = k_cache.device
    mask_tensor = torch.tensor(need_mask, dtype=torch.bool, device=device)

    
    """seq_idxs are the indices (0..batch_size-1) of sequences to compress in this batch."""
    seq_idxs = torch.nonzero(mask_tensor, as_tuple=True)[0]   
    m = seq_idxs.numel() # num of seq to compress

    """Gather Q/K/V for the sequences-to-compress computation.
    TritonGetQKVForComp returns:
      k_sub: (m, Hk, S, D)
      v_sub: (m, Hk, S, D)
    where S is the number of source tokens used by the compression algorithm.
    """
    # ---- Query Window 2.0: gather recent W queries from query_window_manager
    seq_ids = context.seq_ids
    assert seq_ids is not None, "context.seq_ids is None"

    compress_seq_ids = [seq_ids[i] for i in seq_idxs.tolist()]
    q_sub = query_window_manager.gather(compress_seq_ids, layer_id)  # [m, Hq, W, D]

    # ---- Gather K/V from paged KV cache
    k_sub, v_sub = TritonGetKVForComp(
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=context.block_tables,
        seq_idxs=seq_idxs,
        S=S,
    )

    """slots_flat[i, t] is the absolute slot (block_id*block_size + offset) of token t in sequence i."""
    bs, max_blocks = context.block_tables.size()
    offsets = torch.arange(block_size, device=device, dtype=torch.int64)
    slots_per_block = (context.block_tables.unsqueeze(-1).to(torch.int64) * block_size) + offsets  # (bs, max_blocks, block_size)
    slots_flat = slots_per_block.reshape(bs, max_blocks * block_size)  # (bs, max_blocks*block_size)

    """src_sub: slots of the most recent S tokens for each sequence to be compressed.
    Since sequences in the batch may be longer than S, for parallelism we always take the last S token slots as the compression window.
    In practice S is usually >= 511, so this often ends up equivalent to taking the first S token slots (because the sequence length is typically <= S 
    when compression is triggered)."""
    seq_lens = context.context_lens
    range_idx = torch.arange(S, device=device)
    start_idx = (seq_lens - S).clamp(min=0)
    indices = start_idx.unsqueeze(1) + range_idx          
    src_slots = torch.gather(slots_flat, 1, indices)      
    src_sub = src_slots[seq_idxs]  

    """Call the compression algorithm to obtain keep_idx.
    keep_idx with shape (m, R), containing indices in [0, S-1].
    Indices should be sorted (time order). If the algorithm does not guarantee it, sort here.
    """
    #Put your KV compress algorihtm here
    # keep_idx= StreamingLLM(q_sub, k_sub, v_sub,window = query_window_size,num_keep = R - query_window_size - 1)
    # keep_idx= SnapKV(q_sub, k_sub, v_sub,window = query_window_size,num_keep = R - query_window_size - 1)
    keep_idx= StrideKV(q_sub, k_sub, v_sub,window = query_window_size,num_keep = R - query_window_size - 1)
    if keep_idx is False:
        return False

    """for debug"""
    # if layer_id == 5 and q_sub.get_device() == 0:
        # print(context.context_lens,need_mask)
        # print("compressed--------------------------------------------------------")
    
    """Convert keep_idx (relative indices within the S) to absolute cache slots.
    src_idx: (m, R) absolute slots to read from.
    """
    src_idx = torch.gather(src_sub, 1, keep_idx)  # (m, R) int64
    src_flat = src_idx.reshape(-1).to(torch.int64)  # (m*R,)

    """move the kept R tokens into the first R slots of the kept blocks,avoid fragment memory.
    keep_blocks = ceil(R / block_size).
    dest_blocks are the first keep_blocks block id of each sequence.
    """
    keep_blocks = (R + block_size - 1) // block_size
    dest_blocks = context.block_tables[seq_idxs, :keep_blocks].to(torch.int64)  # (m, keep_blocks)
    
    """dst_flat are the absolute destination slots (first R slots in the kept blocks)."""
    offsets = torch.arange(block_size, device=device, dtype=torch.int64)  # (block_size,)
    dest_slots_per_block = (dest_blocks.unsqueeze(-1) * block_size) + offsets.view(1, 1, -1)  # (m, keep_blocks, block_size)
    dest_slots_flat = dest_slots_per_block.reshape(m, keep_blocks * block_size)  # (m, keep_blocks*block_size)
    dest_idx = dest_slots_flat[:, :R]  # (m, R) 
    dst_flat = dest_idx.reshape(-1).to(torch.int64)  # (m*R,)

    """Flatten caches for easier indexed load/store.
    k_cache/v_cache: (num_blocks, block_size, Hk, D) -> (num_blocks*block_size, Hk*D)
    q_cache:(num_blocks_q, block_size, Hq, Dq) -> (num_blocks_q*block_size, Hq*Dq)
    """
    num_blocks_k, bsize_k, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks_k * block_size
    D_k = num_kv_heads * head_dim
    k_flat = k_cache.reshape(total_slots, D_k)  # (total_slots, D_k)
    v_flat = v_cache.reshape(total_slots, D_k)


    """Physically move the kept rows (source -> destination).
    We use a two-phase approach (load to temp, then store) to avoid overwrite hazards.
    A pure torch implementation would be:
    """
    vals_k = k_flat.index_select(0, src_flat).clone()
    vals_v = v_flat.index_select(0, src_flat).clone()
    k_flat.index_copy_(0, dst_flat, vals_k)
    v_flat.index_copy_(0, dst_flat, vals_v)
    """After compaction, update context_lens for compressed sequences to R.
    This is critical so that:
      1) flash_attn_with_kvcache reads the correct KV range this step,
      2) next-step slot_mapping is computed consistently,
      3) block allocation/truncation logic uses the compressed length.
    NOTE: RoPE positions must be tracked separately (do NOT reuse context_lens for RoPE).
    """
    context.context_lens[seq_idxs] = R # !!!!!!
    
    if layer_id + 1 >= num_layers:
        freed_blocks = context.block_tables[seq_idxs, keep_blocks:]  # (m, max_blocks-keep_blocks)
        freed_blocks_cpu = freed_blocks.cpu().numpy().tolist()  # !!!!!
        """Append compression events to the global context variable.
        These events are consumed after the step to:
        - truncate seq.block_table,
        - update BlockManager refcounts/free lists,
        """
        if context.compression_events is None:
            context.compression_events = []
        for i, bidx in enumerate(seq_idxs.tolist()):
            ev = {
                "batch_index": int(bidx),
                "layer": int(layer_id),
                "R": int(R),
                "keep_blocks": int(keep_blocks),
                "freed_block_ids": [int(x) for x in freed_blocks_cpu[i] if int(x) >= 0]
            }
            context.compression_events.append(ev)
        for seq_id in compress_seq_ids:
            query_window_manager.free(seq_id)
    # Query Window 2.0: free recent-query buffers after one compression round
    return True


@triton.jit
def _gather_kv_chunk_kernel(
    k_flat_ptr,
    v_flat_ptr,
    out_k_ptr,
    out_v_ptr,
    block_tables_ptr,
    max_blocks: tl.constexpr,
    block_size: tl.constexpr,
    S: tl.constexpr,
    D_kv: tl.constexpr,
    chunk_size: tl.constexpr,
):
    seq_i = tl.program_id(0)
    chunk_i = tl.program_id(1)

    base_p = chunk_i * chunk_size
    offs = tl.arange(0, chunk_size)
    p_idx = base_p + offs
    valid_p = p_idx < S

    block_idx = p_idx // block_size
    offset_in_block = p_idx % block_size
    valid_block_idx = block_idx < max_blocks
    load_block_mask = valid_p & valid_block_idx

    base_row = seq_i * max_blocks
    block_addr = base_row + block_idx
    block_id = tl.load(block_tables_ptr + block_addr, mask=load_block_mask, other=-1)

    slot = block_id * block_size + offset_in_block
    valid_slot = (block_id >= 0) & load_block_mask

    offs_kv = tl.arange(0, D_kv)
    base_addr_kv = slot * D_kv

    k_rows = tl.load(k_flat_ptr + base_addr_kv[:, None] + offs_kv, mask=valid_slot[:, None], other=0.0)
    v_rows = tl.load(v_flat_ptr + base_addr_kv[:, None] + offs_kv, mask=valid_slot[:, None], other=0.0)

    out_base_idx = seq_i * S + p_idx
    out_base_addr = out_base_idx * D_kv
    tl.store(out_k_ptr + out_base_addr[:, None] + offs_kv, k_rows)
    tl.store(out_v_ptr + out_base_addr[:, None] + offs_kv, v_rows)

def TritonGetKVForComp(k_cache: torch.Tensor, v_cache: torch.Tensor,
                       block_tables: torch.Tensor, seq_idxs: torch.Tensor, S: int, chunk_size: int = 128):
    """
    Gather k_sub/v_sub for seq_idxs with chunked Triton kernel.
    Returns:
      k_sub: (m, num_kv_heads, S, head_dim)
      v_sub: (m, num_kv_heads, S, head_dim)
    """
    device = k_cache.device
    num_blocks_k, block_size_k, num_kv_heads, head_dim = k_cache.shape
    D_kv = num_kv_heads * head_dim
    total_k_slots = num_blocks_k * block_size_k

    seq_idxs = seq_idxs.to(device=device)
    m = seq_idxs.numel()

    k_flat = k_cache.contiguous().view(total_k_slots, D_kv)
    v_flat = v_cache.contiguous().view(total_k_slots, D_kv)

    out_k = torch.empty((m * S, D_kv), dtype=k_flat.dtype, device=device)
    out_v = torch.empty((m * S, D_kv), dtype=v_flat.dtype, device=device)

    sub_block_tables = block_tables[seq_idxs].contiguous()
    bs, max_blocks = sub_block_tables.shape
    num_chunks = (S + chunk_size - 1) // chunk_size
    grid = (m, num_chunks)
    _gather_kv_chunk_kernel[grid](
        k_flat, v_flat,
        out_k, out_v,
        sub_block_tables, max_blocks, block_size_k, S, D_kv, chunk_size
    )

    k_sub = out_k.view(m, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v_sub = out_v.view(m, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return k_sub, v_sub

@triton.jit
def _gather_kv_rows_kernel(
    k_ptr, v_ptr,              # [total_slots, D_k] flattened 2D contiguous
    src_idx_ptr,               # [num_items]
    temp_k_ptr, temp_v_ptr,    # [num_items, D_k] flattened contiguous
    D_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)   # each program handles one source row

    src = tl.load(src_idx_ptr + pid)
    offs = tl.arange(0, BLOCK_D)

    for d in range(0, D_k, BLOCK_D):
        off = d + offs
        mask = off < D_k

        src_addr = src * D_k + off
        tmp_addr = pid * D_k + off

        k_vals = tl.load(k_ptr + src_addr, mask=mask, other=0.0)
        v_vals = tl.load(v_ptr + src_addr, mask=mask, other=0.0)

        tl.store(temp_k_ptr + tmp_addr, k_vals, mask=mask)
        tl.store(temp_v_ptr + tmp_addr, v_vals, mask=mask)

# Triton kernel: store (scatter) temp rows into dst slots
@triton.jit
def _scatter_kv_rows_kernel(
    k_ptr, v_ptr,              # [total_slots, D_k]
    dst_idx_ptr,               # [num_items]
    temp_k_ptr, temp_v_ptr,    # [num_items, D_k]
    D_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)   # each program handles one destination row

    dst = tl.load(dst_idx_ptr + pid)
    offs = tl.arange(0, BLOCK_D)

    for d in range(0, D_k, BLOCK_D):
        off = d + offs
        mask = off < D_k

        dst_addr = dst * D_k + off
        tmp_addr = pid * D_k + off

        k_vals = tl.load(temp_k_ptr + tmp_addr, mask=mask, other=0.0)
        v_vals = tl.load(temp_v_ptr + tmp_addr, mask=mask, other=0.0)

        tl.store(k_ptr + dst_addr, k_vals, mask=mask)
        tl.store(v_ptr + dst_addr, v_vals, mask=mask)


@triton.jit
def store_qkvcache_kernel(
    key_ptr,
    value_ptr,
    query_ptr,
    q_cache_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D_kv: tl.constexpr,
    D_q_out: tl.constexpr,
    orig_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    stored_heads: tl.constexpr,
    q_dtype_flag: tl.constexpr,   # 0=float32,1=float16,2=bfloat16
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    offs_kv = tl.arange(0, D_kv)
    key_row = tl.load(key_ptr + idx * D_kv + offs_kv)
    val_row = tl.load(value_ptr + idx * D_kv + offs_kv)
    k_cache_pos = slot * D_kv + offs_kv
    tl.store(k_cache_ptr + k_cache_pos, key_row)
    tl.store(v_cache_ptr + k_cache_pos, val_row)

    # Query aggregation: accumulate in float32
    base_q_in = idx * (orig_heads * head_dim)
    for oh in range(stored_heads):
        acc = tl.zeros((head_dim,), dtype=tl.float32)
        in_start = oh * group_size
        for g in range(group_size):
            in_head = in_start + g
            q_in_offsets = base_q_in + in_head * head_dim + tl.arange(0, head_dim)
            q_vals = tl.load(query_ptr + q_in_offsets)
            acc = acc + q_vals.to(tl.float32)
        acc = acc / group_size

        q_out_pos = slot * D_q_out + oh * head_dim + tl.arange(0, head_dim)
        # write back according to q dtype
        if q_dtype_flag == 1:
            # store as fp16
            tl.store(q_cache_ptr + q_out_pos, acc.to(tl.float16))
        elif q_dtype_flag == 2:
            # store as bfloat16
            tl.store(q_cache_ptr + q_out_pos, acc.to(tl.bfloat16))
        else:
            # store as float32
            tl.store(q_cache_ptr + q_out_pos, acc)

def store_qkvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                   slot_mapping: torch.Tensor, query: torch.Tensor, q_cache: torch.Tensor):
    N = key.size(0)
    orig_heads = query.size(1)
    head_dim = query.size(2)
    group_size = query.size(1) // key.size(1)
    stored_heads = orig_heads // group_size

    D_kv = key.size(1) * head_dim
    D_q_out = stored_heads * head_dim

    key_flat = key.contiguous().view(N, D_kv)
    value_flat = value.contiguous().view(N, D_kv)

    # Ensure query_flat has the same dtype as q_cache to avoid implicit casts
    target_q_dtype = q_cache.dtype
    query_flat = query.contiguous().to(target_q_dtype).view(N, orig_heads * head_dim)

    k_cache_flat = k_cache.contiguous().view(-1, D_kv)
    v_cache_flat = v_cache.contiguous().view(-1, D_kv)
    q_cache_flat = q_cache.contiguous().view(-1, D_q_out)

    # set flag for kernel: 0=float32,1=float16,2=bfloat16
    if target_q_dtype == torch.float16:
        q_dtype_flag = 1
    elif target_q_dtype == torch.bfloat16:
        q_dtype_flag = 2
    else:
        q_dtype_flag = 0

    # call kernel: pass strides if needed (or as in your earlier version)
    store_qkvcache_kernel[(N,)](
        key_flat,
        value_flat,
        query_flat,
        q_cache_flat,
        k_cache_flat,
        v_cache_flat,
        slot_mapping,
        D_kv, D_q_out,
        orig_heads, group_size, head_dim, stored_heads,
        q_dtype_flag
    )
