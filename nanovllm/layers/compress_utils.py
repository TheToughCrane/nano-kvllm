import torch
from torch import nn
import triton
import triton.language as tl
from nanovllm.utils.context import get_context
from nanovllm.layers.CompressMethod import MyCompressHeadWise,SnapKV,MyCompressHeadWise_BlockDiag



def MyCompressCompact(q_cache: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                      layer_id: int, block_size: int, S: int, R: int):
    
    """
    KV/Q cache compress&compaction for decode phase.

    Inputs:
      - q_cache: [num_blocks_q, block_size, num_heads_q, head_dim] query cache for this layer+rank
      - k_cache: [num_blocks_k, block_size, num_kv_heads, head_dim] key cache for this layer+rank
      - v_cache: [num_blocks_k, block_size, num_kv_heads, head_dim] value cache for this layer+rank
      - layer_id: int, current layer id (used for recording compression events)
      - block_size: int, KV block size (e.g., 256)
      - S: int, compression window length (take last S tokens for each sequence)
      - R: int, number of tokens to keep after compression (R <= S)
 
    Outputs:
      - If compression happens: in-place compact/move kept Q/K/V vectors into the first R slots of the kept blocks,
        update context.context_lens for those sequences to R, and append events to context.compression_events.
      - Returns True if compression executed; returns False if algorithm decides to skip;
    """

    context = get_context()

    """Do not compress during prefill.KV compression is only applied in decode.
    """
    if context.is_prefill or context.context_lens is None or context.block_tables is None:
        return False
    
    """Fast early-exit.
        In prepare_decode() we precompute whether any sequence reaches the compression threshold.
        Doing the threshold check inside here and every attention layer would add ~10%~20% decode overhead(throughput down).
    """    
    if not getattr(context, "compress_any", False):
        return False
    
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
      q_sub: (m, Hq, S, Dq)
      k_sub: (m, Hk, S, D)
      v_sub: (m, Hk, S, D)
    where S is the number of source tokens used by the compression algorithm.
    """
    q_sub, k_sub, v_sub = TritonGetQKVForComp(
        k_cache=k_cache, v_cache=v_cache, q_cache=q_cache,
        block_tables=context.block_tables, seq_idxs=seq_idxs, S=S
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
    keep_idx= SnapKV(q_sub, k_sub, v_sub,window = 30,num_keep = R - 30 - 1)
    if keep_idx is False:
        return False

    """for debug"""
    if layer_id == 3 and q_sub.get_device() == 0:
        print("compressed--------------------------------------------------------")
    
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
    num_blocks_q, bsize_q, num_heads_q, head_dim_q = q_cache.shape
    total_slots_q = num_blocks_q * block_size  # q_cache num_blocks may differ
    D_q = num_heads_q * head_dim_q
    q_flat = q_cache.reshape(total_slots_q, D_q)  # (total_slots, D_q)

    """Physically move the kept rows (source -> destination).
    We use a two-phase approach (load to temp, then store) to avoid overwrite hazards.
    A pure torch implementation would be:
    """
    # vals_k = k_flat.index_select(0, src_flat).clone()
    # vals_v = v_flat.index_select(0, src_flat).clone()
    # vals_q = q_flat.index_select(0, src_flat).clone()

    # k_flat.index_copy_(0, dst_flat, vals_k)
    # v_flat.index_copy_(0, dst_flat, vals_v)
    # q_flat.index_copy_(0, dst_flat, vals_q)

    """triton version of cache movement, same function as above"""
    num_items = src_flat.numel()
    temp_k = torch.empty((num_items * D_k,), dtype=k_flat.dtype, device=device)
    temp_v = torch.empty((num_items * D_k,), dtype=v_flat.dtype, device=device)
    temp_q = torch.empty((num_items * D_q,), dtype=q_flat.dtype, device=device)
    # Choose block sizes for inner loops in kernel
    BLOCK_D_k = 64  
    BLOCK_D_q = 64
    # launch load kernel: grid = (num_items,)
    _triton_load_src_kernel[(num_items,)](
        k_flat, v_flat, q_flat,
        src_flat,
        temp_k, temp_v, temp_q,
        D_k, BLOCK_D_k,
        D_q, BLOCK_D_q,
        total_slots_q
    )
    # launch store kernel: grid = (num_items,)
    _triton_store_dst_kernel[(num_items,)](
        k_flat, v_flat, q_flat,
        dst_flat,
        temp_k, temp_v, temp_q,
        D_k, BLOCK_D_k,
        D_q, BLOCK_D_q,
        total_slots_q
    )

    """After compaction, update context_lens for compressed sequences to R.
    This is critical so that:
      1) flash_attn_with_kvcache reads the correct KV range this step,
      2) next-step slot_mapping is computed consistently,
      3) block allocation/truncation logic uses the compressed length.
    NOTE: RoPE positions must be tracked separately (do NOT reuse context_lens for RoPE).
    """
    context.context_lens[seq_idxs] = R # !!!!!!
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

    return True


@triton.jit
def _gather_qkv_chunk_kernel(
    k_flat_ptr,
    v_flat_ptr,
    q_flat_ptr,
    out_k_ptr,
    out_v_ptr,
    out_q_ptr,
    block_tables_ptr,   
    max_blocks: tl.constexpr,
    block_size: tl.constexpr,
    S: tl.constexpr,
    D_kv: tl.constexpr,
    total_q_slots: tl.constexpr,
    chunk_size: tl.constexpr,
):
    seq_i = tl.program_id(0)   # 0..m-1 index into sub_block_tables
    chunk_i = tl.program_id(1)

    base_p = chunk_i * chunk_size
    offs = tl.arange(0, chunk_size)
    p_idx = base_p + offs
    valid_p = p_idx < S

    block_idx = p_idx // block_size
    offset_in_block = p_idx%block_size
    valid_block_idx = block_idx < max_blocks
    load_block_mask = valid_p & valid_block_idx

    base_row = seq_i * max_blocks
    block_addr = base_row + block_idx  # vector length chunk_size
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

    valid_q = valid_slot & (slot < total_q_slots)
    offs_q = tl.arange(0, D_kv)
    q_rows = tl.load(q_flat_ptr + (slot * D_kv)[:, None] + offs_q, mask=valid_q[:, None], other=0.0)
    out_base_addr_q = (seq_i * S + p_idx) * D_kv
    tl.store(out_q_ptr + out_base_addr_q[:, None] + offs_q, q_rows)

def TritonGetQKVForComp(k_cache: torch.Tensor, v_cache: torch.Tensor, q_cache: torch.Tensor,
                                  block_tables: torch.Tensor, seq_idxs: torch.Tensor, S: int, chunk_size: int = 128):
    """
    Gather q_sub/k_sub/v_sub for seq_idxs with chunked Triton kernel.
    Returns:
      q_sub: (m, num_heads_q, S, head_dim_q)
      k_sub: (m, num_kv_heads, S, head_dim)
      v_sub: (m, num_kv_heads, S, head_dim)
    """
    device = k_cache.device
    num_blocks_k, block_size_k, num_kv_heads, head_dim = k_cache.shape
    D_kv = num_kv_heads * head_dim
    total_k_slots = num_blocks_k * block_size_k

    num_blocks_q, block_size_q, num_heads_q, head_dim_q = q_cache.shape
    total_q_slots = num_blocks_q * block_size_q

    seq_idxs = seq_idxs.to(device=device)
    m = seq_idxs.numel()

    k_flat = k_cache.contiguous().view(total_k_slots, D_kv)
    v_flat = v_cache.contiguous().view(total_k_slots, D_kv)
    q_flat = q_cache.contiguous().view(total_q_slots, D_kv)

    out_k = torch.empty((m * S, D_kv), dtype=k_flat.dtype, device=device)
    out_v = torch.empty((m * S, D_kv), dtype=v_flat.dtype, device=device)
    out_q = torch.empty((m * S, D_kv), dtype=q_flat.dtype, device=device)

    sub_block_tables = block_tables[seq_idxs].contiguous()
    bs, max_blocks = sub_block_tables.shape
    num_chunks = (S + chunk_size - 1) // chunk_size
    grid = (m, num_chunks)
    _gather_qkv_chunk_kernel[grid](
        k_flat, v_flat, q_flat,
        out_k, out_v, out_q,
        sub_block_tables,max_blocks, block_size_k, S, D_kv, total_q_slots, chunk_size
    )

    k_sub = out_k.view(m, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v_sub = out_v.view(m, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    q_sub = out_q.view(m, S, num_heads_q, head_dim_q).permute(0, 2, 1, 3).contiguous()
    return q_sub, k_sub, v_sub

@triton.jit
def _triton_load_src_kernel(
    k_ptr, v_ptr, q_ptr,   # flattened global caches
    src_idx_ptr,          # int32 indices (num_items,)
    temp_k_ptr, temp_v_ptr, temp_q_ptr,  # flattened temp buffers (num_items * D)
    D_k: tl.constexpr, BLOCK_D_k: tl.constexpr,
    D_q: tl.constexpr, BLOCK_D_q: tl.constexpr,
    total_q_slots: tl.constexpr
):
    pid = tl.program_id(0)                       # 每个 program 负责一个 src-item
    # load src index (int32)
    src = tl.load(src_idx_ptr + pid)
    # compute base addresses
    base_k = src * D_k
    offs_k = tl.arange(0, BLOCK_D_k)
    # loop over k/v elements in blocks of BLOCK_D_k
    for d in range(0, D_k, BLOCK_D_k):
        off = d + offs_k
        mask = off < D_k
        src_off = base_k + off
        vals_k = tl.load(k_ptr + src_off, mask=mask, other=0.0)
        tl.store(temp_k_ptr + pid * D_k + off, vals_k, mask=mask)
        vals_v = tl.load(v_ptr + src_off, mask=mask, other=0.0)
        tl.store(temp_v_ptr + pid * D_k + off, vals_v, mask=mask)

    # q may have different D_q; check src < total_q_slots before reading
    base_q = src * D_q
    offs_q = tl.arange(0, BLOCK_D_q)
    for d in range(0, D_q, BLOCK_D_q):
        off = d + offs_q
        mask_q = off < D_q
        src_q_off = base_q + off
        # valid_q_mask: src < total_q_slots
        valid_q = src < total_q_slots
        vals_q = tl.load(q_ptr + src_q_off, mask=(mask_q & valid_q), other=0.0)
        tl.store(temp_q_ptr + pid * D_q + off, vals_q, mask=mask_q & valid_q)

# Triton kernel: store (scatter) temp rows into dst slots
@triton.jit
def _triton_store_dst_kernel(
    k_ptr, v_ptr, q_ptr,   # flattened global caches
    dst_idx_ptr,           # int32 indices (num_items,)
    temp_k_ptr, temp_v_ptr, temp_q_ptr,  # flattened temp buffers (num_items * D)
    D_k: tl.constexpr, BLOCK_D_k: tl.constexpr,
    D_q: tl.constexpr, BLOCK_D_q: tl.constexpr,
    total_q_slots: tl.constexpr
):
    pid = tl.program_id(0)
    dst = tl.load(dst_idx_ptr + pid)
    base_k = dst * D_k
    offs_k = tl.arange(0, BLOCK_D_k)
    for d in range(0, D_k, BLOCK_D_k):
        off = d + offs_k
        mask = off < D_k
        dst_off = base_k + off
        vals_k = tl.load(temp_k_ptr + pid * D_k + off, mask=mask, other=0.0)
        tl.store(k_ptr + dst_off, vals_k, mask=mask)
        vals_v = tl.load(temp_v_ptr + pid * D_k + off, mask=mask, other=0.0)
        tl.store(v_ptr + dst_off, vals_v, mask=mask)

    # q write: bound check for dst < total_q_slots
    base_q = dst * D_q
    offs_q = tl.arange(0, BLOCK_D_q)
    for d in range(0, D_q, BLOCK_D_q):
        off = d + offs_q
        mask_q = off < D_q
        dst_q_off = base_q + off
        valid_q = dst < total_q_slots
        vals_q = tl.load(temp_q_ptr + pid * D_q + off, mask=mask_q, other=0.0)
        tl.store(q_ptr + dst_q_off, vals_q, mask=mask_q & valid_q)


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



def GetQKVForComp(q_cache_layer: torch.Tensor, k_cache_layer: torch.Tensor, v_cache_layer: torch.Tensor,
                                block_tables: torch.Tensor, context_lens: torch.Tensor, S: int):
    device = k_cache_layer.device
    num_blocks, block_size, _, head_dim = k_cache_layer.shape
    bs = block_tables.size(0)
    max_blocks = block_tables.size(1)

    offsets = torch.arange(block_size, device=device, dtype=torch.int64)  


    block_ids = block_tables.to(dtype=torch.int64, device=device)  
    abs_slots = block_ids.unsqueeze(-1) * block_size + offsets.view(1, 1, -1)  
    abs_slots_flat = abs_slots.view(bs, max_blocks * block_size)  

    src_slots = abs_slots_flat[:, :S].contiguous()  

    k_flat = k_cache_layer.view(-1, k_cache_layer.size(2), head_dim)  
    v_flat = v_cache_layer.view(-1, v_cache_layer.size(2), head_dim)
    q_flat = q_cache_layer.view(-1, q_cache_layer.size(2), head_dim)  

    k_batch = k_flat[src_slots]  # (bs, S, num_kv_heads, head_dim)
    v_batch = v_flat[src_slots]
    q_batch = q_flat[src_slots]  #(bs, S, num_heads_q, head_dim)


    k_batch = k_batch.permute(0, 2, 1, 3).contiguous()  # (bs, num_kv_heads, S, head_dim)
    v_batch = v_batch.permute(0, 2, 1, 3).contiguous()
    q_batch = q_batch.permute(0, 2, 1, 3).contiguous()  # (bs, num_heads_q, S, head_dim)

    return q_batch, k_batch, v_batch, src_slots