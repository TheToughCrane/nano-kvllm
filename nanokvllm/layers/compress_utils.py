import torch
from torch import nn
import triton
import triton.language as tl
from nanokvllm.utils.context import get_context
from nanokvllm.layers.CompressMethod import SnapKV


def get_tail_window_and_tail_slots(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    seq_idxs: torch.Tensor,
    block_size: int,
    window_blocks: int,
):
    """
    Vectorized helper.

    For each selected seq:
      - locate the last `window_blocks` full blocks at the tail
      - if there is an extra tail partial block, identify its block id and length

    Returns:
      window_src_slots: [m, window_blocks * block_size]
      old_context_lens: [m]
      tail_lens: [m]
      tail_block_ids: [m]   # valid only where tail_lens > 0, otherwise forced to -1
    """
    device = block_tables.device
    old_context_lens = context_lens.index_select(0, seq_idxs).to(torch.long)          # [m]
    selected_block_tables = block_tables.index_select(0, seq_idxs).to(torch.long)     # [m, max_blocks]

    B = block_size
    m = seq_idxs.numel()

    full_blocks = old_context_lens // B                                # [m]
    tail_lens = old_context_lens % B                                   # [m]

    assert torch.all(full_blocks >= window_blocks), (
        f"some full_blocks < window_blocks: {full_blocks}"
    )

    # indices of the last `window_blocks` full blocks
    block_offsets = torch.arange(window_blocks, device=device, dtype=torch.long).view(1, -1)  # [1, wb]
    window_block_idx = (full_blocks - window_blocks).unsqueeze(1) + block_offsets             # [m, wb]

    window_block_ids = torch.gather(selected_block_tables, 1, window_block_idx)               # [m, wb]
    assert torch.all(window_block_ids >= 0), "window_block_ids contains invalid block id"

    # expand block ids to absolute slots
    token_offsets = torch.arange(B, device=device, dtype=torch.long).view(1, 1, B)            # [1,1,B]
    window_src_slots = window_block_ids.unsqueeze(-1) * B + token_offsets                      # [m, wb, B]
    window_src_slots = window_src_slots.reshape(m, window_blocks * B)                          # [m, wb*B]

    # optional tail partial block: its block index is `full_blocks`
    max_blocks = selected_block_tables.size(1)
    safe_tail_block_idx = torch.clamp(full_blocks, max=max_blocks - 1)
    tail_block_ids = torch.gather(
        selected_block_tables,
        1,
        safe_tail_block_idx.unsqueeze(1),
    ).squeeze(1)   # [m]

    # IMPORTANT:
    # if a seq has no tail partial block, force tail_block_id to -1
    tail_block_ids = torch.where(
        tail_lens > 0,
        tail_block_ids,
        torch.full_like(tail_block_ids, -1),
    )

    return window_src_slots, old_context_lens, tail_lens, tail_block_ids


def gather_kv_by_slots(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    src_slots: torch.Tensor,
):
    """
    src_slots: [m, S]
    returns:
      k_sub: [m, Hk, S, D]
      v_sub: [m, Hk, S, D]
    """
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks * block_size

    k_flat = k_cache.view(total_slots, num_kv_heads, head_dim)
    v_flat = v_cache.view(total_slots, num_kv_heads, head_dim)

    k_batch = k_flat[src_slots]  # [m, S, Hk, D]
    v_batch = v_flat[src_slots]

    k_batch = k_batch.permute(0, 2, 1, 3).contiguous()
    v_batch = v_batch.permute(0, 2, 1, 3).contiguous()
    return k_batch, v_batch


def MyCompressCompact(
    q_current: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    layer_id: int,
    block_size: int,
    window_blocks: int,
    keep_blocks: int,
    keep_extra_tokens: int,
    num_layers: int,
    context=None,
):
    """
    Periodic compression version:
      - use current q of this decode step
      - compress the last `window_blocks` full blocks at tail
      - if there is a tail partial block, move it after the compressed kept tokens

    Current implementation strategy:
      - src_keep comes from selected tokens inside the full-block compression window
      - dst_keep uses the first keep_tokens absolute slots in that window region
      - tail src/dst are flattened and processed in parallel
      - torch index_select/index_copy_ is used for safe KV compaction
    """
    if context is None:
        context = get_context()

    if context.is_prefill or context.context_lens is None or context.block_tables is None:
        return False

    selected_batch_indices = context.compress_selected_batch_indices
    if not selected_batch_indices:
        return False

    device = k_cache.device
    seq_idxs = torch.tensor(selected_batch_indices, dtype=torch.long, device=device)
    m = seq_idxs.numel()
    if m == 0:
        return False

    B = block_size
    window_tokens = window_blocks * B
    keep_tokens = keep_blocks * B + keep_extra_tokens

    # Current q of this step: [bsz, Hq, D] -> selected subset -> [m, Hq, 1, D]
    q_sub = q_current.index_select(0, seq_idxs)
    q_sub = q_sub.unsqueeze(2)

    # IMPORTANT:
    # use the base context lens captured before any layer modifies context.context_lens in this step
    base_context_lens = context.compress_base_context_lens
    assert base_context_lens is not None, "compress_base_context_lens is None during compress step"

    # Gather tail compression window (last `window_blocks` full blocks) and optional tail partial block
    window_src_slots, old_context_lens, tail_lens, tail_block_ids = get_tail_window_and_tail_slots(
        block_tables=context.block_tables,
        context_lens=base_context_lens,
        seq_idxs=seq_idxs,
        block_size=B,
        window_blocks=window_blocks,
    )

    # Gather K/V for the full-block compression window
    k_sub, v_sub = gather_kv_by_slots(k_cache, v_cache, window_src_slots)

    # Compression algorithm
    keep_idx = SnapKV(
        q_sub,
        k_sub,
        v_sub,
        num_keep=keep_tokens - 1 - 1,   # keep_tokens = bos + selected + latest-token policy
        window=1,
    )
    if keep_idx is False:
        return False

    # Sanity checks
    assert keep_idx.dim() == 2
    assert keep_idx.size(0) == m
    assert keep_idx.size(1) == keep_tokens
    assert keep_idx.min().item() >= 0
    assert keep_idx.max().item() < window_tokens, (
        f"keep_idx out of range: max={keep_idx.max().item()}, window_tokens={window_tokens}"
    )

    # Absolute kept slots inside the full-block window
    src_keep = torch.gather(window_src_slots, 1, keep_idx)   # [m, keep_tokens]

    # New cache length after compression
    new_context_lens_tensor = old_context_lens - window_tokens + keep_tokens   # [m]

    # ------------------------------------------------------------------
    # Destination construction
    #
    # keep-part:
    #   write kept tokens into the first keep_tokens absolute slots of the
    #   compression window region
    #
    # tail-part:
    #   if tail_len > 0, write tail tokens immediately after the kept region
    # ------------------------------------------------------------------
    dst_keep = window_src_slots[:, :keep_tokens]   # [m, keep_tokens]

    src_keep_flat = src_keep.reshape(-1)
    dst_keep_flat = dst_keep.reshape(-1)

    # Tail-part source/destination slots (flattened)
    tail_total = int(tail_lens.sum().item())

    if tail_total > 0:
        # For each tail token, which seq does it belong to?
        tail_seq_ids = torch.repeat_interleave(
            torch.arange(m, device=device, dtype=torch.long),
            tail_lens
        )   # [tail_total]

        # Per-token offset inside each seq's tail partial block
        # This still uses a small Python list over m sequences only.
        tail_offsets = torch.cat(
            [torch.arange(int(t.item()), device=device, dtype=torch.long) for t in tail_lens]
        )   # [tail_total]

        # only seqs with tail_len > 0 participate here
        assert torch.all(tail_block_ids[tail_seq_ids] >= 0), "invalid tail_block_id used"

        # Tail source slots: absolute slots inside each seq's tail partial block
        src_tail_flat = tail_block_ids[tail_seq_ids] * B + tail_offsets

        # Tail destination starts right after the last kept slot of each seq
        dst_tail_start = dst_keep[:, -1] + 1   # [m]
        dst_tail_flat = dst_tail_start[tail_seq_ids] + tail_offsets

        src_flat = torch.cat([src_keep_flat, src_tail_flat], dim=0)
        dst_flat = torch.cat([dst_keep_flat, dst_tail_flat], dim=0)
    else:
        src_flat = src_keep_flat
        dst_flat = dst_keep_flat

    # Flatten KV cache
    num_blocks_k, _, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks_k * block_size
    D_k = num_kv_heads * head_dim
    k_flat = k_cache.reshape(total_slots, D_k)
    v_flat = v_cache.reshape(total_slots, D_k)

    # Index sanity
    assert src_flat.min().item() >= 0
    assert src_flat.max().item() < total_slots, (
        f"src_flat out of range: max={src_flat.max().item()}, total_slots={total_slots}"
    )
    assert dst_flat.min().item() >= 0
    assert dst_flat.max().item() < total_slots, (
        f"dst_flat out of range: max={dst_flat.max().item()}, total_slots={total_slots}"
    )

    # Stable torch compact
    vals_k = k_flat.index_select(0, src_flat).clone()
    vals_v = v_flat.index_select(0, src_flat).clone()
    k_flat.index_copy_(0, dst_flat, vals_k)
    v_flat.index_copy_(0, dst_flat, vals_v)

    # Update context_lens for current and subsequent flash attention
    context.context_lens[seq_idxs] = new_context_lens_tensor.to(context.context_lens.dtype)
    

    # Record events only on the last layer
    if layer_id + 1 >= num_layers:
        selected_block_tables = context.block_tables.index_select(0, seq_idxs).to(torch.long)

        if context.compression_events is None:
            context.compression_events = []

        keep_blocks_after_tensor = (new_context_lens_tensor + B - 1) // B   # [m]
        seq_idxs = seq_idxs.tolist()
        for i, bidx in enumerate(seq_idxs):
            keep_blocks_after = int(keep_blocks_after_tensor[i].item())
            freed_blocks = selected_block_tables[i, keep_blocks_after:]
            freed_block_ids = [int(x) for x in freed_blocks.tolist() if int(x) >= 0]

            ev = {
                "batch_index": int(bidx),
                "layer": int(layer_id),
                "new_context_len": int(new_context_lens_tensor[i].item()),
                "keep_blocks": int(keep_blocks_after),
                "freed_block_ids": freed_block_ids,
                "tail_uncompressed_len_after": 0,
            }
            context.compression_events.append(ev)

    return True