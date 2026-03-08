# `nano-kvllm`
## Table of Contents

- [Quick Start](#quick-start)
- [What is nano-vllm?](#what-is-nano-vllm)
- [nano-vllm execution flow (simplified)](#nano-vllm-execution-flow-simplified)
- [What is KV Cache compression, and why is it hard to deploy?](#what-is-kv-cache-compression-and-why-is-it-hard-to-deploy)
- [Compression strategy in nano-kvllm](#compression-strategy-in-nano-kvllm)
- [Code modifications (high-level)](#code-modifications-high-level)
- [Key notes (must read)](#key-notes-must-read)
> `nano-kvllm` integrates **KV Cache Compression** into `nano-vllm` while keeping the original `nano-vllm` code layout **as unchanged as possible**.  
> The goal of `nano-kvllm` is to enable long-context inference with **lower KV-cache memory usage** and **higher decode throughput**, while maintaining generation quality.
>
> Compared with the original `nano-vllm`, this project `nano-kvllm` adds about **100 lines of code** and introduces **2 new files** (`layers/compress_utils.py` and `layers/compress_method.py`).
>
> some project has implement KV compression in nano-vllm before, but a key idea of `nano-kvllm` is to keep the source `nano-vllm` layout minimally modified, so readers can simultaneously understand both the core `nano-vllm` mechanisms and practical KV-compression integration.
## Quick Start

1. **Implement your own KV compression function** in:

```bash
layers/compress_method.py
```

2. run the example script

```bash
bash examples.sh
```

**Tips:** In postprocess function of `scheduler.py`,  you can use following code to view generated text in real-time：

```python
print(self.tokenizer.decode(seq.token_ids))
```
3. set compress hyper-parameters in nano-kvllm/config.py
```python
 kv_compress_enabled: bool = True       
 kv_compress_N: int = 2                 
 kv_compress_S: int = kvcache_block_size * (kv_compress_N + 1) -1    
 kv_compress_R: int = kvcache_block_size * kv_compress_N + 1      
 num_query_blocks:int = 200
```
### 
---

## What is nano-vllm?

`nano-vllm` is a lightweight implementation of vLLM that preserves core inference mechanisms:

- PagedAttention (block-based KV cache management)
- FlashAttention (efficient prefill/decode attention)
- TP (Tensor Parallel) execution pipeline

At the same time, it keeps the codebase compact and easy to read/extend.

## nano-vllm execution flow (simplified)

```text
LLMEngine.step()  (loop until all sequences finish)
   ├── Scheduler.schedule()
   │     └── select seqs for this decode step
   │
   ├── ModelRunner.run(seqs, is_prefill=False)
   │     ├── prepare_decode(seqs)
   │     │     ├── build input_ids / positions
   │     │     ├── build slot_mapping / context_lens / block_tables
   │     │     └── precompute compress_need_mask (new)
   │     ├── model.forward(...)
   │     │     └── Attention.forward(...)
   │     │           ├── store_kvcache(...)
   │     │           ├── KVCacheCompress(...)   # KV compression logic
   │     │           └── flash_attn_with_kvcache(...)
   │     └── return token_ids + compression_events
   │
   ├── Scheduler.postprocess(seqs, token_ids, compression_events)
   │     ├── append_token / finish checks
   │     ├── update sequence metadata from compression_events
   │     └── release redundant blocks (BlockManager)
   │
   └── next step() iteration for decode
```
### Key files in nano-vllm

- `llm_engine.py`: inference entry point; handles request submission, step loop, scheduling/execution orchestration  
- `scheduler.py`: batching/lifecycle management (`waiting`/`running` queues, preemption, postprocess)  
- `sequence.py`: request object/state definition (tokens, block_table, finish status, etc.)  
- `block_manager.py`: PagedAttention block allocation/release and metadata (including hash reuse)  
- `model_runner.py`: single-/multi-GPU runner; handles input prep, context construction, model forward, sampling, and process communication

> Note: `nano-vllm` captures core vLLM ideas but **does not include PD separation**; by default, all sequences in a batch finish prefill before entering decode together.

---

## What is KV Cache compression, and why is it hard to deploy?

Basic KV-compression pattern:  
When sequence length reaches threshold `S`, select `R (R <= S)` important tokens from the `S` tokens and rebuild KV cache. Subsequent decode continues on this compressed cache.

### Deployment challenges (in vLLM / nano-vllm)

1. Must remain compatible with PagedAttention block mapping (`slot_mapping`, `block_tables`, `context_lens`)
2. Requires **physical memory compaction** after compression; otherwise memory fragmentation and invalid-slot accesses occur
3. Must keep FlashAttention input semantics consistent (no OOB / misalignment)
4. Under TP parallelism, metadata must stay synchronized across ranks

---

## Compression strategy in nano-kvllm

- **Decode-only compression**: compression is applied only during decode (not prefill)
- Why:
  - Decode is the critical bottleneck path
  - Modifying prefill and decode simultaneously increases engineering complexity and bug risk
  - Prefill compression may hurt metrics such as TTFT, thus lower ROI in this stage

- **Trigger policy**: compress from `S` to `R` when sequence length reaches `S`  
  This project uses  
  **`S = block_size * (N + 1)`**, **`R = block_size * N + 1`**  
  (e.g., `N=1`, `R=257`, `S=511`)  
  to reduce repeated block allocate/free cycles and avoid unnecessary prefix-hash recomputation.

---

## Code modifications (high-level)

### 1) `layers/attention.py`

Compression is inserted in `Attention.forward()` as:

```python
store_kvcache(...)
# [new] KVCacheCompress(...)
flash_attn_with_kvcache(...)
```

That is: **store new token KV first, compress second, run attention third**.

#### Original key variables

- `slot_mapping`: token -> global KV physical slot mapping (used by `store_kvcache`)
- `context_lens`: effective attention length per sequence (used by `flash_attn_with_kvcache`)
- `block_tables`: logical block -> global block-id mapping per sequence (used by `flash_attn_with_kvcache`)

#### Added logic

1. Determine which sequences in the batch meet compression conditions; extract target KV using `context_lens` and `block_tables`
2. Run compression algorithm to obtain `keep_idx`; compact retained KV at the **GPU physical-memory level** to avoid fragmentation
3. Update `context_lens` to `R`, and record `R` and `freed_block_ids` into global events for main-rank postprocessing

---

### 2) `engine/model_runner.py`

#### `prepare_decode()` changes

Before each decode step, precompute whether any sequence meets compression conditions, avoiding repeated per-layer checks in attention.  
This reduces hot-path synchronization overhead and gives about **5%–10% TPS improvement** in practice.

#### `run()` changes

Return compression events (e.g., `batch_index / R / keep_blocks / freed_block_ids`) to `scheduler.py` for unified metadata update.

---

### 3) `engine/scheduler.py`

In `postprocess()`, apply compression events to update:

- physical-length-related fields (e.g., `num_tokens`)
- `block_table`
- release unused blocks via BlockManager

---

### 4) `engine/block_manager.py`

Add truncation/release utility for compression scenario (`truncate_blocks(...)`):

- release tail blocks no longer needed after compression
- mark the last kept block with non-full semantics (hash-state handling) to avoid append assertion failures

---

### 5) New file: `layers/compress_utils.py`

Contains core tools:

- main compression pipeline function `MyCompressCompact`
- Triton-based KV extraction kernels
- Triton-based physical-memory compaction kernels (compact write-back of retained KV)

---

### 6) New file: `layers/compress_method.py`

Defines compression algorithm interface.  
Users only need to implement:

```python
keep_idx = compress_fn(k_cache, v_cache, **kwargs)
```

- input: `[bsz, heads, seq_len, head_dim]`
- output: per-sequence retained indices `keep_idx`
- recommendation: keep a recent-token window (e.g., last `K=10`) for generation stability

---

## Key notes (must read)

### 1) TP parallel compatibility

- Each rank compresses its local heads locally
- **All ranks must use the same `R`** to keep `context_lens/slot_mapping` consistent across ranks
- Compression events are aggregated on the main rank for unified metadata updates

### 2) Decoupling logical length from physical cache length

Original nano-vllm tends to couple logical sequence length with physical cache length.  
After compression, this project separates them:

- `seq.rope_pos` and `seq.generated_completion_tokens`: logical generation length (for RoPE / stopping criteria)
- `seq.num_tokens`: physical valid cache length (for context/cache access)

### 3) Prefix-sharing hash behavior

Original `nano-vllm` hashes full blocks for prefix reuse.  
In compression scenarios, this project temporarily deprioritizes hash reuse to guarantee correctness first.

Specifically:

- If two prefix-sharing sequences are in the **same batch** (close arrival time), they may still share physical KV memory after compression
- If they are in **different batches** (larger arrival gap), once the earlier sequence is compressed and its KV changes, later sequences cannot reuse the old shared-prefix physical memory

### 4) Temporary `q_cache` implementation

Some KV-compression methods require query history, so query cache is needed.

For implementation simplicity, this project stores query cache with the same attention head number and mapping scheme as KV cache (shared `slot_mapping` and `block_tables`).  
Therefore, **<u>total query blocks must not exceed the number of KV-occupied blocks for current batch sequences, otherwise OOB errors may occur!!!</u>** This is a temporary solution and will be further optimized.
