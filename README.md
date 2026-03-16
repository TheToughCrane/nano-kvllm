# `nano-kvllm v0.1.5`

<p align="center">
  <img src="assets/logo.png" alt="nano-kvllm logo" width="350"/>
</p>

**KvLLM** is an **AI/LLM inference framework** built on top of `nano-vllm`, focused on efficient KV-cache memory management for large language models.  
The current release lands **KV-cache compression** in a lightweight research-and-development framework, aiming to improve inference efficiency in **high-concurrency** and **long-context generation** scenarios by alleviating the KV-cache memory bottleneck.

In the coming weeks, **KvLLM** will continue integrating frontier KV-cache management techniques, including:

- **KV-cache compression**
- **KV-cache offloading**
- **KV-cache retrieval**

to form a more complete and practical memory-management stack for LLM serving.

---

## About KvChat

**KvChat** is an application-level demo built on top of **KvLLM**.  
It provides a lightweight **single-user multi-turn chat** interface and demonstrates how **online KV-cache compression** can be applied during long conversations.

For long multi-turn dialogue, KV-cache grows continuously with generated context, which may lead to:

- increasing GPU memory pressure,
- reduced throughput,
- earlier OOM under long sessions.

**KvChat** addresses this by enabling **runtime KV-cache compression**, and allows users experience a fast and lightweight conversation by:

- turn compression on/off during chat,
- adjust compression-related parameters,

# Quick Start

## 1. Configure `chat_cli.py`

Set the following parameters in `chat_cli.py`:

```python
enforce_eager=False     # Enable graph-friendly decode path when possible; can significantly improve decode speed
tensor_parallel_size=2  # Number of GPUs used for tensor parallel inference
max_tokens = 32000      # Maximum generation length in multi-turn dialogue
```

## 2. Run

```bash
python chat_cli.py
```

You can then start multi-turn conversation directly from the CLI.

KvChat supports **runtime KV-cache compression** to significantly delay KV-cache growth during long conversations.  
You can tune compression-related parameters in `nanokvllm/config.py`:

```python
kv_compress_S: int = 511      # Trigger compression when generated context length reaches S
kv_compress_R: int = 257      # Retain prompt length + R tokens in KV cache after compression
query_window_size: int = 50   # Query-window parameter for compression algorithms; recommended range: 10~100
```

---

# About `nano-kvllm`

`nano-kvllm` is the **development framework** behind **KvLLM**.  
It provides a compact and practical research framework for landing KV-cache compression algorithms on top of `nano-vllm`.

Developers can use it to:

1. **Implement and rapidly validate KV-cache compression algorithms**
2. Further explore and extend **other KV-cache memory-management techniques**

The current release is **nano-kvllm v0.1.5**.  
For earlier versions, please refer to **v0.1.0**.

---

# What’s New in `nano-kvllm v0.1.5`

## 1. `query_window_manager`
In `v0.1.0`, query cache was stored in a more global/block-style manner, which could lead to unnecessary memory overhead.

In `v0.1.5`, we introduce **`query_window_manager.py`**:

- Query cache is only stored when a sequence enters the compression window
- Only a small recent query window is kept
- Query cache is released after compression

This design significantly reduces wasted memory usage.

> Note: Most KV-cache compression algorithms require only a small amount of recent query cache, not the full-sequence query history.

---

## 2. Graph-mode-aware compression
CUDA graph mode in vLLM-style systems can reduce kernel launch overhead and often improve decode throughput substantially.  
However, graph mode also increases memory usage, especially in **high-concurrency** and **long-context** scenarios.

`nano-kvllm v0.1.5` supports a practical compromise:

- **Enable graph mode on non-compression decode steps**
- **Disable graph mode on compression steps**

This makes it possible to preserve most of the graph-mode speedup while still allowing online KV-cache compression, improving overall decode efficiency under memory pressure.

---

## 3. Recompute compatibility after compression
In `v0.1.0`, recomputation after sequence compression could fail because during `prepare_prefill`, the logical sequence state and the input token reconstruction path could become inconsistent.

In `v0.1.5`, this is addressed by:

- resetting `num_tokens` during recomputation-related preemption handling (`scheduler.py::preempt`)
- passing full `token_ids` during sequence serialization

This improves robustness when compressed sequences are later recomputed.

---

## 4. Other improvements
Additional updates in `v0.1.5` include:

- removing the Triton-based KV compact implementation and falling back to a more stable Torch implementation
- improving overall code readability and maintainability

# Project Structure

- **KvChat**: application/demo layer for single-user multi-turn chat
- **nano-kvllm**: research/development framework for KV-cache memory management on top of `nano-vllm`

---

## Changelog

See [CHANGE_LOG.md](./CHANGE_LOG.md) for version history.
