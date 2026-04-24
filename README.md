# `nano-kvllm v0.2.0`

<p align="center">
  <img src="assets/logo.png" alt="nano-kvllm logo" width="350"/>
</p>

**Here comes version 2.0!**

**KvLLM** is an **AI/LLM inference framework** built on top of `nano-vllm`, focused on efficient KV-cache memory management for large language models.  
The current release lands **KV-cache compression** in a lightweight research-and-development framework, aiming to improve inference efficiency in **high-concurrency** and **long-context generation** scenarios by alleviating the KV-cache memory bottleneck.

In the coming weeks, **KvLLM** will continue integrating frontier KV-cache management techniques, including:

- **KV-cache compression**
- **KV-cache offloading**
- **KV-cache retrieval**

to form a more complete and practical memory-management stack for LLM serving.

---

# What’s New in `nano-kvllm v0.2.0`
## Motivation

Current mainstream KV cache compression (sparsification) methods mostly rely on a **threshold-triggered compression mechanism**: once the KV cache length of a sequence reaches a predefined threshold, compression is triggered.
This design may be friendly for single-user deployment or agent-style scenarios, but it has several critical drawbacks:
1. **Compression may affect system prompt KV cache** 
   Since the compression mechanism operates on whole historical KV cache, it may prune cache entries corresponding to the system prompt, which can negatively affect generation quality.
2. **Compression overhead can hurt throughput under high batch size**  
   In high-concurrency scenarios, nearly every request may trigger a compression event at every decode step, which can actually reduce output throughput instead of improving it.
---
## Method
KvLLM 2.0 introduces a **window-based + periodic compression mechanism**.
Specifically:
- We start counting decode steps globally from the beginning.
- Every fixed number of decode steps (e.g., every 1024 decode steps globally), we trigger compression.
- At each compression step, we select the **Top-K** sequences.
- For each selected sequence, we treat its **last 4 blocks** as a compression window.
- These blocks are guaranteed to be **uncompressed**.
- We use the **latest token's query** to compute attention scores, select and preserve the important KV pairs, and then perform **memory compaction** and **metadata updates**.
---
## Advantages
### Window-based compression
The blocks inside the compression window are guaranteed to be **uncompressed blocks**, which avoids repeatedly compressing already-compressed content and helps preserve generation quality.
### Periodic compression
Periodic compression controls when compression events happen, avoiding too many requests compressing at the same time. This can effectively improve output throughput in high-concurrency scenarios.

## Experiments

We ran experiments on the **Math500** dataset with:

- **Batch size**: 500 (all samples are fed at once)
- **Compression period**: 1024
- **Number of compressed sequences**: 20
- **Compression ratio**: 50%

The output throughput improved by **10%**:

- **2000 tok/s → 2200 tok/s**

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

1. Further improve output throughput in large-batch serving
2. **Implement and rapidly validate KV-cache compression algorithms**
3. Further explore and extend **other KV-cache memory-management techniques**

The current release is **nano-kvllm v0.2.0**.  
For earlier versions, please refer to **v0.1.0** and **v0.1.5**.

---

# Project Structure

- **KvChat**: application/demo layer for single-user multi-turn chat
- **nano-kvllm**: research/development framework for KV-cache memory management on top of `nano-vllm`

---

## Changelog

See [CHANGE_LOG.md](./CHANGE_LOG.md) for version history.
