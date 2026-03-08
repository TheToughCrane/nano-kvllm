import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    #New properties for KV compression
    kv_compress_enabled: bool = True       #Enable KV-cache compression during decode (prefill is not compressed).
    kv_compress_N: int = 2                 # This is a compress mechanism tailored for paged attention, which can avoid frequent block allocate&deallocate
    kv_compress_S: int = kvcache_block_size * (kv_compress_N + 1) -1     #Compression trigger threshold: sequence KV-context length reaches S and not reach a block's end
    kv_compress_R: int = kvcache_block_size * kv_compress_N + 1       # Retained tokens
    num_query_blocks:int = 200 #!!!Important,it's a budget to allocate block for query cache, we store query like KV cache,
    ## For convenience, I designed the query-cache storage to be performed synchronously with the KV-cache storage,
    # and they share the same slot_mapping; this means num_query_blocks must be greater than
    # (the number of sequences in a batch) × (the maximum number of blocks allocated per sequence).

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
