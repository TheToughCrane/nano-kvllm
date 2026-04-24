from copy import copy
from enum import Enum, auto
from itertools import count

from nanokvllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.generated_completion_tokens = 0
        self.rope_pos = self.num_tokens - 1
        self.tail_uncompressed_len = 0
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        
        self.generated_completion_tokens += 1
        self.rope_pos += 1
        self.tail_uncompressed_len += 1
    def __getstate__(self):

        return {
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "block_table": self.block_table,
            "token_ids": self.token_ids,
            "last_token": getattr(self, "last_token", None),
            "generated_completion_tokens": getattr(self, "generated_completion_tokens", 0),
            "rope_pos": getattr(self, "rope_pos", self.num_tokens - 1),
            "seq_id": getattr(self, "seq_id", None),#!!!新增
            "tail_uncompressed_len": getattr(self, "tail_uncompressed_len", 0),
        }

    def __setstate__(self, state):

        self.num_tokens = state.get("num_tokens")
        self.num_prompt_tokens = state.get("num_prompt_tokens")
        self.num_cached_tokens = state.get("num_cached_tokens")
        self.block_table = state.get("block_table", [])

        token_ids = state.get("token_ids", None)
        last_token = state.get("last_token", None)

        if token_ids is not None:
            self.token_ids = token_ids
            if not isinstance(self.token_ids, list):
                self.token_ids = [self.token_ids]
            if self.token_ids:
                self.last_token = self.token_ids[-1]
            else:
                self.last_token = last_token
        else:
            if last_token is None:
                self.last_token = 0
            else:
                self.last_token = last_token
            self.token_ids = [self.last_token]
        self.generated_completion_tokens = state.get("generated_completion_tokens", 0)
        self.rope_pos = state.get("rope_pos", self.num_tokens - 1)
        self.seq_id = state.get("seq_id", None)#!!!新增
        self.tail_uncompressed_len = state.get("tail_uncompressed_len", 0)