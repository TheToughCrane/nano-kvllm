from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    compress_enabled: bool | None = None
    compress_S: int | None = None
    compress_R: int | None = None
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
