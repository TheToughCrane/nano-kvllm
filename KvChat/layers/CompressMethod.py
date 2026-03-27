import torch
import math

def SnapKV(Q, K, V, num_keep=220, window=5):
    """
    Q: [B, Hq, window, D]
    K: [B, Hk, L, D]
    V: [B, Hk, L, D]   # kept for interface consistency, currently unused
    Returns:
        final_idx:[B,1+num_keep+window]
        or False if compression should be skipped
    """
    B, Hk, L, D = K.shape
    Hq = Q.size(1)
    device = Q.device

    if L - window <= 0 or L <= num_keep + window:
        return False
    K_cut = K[:, :, :-window, :]      
    scale = 1.0 / (D ** 0.5)
    if Hq == Hk:
        # Standard attention
        attn_scores = torch.matmul(Q, K_cut.transpose(-1, -2)) * scale
    else:
        # GQA case: 
        assert Hq % Hk == 0, f"GQA requires Hq % Hk == 0, but got Hq={Hq}, Hk={Hk}"
        group_size = Hq // Hk
        Q_grouped = Q.view(B, Hk, group_size, window, D)
        # [B, Hk, 1, D, L-window]
        K_t = K_cut.transpose(-1, -2).unsqueeze(2)
        # [B, Hk, group_size, window, D] @ [B, Hk, 1, D, L-window]
        # -> [B, Hk, group_size, window, L-window]
        attn_scores = torch.matmul(Q_grouped, K_t) * scale
        attn_scores = attn_scores.view(B, Hq, window, L - window)
    # Avoid attention sink
    attn_scores[:, :, :, 0] = -float('inf')
    attn_probs = torch.softmax(attn_scores, dim=-1)
    # Sum over the query-window dimension
    key_importance = attn_probs.sum(dim=2)   # [B, Hq, L-window]
    # Aggregate across heads to get one token-importance score per sequence
    if key_importance.size(1) > 1:
        key_importance = key_importance.sum(dim=1, keepdim=True)   # [B, 1, L-window]

    _, idx_keep = torch.topk(
        key_importance,
        k=num_keep,
        dim=-1,
        largest=True,
        sorted=False
    )
    idx_keep = torch.sort(idx_keep, dim=-1).values   # [B, 1, num_keep]

    base_idx = idx_keep.view(B, -1)  # [B, num_keep]
    tail_idx = torch.arange(L - window, L, device=device).unsqueeze(0).expand(B, -1)   # [B, window]
    bos_idx = torch.zeros((B, 1), dtype=torch.long, device=device)                      # [B, 1]

    final_idx = torch.cat([bos_idx, base_idx, tail_idx], dim=-1)   # [B, 1 + num_keep + window]
    return final_idx



def StreamingLLM(Q, K, V, num_keep=220, window=5, **kwargs):
    """
    Unified StreamingLLM implementation.
    Philosophy: [BOS (1)] + [Attention Sinks (num_keep)] + [Local Window (window)]
    
    Args:
        Q, K, V: Attention tensors. K shape is [B, Hk, L, D]
        num_keep: Number of initial tokens to keep as attention sinks.
        window: Number of recent tokens to keep for local context fluency.
        
    Returns:
        final_idx: [B, 1 + num_keep + window] containing indices to retain.
    """
    B, Hk, L, D = K.shape
    device = Q.device
    
    # Skip compression if the sequence is shorter than the target compressed length
    if L <= num_keep + window + 1:
        return False
        
    # 1. Absolute BOS Token (Index 0, always kept to stabilize attention denominator)
    bos_idx = torch.zeros((B, 1), dtype=torch.long, device=device)
    
    # 2. Attention Sinks (The initial tokens immediately following BOS)
    # This represents the "Middle Strategy" for StreamingLLM
    sink_idx = torch.arange(1, 1 + num_keep, device=device).unsqueeze(0).expand(B, -1)
    
    # 3. Local Window (The most recent tokens to maintain syntactic fluency)
    tail_idx = torch.arange(L - window, L, device=device).unsqueeze(0).expand(B, -1)
    
    # Concatenate to form the final indices
    # Shape guarantee: 1 + num_keep + window
    final_idx = torch.cat([bos_idx, sink_idx, tail_idx], dim=-1)
    
    return final_idx


def StrideKV(Q, K, V, num_keep=220, window=5, **kwargs):
    """
    Unified StrideKV implementation.
    Philosophy: [BOS (1)] + [Strided Sampling (num_keep)] + [Local Window (window)]
    
    Args:
        Q, K, V: Attention tensors. K shape is [B, Hk, L, D]
        num_keep: Number of tokens to uniformly sample from the middle context.
        window: Number of recent tokens to keep for local context fluency.
        
    Returns:
        final_idx: [B, 1 + num_keep + window] containing indices to retain.
    """
    B, Hk, L, D = K.shape
    device = Q.device
    
    # Skip compression if the sequence is shorter than the target compressed length
    if L <= num_keep + window + 1:
        return False
        
    # 1. Absolute BOS Token (Index 0)
    bos_idx = torch.zeros((B, 1), dtype=torch.long, device=device)
    
    # 2. Local Window (The most recent tokens)
    tail_idx = torch.arange(L - window, L, device=device).unsqueeze(0).expand(B, -1)
    
    # 3. Strided Sampling (The "Middle Strategy" for StrideKV)
    # Uniformly sample 'num_keep' indices from the middle context: [1, L - window - 1]
    # Using linspace guarantees the output tensor size is exactly 'num_keep'
    base_idx = torch.linspace(1, L - window - 1, steps=num_keep, device=device).long().unsqueeze(0).expand(B, -1)
    
    # Concatenate to form the final indices
    # Shape guarantee: 1 + num_keep + window
    final_idx = torch.cat([bos_idx, base_idx, tail_idx], dim=-1)
    
    return final_idx