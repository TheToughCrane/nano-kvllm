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