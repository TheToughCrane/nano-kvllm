import torch
import math

def SnapKV(Q,K,V,num_keep = 220,window = 5):
    B, H, L, D = Q.shape
    device = Q.device
    if L - window <=0 or L<=num_keep+window:
        return False

    Q_cut = Q[:, :, -window:, :]     
    K_cut = K[:, :, :-window, :]     
    scale = 1.0 / (D ** 0.5)
    attn_scores = torch.matmul(Q_cut, K_cut.transpose(-1, -2)) * scale

    attn_scores[:, :, :, 0] = -float('inf')#avoid attention sink

    attn_probs = torch.softmax(attn_scores, dim=-1)
    
    key_importance = attn_probs.sum(dim=2)
    if key_importance.size(1)>1:#mean head if we have multiple heads in this rank
        key_importance = key_importance.sum(dim = 1,keepdim=True)
    _, idx_keep = torch.topk(key_importance, k=num_keep, dim=-1, largest=True, sorted=False)
    idx_keep = torch.sort(idx_keep, dim=-1).values  
    
    base_idx = idx_keep.view(B,-1)
    
    tail_idx = torch.arange(L - window, L, device=device).unsqueeze(0).expand(B, -1)    
    bos_idx = torch.zeros((B,1),dtype=torch.long, device=device)
    final_idx = torch.cat([bos_idx,base_idx,tail_idx], dim=-1)
    return final_idx