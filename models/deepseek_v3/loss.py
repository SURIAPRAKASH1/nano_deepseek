import torch

def complementary_seqwise_auxi_loss(affinity_scores: torch.Tensor, topk: int, alpha: float = 0.0001)-> torch.Tensor:
    """
    Computes Complementary Sequence-Wise Auxiliary Loss. 

    Args:

        affinity_scores (torch.Tensor): Raw scores from Gate Layer with shape (batch, seq_len, n_routing_experts)
        topk (int): Number of Routing Activated Experts
        alpha (int) : Balance factor. As a default 1e-4

    Returns:
        loss (torch.Tensor): Scalar loss

    """
    B, T, Nr = affinity_scores.shape

    _, top_indices = torch.topk(affinity_scores, topk, dim = -1)                # (B, T, topk)

    mask = torch.zeros_like(affinity_scores).scatter(-1, top_indices, 1.0)   # (B, T, Nr)

    # frequency of How often each expert in topk
    fi = (Nr/(topk * T)) * mask.sum(1)                            # (B, Nr)

    # routing probabilites. normalized accross experts
    norm_scores = torch.softmax(affinity_scores, dim = -1)

    # mean of seq level
    pi = norm_scores.mean(dim = 1)             # (B, Nr)

    # calculating loss
    loss = alpha * (fi * pi).sum(dim = 1).mean()
    return loss
