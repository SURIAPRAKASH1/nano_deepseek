import torch

from models.config.default import DeepSeekConfig


def precompute_freq_cis(args: DeepSeekConfig)-> torch.Tensor:

    """

    Precomputing frequency-based complex exponential for positional embedding.
    when viewing this complex values as real numbers we can get sin/cos

    Args:
        args (DeepSeekConfig): configuration arguments for positional embedding

    Returns:
        torch.Tensor: Precomputed complex exponential values for PE. dtype-> complex64

    """
    dim =  args.qk_rope_head_dim
    omega = args.omega
    seq_len = args.max_seq_len

    freqs = 1.0 / (omega ** (torch.arange(0, dim, 2).float()[: dim // 2] / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)        # shape (seq_len, dim // 2) -> complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:

    """

    For given input tensor applys rotary positional embeddings. rope inserts positional information about each token to itself

    Args:
        x (torch.Tensor): Input tensor that rope apply to. Shape of (B , seq_len, : , rope_dim) . (:) => n_heads when q, 1 when k
        freqs (torch.Tensor):  Precomputed complex exponential values. Shape of (seq_len, half) . half => rope_dim // 2

    Returns:

        torch.Tensor: Output tensor that rope applied. Shape of (B, seq_len, : , rope_dim)

    """
    dtype = x.dtype

    # when we view tensor as complex num with shape (B, seq_len, : , half , 2)
    # last dim get trun to complex number's real+imagine parts so shape is (B, seq_len, : , half)

    x =  torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))    # (B, seq_len, : , half)
    freqs = freqs.view(1, x.size(1), 1, x.size(-1))                     # (1, seq_len, 1, half)
    y = torch.view_as_real(x * freqs).flatten(3)    # (B, seq_len, :, half) * (1, seq_len, 1, half) -> (B, seq_len, :, rope_dim)

    return y.to(dtype)
