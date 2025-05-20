from dataclasses import dataclass

"default configuration for deepseek-v3. for quick training and see How model converges"
@dataclass
class GPTMOEConfig:

    n_embd: int = 128                # token embedding dim 
    vocab_size: int = 102400         # number of unique tokens
    n_layers: int = 4                 # num of layers
    block_size: int = 16             # seq_len/context lenght what ever you name it
    batch_size: int = 32             # How many sequence are packed togather ?

    # MLA - Multihead Latent Attention
    ld: int = 8                          # fixed compressed dim for q, k, v to down-projection
    n_heads: int = 4                      # number of heads
    head_dim: int = n_embd // n_heads    # dim of each head 

    # DeepSeekMOE
    m: int = 4                                                # fine-grained expert segmentation
    num_experts: int = 16 * m                                 # How many experts in MOE?
    n_shared_experts: int = 2                                 # experts that are activated always
    k: int = (2 * m) - n_shared_experts                       # How many routing active expert ?
    n_routing_expert: int = num_experts - n_shared_experts    # num of experts that are activated sparsely
    expert_dim: int = (4 * n_embd )// m                       # what's dim of expert?

    bias: bool = True
    dropout: float = 0.2