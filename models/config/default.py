from dataclasses import dataclass
from typing import Literal

" Default configuration for deepseek-v3. For quick training and see How model converges "

@dataclass
class DeepSeekConfig:

    """
    not here -> Deepseek-v3 didn't follow

    n_embd (int): Embedding Dimention for each token.

    vocab_size (int): total Number of unique tokens.
    
    n_layers (int): Number of layers in model.

    block_size (int): Sequence length or Context size.

    batch_size (int): How many sequence as a group ?.

    inter_dim (int): Intermediate Dimentionality for MLP. (Usually 4 * n_embd not here).

    n_dense_layers (int): How many Dense layer wanted to transformer have ?.

    # MLA
    q_lora_dim (int): Dimentionality for Low Rank Projection for query.

    kv_lora_dim (int): Dimentionality for Low Rank Projection for key, value.

    n_heads (int): Number of Heads in attention.

    qk_nope_head_dim (int): Head Dim for query/key without positional embeddings. (Usually n_embd // n_heads not here).

    qk_rope_head_dim (int): Head Dimentionality for q, k with positional embeddings.

    v_head_dim (int): Head Dimentionality for v. (Usually n_embd // n_heads not here).
    
    # DeepSeekMOE
    m (int): Fine-grained expert segmentation

    num_experts (int): Total Number of Experts in MOE

    n_shared_experts (int): Experts that are activated always. isolated shared experts

    n_routing_experts (int): Number of experts that are activated sparsely

    n_activated_experts (int): How many routing experts will be activated per token?

    experts_dim (int): what's hidden state Dimentionality of Experts ? (Usually (4 * n_embd)//m not here)

    score_func (Literal['sigmoid', 'softmax']): Score function to gave affinity scores

    route_scale (float): Routing Scale factor

    # MTP

    mtp_depth (int): How many Multitoken Prediction Module do we want?

    mtp_lambda (float): Scaling factor for mtp loss

    # RoPE
    omega (int): Predefined constant for computing complex exponential values

    # others
    bias (bool): Do our model want to have bias term?

    dropout (float): Regularization for randomly deactivating activation

    max_seq_len (int): Maximum sequence lenght that model can handle

    weight_tying (bool): Input Matrix (Embeddings) and Ouput Matrix (Head) wanna share weights. Reduces parameters counts
    
    """

    n_embd: int = 192                
    vocab_size: int = 128000        
    n_layers: int = 5                 
    block_size: int = 32             
    batch_size: int = 16            
    inter_dim: int = 768              

    # MLA
    q_lora_dim: int = 10                       
    kv_lora_dim: int = 10                      
    n_heads: int = 4                           
    qk_nope_head_dim: int = 48             
    qk_rope_head_dim: int = 24                 
    v_head_dim: int = 48                       
    n_dense_layers: int = 1                   

    # DeepSeekMOE
    m: int = 4                                               
    num_experts: int = 16 * m                               
    n_shared_experts: int = 2                                
    n_routing_experts: int = num_experts - n_shared_experts  
    n_activated_experts: int = (2 * m) - n_shared_experts    
    experts_dim: int = (4 * n_embd ) // m                    
    score_func: Literal['sigmoid', 'softmax'] = 'sigmoid'    
    route_scale: float = 1.0

    # MTP
    mtp_depth: int = 1
    mtp_lambda: float = 3e-1

    # rope
    omega: int = 10000.0

    # others
    bias: bool = False
    dropout: float = 0.1
    max_seq_len: int = block_size
    weight_tying: bool = False