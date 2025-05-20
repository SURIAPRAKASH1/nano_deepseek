import torch 
import torch.nn.functional as F
import torch.nn as nn 

from models.config.default import DeepSeekConfig
from rope import apply_rope
from loss import complementary_seqwise_auxi_loss

from typing import Optional


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):

    """
    Applies Linear Transformation to incomming input y = xA^T + b

    Args:

        x (torch.Tensor): The input tensor
        weight (torch.Tensor): now it just normal weight that transforms x .
                                but future if we use pretrained weights then we have to take care dequantization weights
        bias (torch.Tensor): Default is None

    Returns:

        torch.Tensor: transformed input after linear projection

    """

    if weight.element_size() > 1:  # if dtype byte size not 1 (int8) then it's usually bfloat16/float16/flat32
        return F.linear(x, weight, bias)
    # other cases not implemented right now


class Linear(nn.Module):
  
    """

    Custom Linear Layer

    Args:
        in_features (int): Number of inpurt features
        out_features (int): Number of output features
        bias (bool) : Wheather to add bias term or not. Default to False
        dtype (optional): data type for linear layer . Default to float32 if we use pretrained weights then we have change to match

    """
    dtype = torch.get_default_dtype()

    def __init__(self, in_features: int, out_features: int, dtype = None , bias: bool = False ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype = dtype or Linear.dtype))

        # if we use pretrained weights
        if self.weight.element_size() == 1:
            pass
        else:
            nn.init.normal_(self.weight, mean = 0.0, std = 0.2)
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class DeepSeekRMSNorm(nn.Module):

    """
    Root Mean Square Normalization. y = x/(sqrt(mean(x^2)))

    Args:

        dim (int): Input tensor dim that will be normalized along
        eps (float): epsilon value for numerical stability to avoid diving x by 0 . default 1e-6

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.dim = dim
        self.eps = eps
        # learnable weight
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)



class DeepSeekMLA(nn.Module):

    """

    MultiHead Latent Attention (MLA) Layer.

    Attributes:

        dim (int): Dimentionality of Input tensor. Usually same as Embedding dim
        n_heads (int): Number of Heads in MLA
        qk_rope_head_dim (int): RoPE's Head Dimentionality for query, key
        qk_nope_head_dim (int): Dimentionality of Head for query, key
        q_lora_dim (int): Lower Rank projection Dimentionality for query
        kv_lora_dim (int): Low Ranl projection Dimentionality for key, value
        v_head_dim (int): Head dimention for value
        softscale (int): Scaling factor for query, key scores


    """

    def __init__(self, args: DeepSeekConfig):
        super().__init__()

        self.dim = args.n_embd
        self.n_heads = args.n_heads
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_lora_dim = args.q_lora_dim
        self.kv_lora_dim  = args.kv_lora_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.softscale = self.qk_head_dim * -0.5

        # don't down project q if we wanna
        if self.q_lora_dim == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, args.bias)

        # down and up projection matrix for q and q_rope
        else:
            self.wq_a = Linear(self.dim, self.q_lora_dim)
            self.q_norm = DeepSeekRMSNorm(self.q_lora_dim)
            self.wq_b = nn.Linear(self.q_lora_dim, self.n_heads * self.qk_head_dim, args.bias)

        # down and up projection matrix for kv and k_rope
        self.wkv_a = Linear(self.dim, self.kv_lora_dim + self.qk_rope_head_dim)
        self.kv_norm = DeepSeekRMSNorm(self.kv_lora_dim)
        self.wkv_b = nn.Linear(self.kv_lora_dim, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), args.bias)

        # final projection
        self.o = nn.Linear(self.n_heads * self.v_head_dim, self.dim, args.bias)

        # regularization
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor]= None) -> torch.Tensor:

        B, T, C = x.shape                               # (batch_size, seq_len, embddding_dim)
        assert self.dim == C

        # down and up project q
        if self.q_lora_dim == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))   # (B, T, n_heads* hd). hd -> (qk_nope_head_dim + qk_rope_head_dim)

        # split the q for applying rope on q's positional embedding projection
        q = q.view(B, T, self.n_heads, self.qk_head_dim)                                         # (B, T, nh, hd)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim = -1)   # (B, T, nh, qk_nope_hd), (B, T, nh, qk_rope_hd)
        q_pe = apply_rope(q_pe, freqs)

        # down project kv
        kv = self.wkv_a(x)                                  # (B, T, nh, kv_ld + qk_rope_hd)

        # split the kv for rope. unlike q rope, same k rope will concate with k accross heads, that's why we don't have nh dim here
        kv, k_pe = torch.split(kv, [self.kv_lora_dim, self.qk_rope_head_dim], dim = -1)    # (B, T, kv_ld), (B, T, qk_rope_hd)
        k_pe = apply_rope(k_pe.unsqueeze(2), freqs)

        # up project kv
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim = -1) # (B, T, nh, qk_nope_hd), (B, T, nh, v_hd)

        # concate to get finall q, k
        q = torch.cat([q_nope, q_pe], dim = -1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim = -1)

        # Attention. Now let tokens communicate each others
        # if you got confuse of einsum in nano_gpt repo this same attention implemented in matrix multiplication way

        attn_scores = torch.einsum("bshd, bthd->bsht", q, k) * self.softscale     # (B, T, hd, T)
        if mask is not None:
            attn_scores += mask.unsqueeze(1)                                        # (B, T, hd, T) + (T, 1, T)
        attn_weight = torch.softmax(attn_scores, dim = -1, dtype = torch.float32).type_as(x)
        attn_weight = self.attn_dropout(attn_weight)
        y = torch.einsum("bsht, bthd->bshd", attn_weight, v)                       # (B, T, hd, v_hd)

        # cancate all heads and do final projection
        out = self.resid_dropout(self.o(y.flatten(2)))

        return out


class DeepSeekMLP(nn.Module):

    """
    MultiLayer Perceptron (MLP) . If we wanna use Dense layer instead of MOE

    Attributes:
        w1 (Linear) : Linear Layer for input to up projection
        w2 (Linear) : Linear Layer for up projection to down projection
        w2 (Linear) : For feature transformation

    """

    def __init__(self, dim:int, inter_dim: int, args:DeepSeekConfig):
        super().__init__()

        """

        Args:
            dim (int): Input and Output dimentionaly
            inter_dim (int): Dimension of hidden layer
            args (DeepSeekConfig): class contains all Configuration for model

        """

        self.w1 = nn.Linear(dim, inter_dim, args.bias)
        self.w2 = nn.Linear(inter_dim, dim, args.bias)
        self.w3 = nn.Linear(dim, inter_dim, args.bias)
        self.mlp_dropout = nn.Dropout(args.dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.mlp_dropout(out)



class DeepSeekGate(nn.Module):
  
    """
    
    Gate Layer - Routes the tokens to topk Experts in MOE

    Attributes:

        dim (int): Dimenstinality of the Input
        n_shared_experts (int): Number of experts activated always
        n_routing_experts (int): Number of Routing experts (that activated sparsely)
        topk (int): Number of activated experts from routing experts
        score_func ('softmax', 'sigmoid'): Score function for Routing Gate to route Input
        route_scale (float) : scaling factor for routing weights
        g_w (nn.Linear): Linear Layer for getting Gate scores
        bias (nn.Parameters): bias for controlling load balance

    """

    def __init__(self, args: DeepSeekConfig):
        super().__init__()

        """
        Args:
            args (DeepSeekConfig): DeepSeek model Configuration class
        """

        self.dim = args.n_embd
        self.n_shared_experts = args.n_shared_experts
        self.n_routing_experts = args.n_routing_experts
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale

        self.g_w = nn.Linear(self.dim, self.n_routing_experts, args.bias)
        # bias term for load balancing experts
        self.bias = nn.Parameter(torch.ones(self.n_routing_experts))

        # stores per batch seq wise auxi loss
        self.auxi_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        B, T, _ = x.shape                 # (B, T, C) ->  (batch_size, seq_len, n_embedding_dim)
        scores = self.g_w(x)              # (B, T, E) ->  E- n_routing_experts

        if self.training:
            # seq_wise loss
            self.auxi_loss = complementary_seqwise_auxi_loss(scores, self.topk)
        else:
            self.auxi_loss = None

        if self.score_func == 'softmax':
            scores = scores.softmax(dim = -1, dtype = torch.float32)
        else:
            scores = scores.sigmoid()

        original_scores = scores
        if self.bias is not None:
            scores  = scores + self.bias

        # take topk experts
        indices = torch.topk(scores, self.topk, dim = -1)[1]  # (B, T, topk)
        weights = original_scores.gather(-1, indices)

        if self.score_func == 'sigmoid':
            # when using sigmoid have to normalize to sum up to 1
            weights /= weights.sum(-1, keepdim = True) 

        weights *= self.route_scale

        # cancating shared experts indices to activated experts indices for Efficiently handling MOE
        shared_experts_indices = torch.arange(0, self.n_shared_experts, device = device).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        expert_indices  = torch.cat([shared_experts_indices, indices + self.n_shared_experts], dim = -1)

        return weights.type_as(x), expert_indices               # (B, T, topk), (B, T, topk + shared_experts)



class DeepSeekMOE(nn.Module):

    """

    DeepSeekMOE have two special implementation:

        1. shared expert isolation.
        2. fine-grained expert segmentation.

    Attributes:
        dim (int): Dimentionaly of Input. Usually same as Embedding dim
        moe_inter_dim (int): Dimentionaly of MOE hidden state
        topk (int): Number of activated experts for Each input
        n_shared_experts (int): Number of Shared Experts
        n_routing_experts (int): Number of Rouitng experts
        num_experts (int): total Number of Experts in MOE
        gate (nn.Module) : Gate Machanisum of Routing Input to Experts
        w1 (nn.Parameter): Input to Hidden projection. All experts packed
        w2 (nn.Parameter): Hidden to Ouput projection. All experts packed
        w3 (nn.Parameter): Feature transformation. All experts packed
        act (nn.SiLU): Activation function for MOE

    """

    def __init__(self, args: DeepSeekConfig):
        super().__init__()

        self.dim = args.n_embd
        self.moe_inter_dim = args.experts_dim
        self.topk = args.n_activated_experts
        self.n_shared_experts  = args.n_shared_experts
        self.n_routing_experts = args.n_routing_experts
        self.num_experts = args.n_shared_experts + args.n_routing_experts

        # gate layer
        self.gate = DeepSeekGate(args)

        # routing and shared experts are packed togather
        self.w1 = nn.Parameter(torch.zeros(self.num_experts, self.dim, self.moe_inter_dim))
        self.w2 = nn.Parameter(torch.zeros(self.num_experts, self.moe_inter_dim, self.dim))
        self.w3 = nn.Parameter(torch.zeros(self.num_experts, self.dim, self.moe_inter_dim))

        nn.init.normal_(self.w1, mean = 0.0, std = 0.02)
        nn.init.normal_(self.w2, mean = 0.0, std = 0.02)
        nn.init.normal_(self.w3, mean = 0.0, std = 0.02)

        # act
        self.act = nn.SiLU()

        # regularization
        self.moe_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        device = x.device
        B, T, C = x.shape                             # (B, T, C) -> (batch_size, block_size(seq_len), embedding_dim)
        K = self.topk + self.n_shared_experts         #  all activated experts (shared and topk)

        # Gate layer only selects topk experts to be activated but shared experts always activated
        routing_weights, expert_indices = self.gate(x)   # (B, T, topk), (B, T, K)

        # flating x for efficient matrix multipilcation with experts
        x_flat = x.view(-1, C)  # (B*T, C)

        # precomputed indices for placing finall moe result into it's correct place
        batch_indices = torch.arange(B , device = device).unsqueeze(-1).unsqueeze(-1).expand(B, T, K)   #(B, T, K)
        block_indices = torch.arange(T, device = device).unsqueeze(0).unsqueeze(-1).expand(B, T, K)     # (B, T, K)

        # flat out everything
        flat_expert_indices = expert_indices.reshape(-1)   # (B*T*K)
        flat_routing_weights = routing_weights.reshape(-1)   # (B*T*topk)

        flat_batch_indices = batch_indices.reshape(-1)   # (B*T*K)
        flat_block_indices = block_indices.reshape(-1)   # (B*T*K)

        # create copy of same token K time (K total active experts in MOE)
        x_flat = x_flat.unsqueeze(1).expand(-1, K, -1).reshape(-1, C) # (B*T*K, C)

        # plug out all experts (routing and shared) that need for moe_layer 1
        w1 = torch.einsum("bd,bde->be",
                            x_flat,                        # (B*T*K, C)
                            self.w1[flat_expert_indices]   # (B*T*K, C, E)
                            )                              # result mm way -> (B*T*K, C) @ (B*T*K, C, E)  --> (B*T*K, E) (E- expert dim)

        # feature sharing
        w3 = torch.einsum("bd,bde->be",
                            x_flat,                        # (B*T*K, C)
                            self.w3[flat_expert_indices]   # (B*T*K, C, E)
                            )                              # result -> (B*T*K, C) @ (B*T*K, C, E)  --> (B*T*K, E)

        # apply silu activation
        intermediate = self.act(w1) * w3    # (B*T*K, 1, E) * (B*T*K, 1, E)

        # moe layer 2
        experts_outputs = torch.einsum("be,bed->bd",
                                        intermediate,                   # (B*T*K, E)
                                        self.w2[flat_expert_indices]    # (B*T*K, E, C)
                                        )                               # result -> (B*T*K, E) @ (B*T*K, E, C)  --> (B*T*K, C)

        # apply dropout
        experts_outputs = self.moe_dropout(experts_outputs)

        # weight the experts (topk only not shared onces) by it crossponding routing weights
        is_routing_expert = flat_expert_indices >= self.n_shared_experts                 # mask for only selecting topk expert's activation
        weighted_expert_outputs = experts_outputs.clone()
        weighted_expert_outputs[is_routing_expert] *= flat_routing_weights.unsqueeze(-1)  # (B*T*k, C) * (B*T*k, 1)

        final_output = torch.zeros((B*T, C) ,device= device)     # (B*T, C)

        # sparse indices for accumulating tokens's activation by experts
        # eg: tensor([0, 0, .., K, 1,1, .., K, B*T-1, B*T-1, ...K])
        sparse_indices = flat_batch_indices * T + flat_block_indices
        final_output.index_add_(0, sparse_indices, weighted_expert_outputs.to(final_output.dtype))

        return final_output.view(B, T, C)
    

class DeepSeekBlock(nn.Module):

    """
    Transformer Block combining Attention and feedforward layers

    Attributes:
        attn (nn.Module): Attention Layer (MLA)
        ffn (nn.Module): Feed-forward network (MOE, MLP)
        attn_norm (nn.Module): Layer normalization for attention
        ffn_norm (nn.Module): Layer normalization for Fee-Forward network

    """

    def __init__(self, layer_id: int, args: DeepSeekConfig):
        super().__init__()

        """
        Args:

            layer_id (int): Current transformer layer idx
            args (DeepSeekConfig): Configuration for model

        """

        self.attn = DeepSeekMLA(args)
        self.ffn = DeepSeekMLP(args.n_embd, args.inter_dim, args) if layer_id < args.n_dense_layers else DeepSeekMOE(args)
        self.attn_norm = DeepSeekRMSNorm(args.n_embd)
        self.ffn_norm = DeepSeekRMSNorm(args.n_embd)

    def forward(self, x: torch.tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor])-> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input that will feed to transformer layers
            freqs (torch.Tensor): Precomputed complex exponential values for rope
            mask (torch.Tensor): casual mask for preventing tokens to communicate into future

        """
        x  = x + self.attn(self.attn_norm(x), freqs, mask)
        x  = x + self.ffn(self.ffn_norm(x))
        return x

