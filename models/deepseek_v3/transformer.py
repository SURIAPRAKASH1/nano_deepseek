import torch
import torch.nn.functional as F 
import torch.nn as nn

from models.config.default import DeepSeekConfig
from transformer_layers import DeepSeekBlock, DeepSeekRMSNorm
from rope import precompute_freq_cis

from typing import Optional



class DeepSeekTransformer(nn.Module):

    """
    Transformer model with Rotary Positional Embedding, Multiple Layers and Output projection

    Attributes:
        embed (nn.Embedding): Embedding layer for transformer
        layers (nn.ModuleList): List of transformer blocks
        norm (nn.Module): Normalization Layer
        head (nn.Linear): Final ouput projection layer
        freqs (torch.Tensor): Precomputed complex exponential values for RoPE

    """

    def __init__(self, args: DeepSeekConfig):
        super().__init__()

        self.vocab_size = args.vocab_size
        self.block_size = args.block_size
        self.n_dense_layers = args.n_dense_layers

        self.embed = nn.Embedding(args.vocab_size, args.n_embd)
        self.layers = nn.ModuleList([
            DeepSeekBlock(i, args) for i in range(args.n_layers)
        ])
        self.norm = DeepSeekRMSNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, args.bias)

        # weight tying
        self.embed.weight = self.head.weight

        # weight initialization
        self.apply(self._init_weight)

        self.register_buffer('freqs', precompute_freq_cis(args), persistent= False)

        # report total parameters in model
        print("total parameters %.2f" % (self._get_total_parameters()/ 1e+6),"M")
        print("active parameters %.2f" % (self._get_active_parameters(args)/1e+6),"M")

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.006)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.006)

    def device(self):
        return next(self.parameters()).device

    def _get_total_parameters(self) -> int:
        """
            Returns total parameters in model
        """
        t = 0
        for p in self.parameters():
            t += p.nelement()
        return t

    def _get_active_parameters(self, args: DeepSeekConfig) -> int:
        """
        Returns active parameters in model
        """
        not_active_experts = args.num_experts - (args.n_activated_experts + args.n_shared_experts)
        not_active_experts_pcount = args.n_embd * args.experts_dim * not_active_experts

        t = 0
        for np, p in self.named_parameters():
            if np.endswith('ffn.w1') or np.endswith('ffn.w2') or np.endswith('ffn.w3'):
                t += p.nelement() - not_active_experts_pcount
            else:
                t += p.nelement()
        return t

    def forward(self, input_ids:torch.Tensor, target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transformer forward pass

        Args:
            input_ids (torch.Tensor): Input tensor of token ID's with shape (batch, seq_len)
            target_ids (Optional[torch.Tensor]): Ground truth ID's of token with shape (batch, seq_len). If provided loss will be computed

        Returns:
            torch.Tensor: Logits with shape of (batch, vocab_size)
            loss (torch.Tensor): Loss will be computed using cross_entropy . if only target is given

        """
        seq_len = input_ids.size(1)

        h = self.embed(input_ids)            # token embedding
        freqs = self.freqs[:seq_len]         # crop freqs
        mask = None

        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float('-inf'), device = input_ids.device).triu_(1)

        for layer in self.layers:
            h = layer(h, freqs, mask)

        # norm before logits
        h = self.norm(h)

        if target_ids is not None:
            logits = self.head(h)
            # main loss
            celoss = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))
            # complementry seq_wize auxiliary loss
            total_auxi_loss = 0.0
            if self.training:
                for i in range(self.n_dense_layers, len(self.layers)):
                    total_auxi_loss +=  self.layers[i].ffn.gate.auxi_loss

            loss = celoss + total_auxi_loss

        else:
            # finall logits modification
            logits = self.head(h[:, [-1], :])
            loss = None

        return logits, loss

    def generate(self, idx, max_tokens, temperature = 0.8, top_k: Optional[int] = None)-> torch.Tensor:

        """
        From given idx model predicting next tokens in cascual way until max_tokens limit

        Attributes:

            idx (torch.Tensor): tensor with shape (batch_size, seq_len) eg: [[0, 0]]
            max_tokens (int): Maximum tokens to predict
            temprature (float): Controlling randomness of next token prediction.
            topk (int): Only get probabilities for next token from only topk logits

        Returns:
            idx (torch.Tensor): after predicted max tokens. tensor with shape (batch_size, seq_len + max_tokens)

        """

        # idx (b, t) by takes previous sequence we try to complete the sequence 
        # so every iteration we increase t size

        with torch.no_grad():

            for _ in range(max_tokens):

                # we croping the block size .we can't take infinite pre-context to predict next token.model context length is limited

                idx_count = idx if idx.size(1) <= self.block_size else idx[:,- self.block_size:]

                # get the logits from model
                logits, _  = self(idx_count)
                # then scale the logits by temperature. by doing this way we can control how next token going to draw
                logits = logits[:, -1, :] / temperature
                # then apply softmax to get prob distripution for our vocab
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.softmax(logits, dim = -1)
                # we drawing next token in random sampling way so token with lowest will get a chance
                next_idx = torch.multinomial(probs, num_samples=1)
                # then add the next token to our token seq so next time model can predict token based on this token
                idx = torch.cat((idx, next_idx), dim=1)

        return idx