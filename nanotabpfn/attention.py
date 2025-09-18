# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MultiheadAttention(nn.Module):
    """
    Minimal Multi-Head Attention using PyTorch's scaled_dot_product_attention (SDPA).

    This implementation benefits from PyTorch's automatic dispatch:
    - On CUDA with supported dtypes (fp16, bf16, fp32) and head_dim <= 128,
      it uses **Flash Attention** kernels for maximum efficiency.
    - Otherwise, it falls back to the memory-efficient or math kernel.

    Tensor shape notation:
        B = Batch size
        T = Sequence length
        E = Embedding dimension
        H = Number of attention heads
        D = Per-head dimension (D = E / H)

    Parameters
    ----------
    embed_dim : int
        Input/output embedding size (E).
    num_heads : int
        Number of attention heads (H). Must divide embed_dim.
    batch_first : bool, default True
        If True, input/output is (B, T, E). If False, (T, B, E).
    qkv_bias : bool, default False
        Include bias terms in the q/k/v/out projections.
    out_proj_bias : bool, default False
        Include bias term in the output projection.
    device, dtype : Optional
        Device and dtype.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        batch_first: bool = True,
        qkv_bias: bool = False,
        out_proj_bias: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        fw = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, **fw)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, **fw)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, **fw)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **fw)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute multi-head attention.

        Uses PyTorch's scaled_dot_product_attention (SDPA), which
        automatically dispatches to the **Flash Attention kernel** when available.

        Args
        ----
        query : Tensor
            (B, Tq, E) if batch_first else (Tq, B, E)
        key : Tensor
            (B, Tk, E) if batch_first else (Tk, B, E)
        value : Tensor
            (B, Tk, E) if batch_first else (Tk, B, E)

        Returns
        -------
        attn_output : Tensor
            Same layout as input (batch_first preserved).
        None :
            Placeholder for attention weights (not computed).
        """
        if not self.batch_first:
            # convert (T, B, E) -> (B, T, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Allow for different sequence lengths in query and key/value
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # Linear projections
        q = self.q_proj(query)  # (B, Tq, E)
        k = self.k_proj(key)    # (B, Tk, E)
        v = self.v_proj(value)  # (B, Tk, E)

        # (B, T, E) -> (B, H, T, D), where D = E / H
        H, D = self.num_heads, self.head_dim
        q = q.view(B, Tq, H, D).transpose(1, 2)  # (B, H, Tq, D)
        k = k.view(B, Tk, H, D).transpose(1, 2)  # (B, H, Tk, D)
        v = v.view(B, Tk, H, D).transpose(1, 2)  # (B, H, Tk, D)

        # SDPA: Flash Attention efficiency when available
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # (B, H, Tq, D)

        # (B, H, Tq, D) -> (B, Tq, E)
        attn = attn.transpose(1, 2).contiguous().view(B, Tq, H * D)
        out = self.out_proj(attn)  # (B, Tq, E)

        if not self.batch_first:
            # convert back (B, T, E) -> (T, B, E)
            out = out.transpose(0, 1)
        # None placeholder for attention weights (not computed)
        return out, None
