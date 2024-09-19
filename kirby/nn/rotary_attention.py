from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None


from kirby.nn.rotary_embedding import apply_rotary_pos_emb


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        batch_type: str = "stacked",
        backend: str = "mem_efficient",
    ):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        if batch_type not in ["stacked", "chained"]:
            raise ValueError(
                f"Unknown batch_type: {batch_type}, must be one of 'stacked', 'chained'"
            )
        self.batch_type = batch_type

        if backend not in ["math", "mem_efficient", "flash"]:
            raise ValueError(
                f"Unknown backend: {backend}, must be one of 'math', "
                "'mem_efficient', 'flash'"
            )

        if backend == "mem_efficient" and xops is None:
            raise ImportError(
                "xformers not installed, please install `xformers` "
                "to use the mem_efficient backend or choose "
                "another backend."
            )

        if backend == "mem_efficient" and batch_type == "chained" and dropout > 0.0:
            raise ValueError(
                "Dropout is not supported with the mem_efficient backend when the input"
                " is chained. This is caused by a current bug in xformers, either set "
                "`dropout` to 0 or choose another backend."
            )

        if backend == "math" and batch_type == "chained":
            raise ValueError(
                f"Chained batching is not supported with the math backend."
            )

        if backend == "flash" and xops is None:
            raise ImportError(
                "xformers not installed, please install `xformers` "
                "to use the flash backend or choose "
                "another backend."
            )
            # raise ImportError(
            #     "flash_attn not installed, please install `flash_attn`"
            #     " to use the flash backend, or choose another backend."
            # )
        self.backend = backend

        # build networks
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x_query,
        x_context,
        query_pos_emb,
        context_pos_emb,
        *,
        context_mask=None,
        query_seqlen=None,
        context_seqlen=None,
    ):
        # normalize and project to q, k, v
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        if self.batch_type == "stacked":
            assert query_seqlen is None and context_seqlen is None

            rotary_attn_func = rotary_attn_backend_map[self.backend]

            out = rotary_attn_func(
                query=q,
                key=k,
                value=v,
                q_pos_emb=query_pos_emb,
                kv_pos_emb=context_pos_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                attn_mask=context_mask,
            )

        elif self.batch_type == "chained":
            assert context_mask is None

            if query_seqlen is None or context_seqlen is None:
                raise ValueError(
                    "Both `query_seqlen` and `context_seqlen` must be "
                    "provided for chained batching."
                )

            rotary_attn_varlen_func = rotary_attn_varlen_backend_map[self.backend]

            out = rotary_attn_varlen_func(
                query=q,
                key=k,
                value=v,
                q_pos_emb=query_pos_emb,
                kv_pos_emb=context_pos_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                q_seqlen=query_seqlen,
                kv_seqlen=context_seqlen,
            )

        # project back to dim
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        batch_type: str = "stacked",
        backend: str = "mem_efficient",
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        if batch_type not in ["stacked", "chained"]:
            raise ValueError(
                f"Unknown batch_type: {batch_type}, must be one of 'stacked', 'chained'"
            )
        self.batch_type = batch_type

        if backend not in ["math", "mem_efficient", "flash"]:
            raise ValueError(
                f"Unknown backend: {backend}, must be one of 'math', "
                "'mem_efficient', 'flash'"
            )

        if backend == "mem_efficient" and xops is None:
            raise ImportError(
                "xformers not installed, please install `xformers` "
                "to use the mem_efficient backend or choose "
                "another backend."
            )

        if backend == "math" and batch_type == "chained":
            raise ValueError(
                f"Chained batching is not supported with the math backend."
            )

        if backend == "flash" and xops is None:
            raise ImportError(
                "xformers not installed, please install `xformers`"
                " to use the flash backend, or choose another backend."
            )
        self.backend = backend

        # build networks
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        rotary_time_emb,
        *,
        x_mask=None,
        x_seqlen=None,
    ):

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if self.batch_type == "stacked":
            rotary_attn_func = rotary_attn_backend_map[self.backend]

            out = rotary_attn_func(
                query=q,
                key=k,
                value=v,
                q_pos_emb=rotary_time_emb,
                kv_pos_emb=rotary_time_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                attn_mask=x_mask,
            )

        elif self.batch_type == "chained":
            assert x_mask is None

            if x_seqlen is None:
                raise ValueError("`x_seqlen` must be provided.")

            rotary_attn_varlen_func = rotary_attn_varlen_backend_map[self.backend]

            out = rotary_attn_varlen_func(
                query=q,
                key=k,
                value=v,
                q_pos_emb=rotary_time_emb,
                kv_pos_emb=rotary_time_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                q_seqlen=x_seqlen,
                kv_seqlen=None,  # self-attention has the same seqlen for q, k, v
            )

        out = self.to_out(out)
        return out


def rotary_attn_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    attn_mask=None,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):
    # uses the default scaled dot product attention from pytorch
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    # this implements basic versions of memory efficient attention and flash attention
    # but more advanced versions are available in xformers and flash_attn (varlen)
    # which allow us to perform complex masking operations
    # TODO add documentation for rotate_value
    r"""Wraps the default attention implementation with rotary embedding application.

    Args:
        query: The query tensor, with shape (b, n_q, (h d))
        key: The key tensor, with shape (b, n_kv, (h d))
        value: The value tensor, with shape (b, n_kv, (h d))
        q_pos_emb: The query rotary position embedding, with shape (b, n_q, d)
        kv_pos_emb: The key rotary position embedding, with shape (b, n_kv, d)
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        rotate_value: Whether to rotate the value in addition to the query and key
        attn_mask: The attention mask, with shape (b, n_kv)

    Returns:
        The output tensor, with shape (b, n_q, (h d))
    """

    # default attention expects shape b h n d
    query = rearrange(query, "b n (h d) -> b h n d", h=num_heads)
    key = rearrange(key, "b n (h d) -> b h n d", h=num_heads)
    value = rearrange(value, "b n (h d) -> b h n d", h=num_heads)

    # apply rotary embeddings
    query = apply_rotary_pos_emb(q_pos_emb, query, head_dim=1)
    key = apply_rotary_pos_emb(kv_pos_emb, key, head_dim=1)
    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb, value, head_dim=1)

    # attention mask
    if attn_mask is not None:
        attn_mask = rearrange(attn_mask, "b n -> b () () n")

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb, out, head_dim=1)

    # return (b, n, (h d), )
    out = rearrange(out, "b h n d -> b n (h d)")
    return out


def mem_efficient_rotary_attn_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    attn_mask=None,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):
    r"""Wraps the memory efficient attention implementation with rotary embedding
    application.

    Args:
        query: The query tensor, with shape (b n (h d))
        key: The key tensor, with shape (b n (h d))
        value: The value tensor, with shape (b n (h d))
        q_pos_emb: The query rotary position embedding, with shape (b n d)
        kv_pos_emb: The key rotary position embedding, with shape (b n d)
        attn_mask: The attention mask, with shape (b, n_kv)
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        rotate_value: Whether to rotate the value in addition to the query and key

    Returns:
        The output tensor, with shape (b n (h d))
    """
    # xformers attention expects shape (1, n, h, d)
    query = rearrange(query, "b n (h d) -> b n h d", h=num_heads)
    key = rearrange(key, "b n (h d) -> b n h d", h=num_heads)
    value = rearrange(value, "b n (h d) -> b n h d", h=num_heads)

    query = apply_rotary_pos_emb(q_pos_emb, query, head_dim=2)
    key = apply_rotary_pos_emb(kv_pos_emb, key, head_dim=2)

    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb, value, head_dim=2)

    # WARNING: this is very slow, avoid using attn_mask if possible, refer to xformers
    # documentation
    attn_mask = (
        repeat(attn_mask, "b m -> b h n m", h=num_heads, n=query.size(1))
        if attn_mask is not None
        else None
    )
    attn_bias = (
        attn_mask.float().masked_fill(attn_mask, float("-inf"))
        if attn_mask is not None
        else None
    )

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb, out, head_dim=2)

    out = rearrange(out, "b n h d -> b n (h d)")
    return out


def mem_efficient_rotary_attn_varlen_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    q_seqlen,
    kv_seqlen,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):
    r"""Wraps the memory efficient attention implementation with rotary embedding
    application.

    Args:
        query: The query tensor, with shape (n, (h d))
        key: The key tensor, with shape (n, (h d))
        value: The value tensor, with shape (n, (h d))
        query_pos_emb: The query rotary position embedding, with shape (n, d)
        key_pos_emb: The key rotary position embedding, with shape (n, d)
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        rotate_value: Whether to rotate the value in addition to the query and key
        q_seqlen: The sequence length of the query tensor
        kv_seqlen: The sequence length of the key and value tensors

    Returns:
        The output tensor, with shape (n, (h d))
    """
    # xformers attention expects shape (1, n, h, d)
    query = rearrange(query, "n (h d) -> () n h d", h=num_heads)
    key = rearrange(key, "n (h d) -> () n h d", h=num_heads)
    value = rearrange(value, "n (h d) -> () n h d", h=num_heads)

    # TODO check rotation works
    query = apply_rotary_pos_emb(q_pos_emb.unsqueeze(0), query)
    key = apply_rotary_pos_emb(kv_pos_emb.unsqueeze(0), key)

    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb.unsqueeze(0), value)

    if isinstance(q_seqlen, torch.Tensor):
        q_seqlen = q_seqlen.tolist()
    if isinstance(kv_seqlen, torch.Tensor):
        kv_seqlen = kv_seqlen.tolist()

    # fill attention_bias with BlockDiagonalMask
    with torch.no_grad():
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb.unsqueeze(0), out)

    out = rearrange(out, "() n h d -> n (h d)")
    return out


def flash_rotary_attn_varlen_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    q_seqlen,
    kv_seqlen,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):
    r"""Wraps the flash attention implementation (from xformers) with rotary embedding
    application.

    Args:
        query: The query tensor, with shape (n, (h d))
        key: The key tensor, with shape (n, (h d))
        value: The value tensor, with shape (n, (h d))
        query_pos_emb: The query rotary position embedding, with shape (n, d)
        key_pos_emb: The key rotary position embedding, with shape (n, d)
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        rotate_value: Whether to rotate the value in addition to the query and key
        q_seqlen: The sequence length of the query tensor
        kv_seqlen: The sequence length of the key and value tensors

    Returns:
        The output tensor, with shape (n, (h d))
    """
    # xformers attention expects shape (1, n, h, d)
    query = rearrange(query, "n (h d) -> () n h d", h=num_heads)
    key = rearrange(key, "n (h d) -> () n h d", h=num_heads)
    value = rearrange(value, "n (h d) -> () n h d", h=num_heads)

    # TODO check rotation works
    query = apply_rotary_pos_emb(q_pos_emb.unsqueeze(0), query)
    key = apply_rotary_pos_emb(kv_pos_emb.unsqueeze(0), key)

    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb.unsqueeze(0), value)

    if isinstance(q_seqlen, torch.Tensor):
        q_seqlen = q_seqlen.tolist()
    if isinstance(kv_seqlen, torch.Tensor):
        kv_seqlen = kv_seqlen.tolist()

    # fill attention_bias with BlockDiagonalMask
    with torch.no_grad():
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout_p,
        op=xops.MemoryEfficientAttentionFlashAttentionOp,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb.unsqueeze(0), out)

    out = rearrange(out, "() n h d -> n (h d)")
    return out


rotary_attn_backend_map = {
    "math": rotary_attn_func,
    "mem_efficient": mem_efficient_rotary_attn_func,
    "flash": None,  # not implemented
}

rotary_attn_varlen_backend_map = {
    "mem_efficient": mem_efficient_rotary_attn_varlen_func,
    "flash": flash_rotary_attn_varlen_func,
}
