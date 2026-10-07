import math
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Any


def pure_jax_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    causal: bool = False,
    sm_scale: float | None = None,
    softcap: float | None = None,
    use_alibi: bool = False,
    **kwargs,
) -> jax.Array:
    """
    Numerically faithful pure JAX self-attention replacing Triton FlashAttention
    (fa_kernel_softcap_then_bias.py) with exact float32 accumulation.
    """
    B, S_q, H, D = q.shape
    S_k = k.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Accumulate in float32 for exact parity with Triton kernel's accum_dtype=float32
    orig_dtype = q.dtype
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    # [B, S, H, D] -> [B, H, S, D]
    q_t = jnp.swapaxes(q_f32, 1, 2)
    k_t = jnp.swapaxes(k_f32, 1, 2)
    v_t = jnp.swapaxes(v_f32, 1, 2)

    # Scaled dot-product: [B, H, S_q, S_k]
    qk = jnp.matmul(q_t, jnp.swapaxes(k_t, -1, -2))

    # Match fa_kernel_softcap_then_bias exactly:
    # if softcap > 0.:
    #     qk = softcap * tanh(qk * (softmax_scale * 1. / softcap)) + bias
    # else:
    #     qk = qk * softmax_scale + bias
    if softcap is not None and softcap > 0.0:
        scores = softcap * jnp.tanh(qk * (sm_scale / softcap))
    else:
        scores = qk * sm_scale

    # Add bias (e.g. adjacency matrix for graphs)
    if bias is not None:
        bias_f32 = bias.astype(jnp.float32)
        if bias_f32.ndim == 3:
            bias_f32 = jnp.expand_dims(bias_f32, axis=1)  # [B, 1, S_q, S_k]
        if use_alibi:
            alibi_scale = 2.0 ** (-8.0 / H)
            alibi_slopes = alibi_scale ** jnp.arange(1, H + 1, dtype=jnp.float32)
            bias_f32 = bias_f32 * alibi_slopes[None, :, None, None]
        scores = scores + bias_f32

    # Masking: segment_ids and causal
    mask = None
    if segment_ids is not None and segment_ids.size > 0:
        seg_mask = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
        mask = seg_mask if mask is None else (mask & seg_mask)

    if causal:
        causal_mask = jnp.tril(jnp.ones((S_q, S_k), dtype=jnp.bool_))[None, None, :, :]
        mask = causal_mask if mask is None else (mask & causal_mask)

    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn_weights, v_t)  # [B, H, S_q, D]
    out = jnp.swapaxes(out, 1, 2)        # [B, S_q, H, D]
    return out.astype(orig_dtype)


def attention_fn(
    q: jax.Array | None = None,
    k: jax.Array | None = None,
    v: jax.Array | None = None,
    qkv: jax.Array | None = None,
    bias: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    causal: bool = False,
    sm_scale: float | None = None,
    query_seq_lengths: jax.typing.ArrayLike | None = None,
    key_value_seq_lengths: jax.typing.ArrayLike | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    softcap: float | None = None,
    use_alibi: bool = False,
    **kwargs,
) -> jax.Array:
    if qkv is not None:
        assert q is None and k is None and v is None
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
    else:
        assert q is not None and k is not None and v is not None

    return pure_jax_attention(
        q=q,
        k=k,
        v=v,
        bias=bias,
        segment_ids=segment_ids,
        causal=causal,
        sm_scale=sm_scale,
        softcap=softcap,
        use_alibi=use_alibi,
        **kwargs,
    )


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dtype: jax.typing.DTypeLike,
        param_dtype: jax.typing.DTypeLike = jnp.float32,
        qkv_packed: bool = True,        # qkv-packed attn has not been impl yet, so now qkv_packed=False will be faster
        attention_fn: Callable[..., jax.Array] = attention_fn,
        normalize_qk: bool = False,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dtype = dtype
        self.qkv_packed = True
        self.normalize_qk = normalize_qk

        qkv_kernel_init = nnx.initializers.xavier_uniform()
        out_kernel_init = nnx.initializers.xavier_uniform()

        self.qkv_proj_params: nnx.Param | list[nnx.Param]
        self.qkv_proj_params = nnx.Param(
            qkv_kernel_init(
                rngs.params(), (embed_dim, 3 * embed_dim), param_dtype,
            )
        )
        self.set_qkv_packed(qkv_packed)

        self.out_proj_params = nnx.Param(
            out_kernel_init(
                rngs.params(), (embed_dim, embed_dim), param_dtype,
            )
        )
        self.attention_fn = attention_fn

        self.norm_fn = nnx.RMSNorm(embed_dim // num_heads, use_scale=False, rngs=rngs)

    def set_qkv_packed(self, qkv_packed: bool):
        if qkv_packed == self.qkv_packed:
            return
        self.qkv_packed = qkv_packed
        if not qkv_packed:
            # split
            self.qkv_proj_params = [
                nnx.Param(
                    self.qkv_proj_params.value[:, i * self.embed_dim:(i + 1) * self.embed_dim]
                ) for i in range(3)
            ]
        else:
            # join
            self.qkv_proj_params = nnx.Param(
                jnp.concat([p.value for p in self.qkv_proj_params], axis=-1)
            )

    def __call__(self, x: jax.Array, attn_options: dict[str, Any] = {}):
        assert x.ndim == 3

        batch_size, seqlen, embed_dim = x.shape
        num_heads = self.num_heads
        x = x.astype(self.dtype)
        
        out_proj_params = self.out_proj_params.value.astype(self.dtype)
        head_dim = embed_dim // num_heads


        out: jax.Array
        if self.qkv_packed:
            qkv_proj_params = self.qkv_proj_params.value.astype(self.dtype)

            qkv = jnp.dot(x, qkv_proj_params)
            qkv = qkv.reshape(batch_size, seqlen, 3, self.num_heads, head_dim)
           
            if self.normalize_qk:
                qkv = qkv.at[:, :, :2].set(self.norm_fn(qkv[:, :, :2]))
            out = self.attention_fn(qkv=qkv, **attn_options)
        else:
            q_proj_param, k_proj_param, v_proj_param = [p.value.astype(self.dtype) for p in self.qkv_proj_params]
            q, k, v = jax.tree.map(
                lambda proj_param: jnp.dot(x, proj_param).reshape(batch_size, seqlen, num_heads, head_dim), 
                [q_proj_param, k_proj_param, v_proj_param],
            )
            if self.normalize_qk:
                q = self.norm_fn(q)
                k = self.norm_fn(k)
            out = self.attention_fn(q=q, k=k, v=v, **attn_options)
                
        out = out.reshape(batch_size, seqlen, -1)
        out = jnp.dot(out, out_proj_params)
        return out
