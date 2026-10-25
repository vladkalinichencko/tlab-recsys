"""Attention between rows that keeps the count: SiLU(q·k / sqrt(d)) / n in place of softmax (Meta HSTU).

softmax divides every weight by the sum of weights, so ten matching events and one give the same
mean; SiLU / n divides by the fixed number of rows n, so the output grows with the number of matches.
use(model) switches only the in-context block of TabICL (between rows); the blocks over a column and
inside a row keep their pretrained softmax.
"""
import torch.nn.functional as F
from tabicl._model import attention

_softmax_attention = attention.sdpa_with_flattened_batch
_active = False


def counting_attention(q, k, v, attn_mask=None, dropout_p=0.0, ssmax_layer=None):
    if not _active:
        return _softmax_attention(q, k, v, attn_mask, dropout_p, ssmax_layer)
    if ssmax_layer is not None:
        shape = q.shape
        q = ssmax_layer(q.reshape(-1, *shape[-3:]), k.size(-2)).reshape(shape)
    score = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
    if attn_mask is not None and attn_mask.dtype.is_floating_point:
        score = score + attn_mask
    elif attn_mask is not None:
        score = score.masked_fill(~attn_mask, float("-inf"))
    weight = F.silu(score).nan_to_num(0.0)  # SiLU(-inf) is nan, a masked row weighs 0
    return weight @ v / k.size(-2)


def use(model):
    global _active

    def on(*_):
        global _active
        _active = True

    def off(*_):
        global _active
        _active = False

    attention.sdpa_with_flattened_batch = counting_attention
    model.icl_predictor.register_forward_pre_hook(on)
    model.icl_predictor.register_forward_hook(off)
