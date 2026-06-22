from inspect import isfunction
import math
# from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
from model_lib.ControlNet.ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.,
        checkpoint=False,
        style_attention_scale=1.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.style_attention_scale = style_attention_scale

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.checkpoint = checkpoint

    def forward(self, x, context=None, apply_adain=False, return_qkv=False, style_mask=None):
        return checkpoint(self._forward, (x, context, apply_adain, return_qkv, style_mask), self.parameters(), self.checkpoint)

    def adain(self, content, style, eps=1e-5):
        mean_c = content.mean(dim=1, keepdim=True)
        std_c = content.std(dim=1, keepdim=True) + eps
        mean_s = style.mean(dim=1, keepdim=True)
        std_s = style.std(dim=1, keepdim=True) + eps
        return std_s * (content - mean_c) / std_c + mean_s

    def apply_key_style_modulation(self, k, v, batch_size, hw, style_mask=None):
        if k.shape[1] % 5 != 0:
            raise ValueError(
                "StyleFusion attention expects context tokens arranged as "
                "[current, content-front/back, style-front/back]."
            )

        tokens_per_view = k.shape[1] // 5
        structure_k = k[:, :tokens_per_view, :]
        content_k = k[:, tokens_per_view:3 * tokens_per_view, :]
        style_k = k[:, 3 * tokens_per_view:5 * tokens_per_view, :]

        structure_v = v[:, :tokens_per_view, :]
        content_v = v[:, tokens_per_view:3 * tokens_per_view, :]
        style_v = v[:, 3 * tokens_per_view:5 * tokens_per_view, :]

        modulated_content_k = self.adain(content_k, style_k)

        if style_mask is not None:
            if style_mask.dim() == 3:
                style_mask = style_mask.unsqueeze(1)

            H = W = int(math.sqrt(hw))
            style_mask = F.interpolate(style_mask, size=(H, W), mode='bilinear', align_corners=False)
            style_mask = style_mask.repeat(batch_size * self.heads, 1, 1, 1)
            style_mask = style_mask.flatten(2).permute(0, 2, 1).contiguous()
            style_mask = style_mask.expand(-1, -1, self.dim_head)

            if style_mask.shape != style_k.shape:
                style_mask = style_mask[:, :style_k.shape[1], :style_k.shape[2]]

            front_tokens = style_k.shape[1] // 2
            modulated_content_k[:, :front_tokens, :] = (
                modulated_content_k[:, :front_tokens, :] * style_mask
                + content_k[:, :front_tokens, :] * (1 - style_mask)
            )
            style_k[:, :front_tokens, :] = style_k[:, :front_tokens, :] * style_mask

        style_k = style_k * self.style_attention_scale
        k = torch.cat([structure_k, style_k, modulated_content_k], dim=1)
        v = torch.cat([structure_v, style_v, content_v], dim=1)
        return k, v

    def _forward(self, x, context=None, apply_adain=False, return_qkv=False, style_mask=None):
        h = self.heads
        b, hw, _ = x.shape

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if apply_adain:
            k, v = self.apply_key_style_modulation(k, v, b, hw, style_mask=style_mask)

        q_for_return, k_for_return = (q, k) if return_qkv else (None, None)

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if not return_qkv:
            del q, k
    
        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_qkv:
            return self.to_out(out), (q_for_return, k_for_return, v)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    """Cross-attention with StyleFusion360 key-only adaptive style modulation.

    During style read mode, context is arranged as
    [current tokens, content reference tokens, style reference tokens]. The
    content and style reference banks each contain the front/back token pair,
    so the projected key/value sequence is split into five equal chunks:
    current, content-front/back, and style-front/back.

    StyleFusion360 modulates only the content keys with style statistics before
    attention. Values are kept unchanged to preserve identity and geometry while
    the style-scaled keys guide where stylization is injected.
    """
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, checkpoint=False, style_attention_scale=1.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None
        self.checkpoint = checkpoint

        self.style_attention_scale = style_attention_scale

    def forward(self, x, context=None, apply_adain=False, return_qkv=False, style_mask=None):
        return checkpoint(self._forward, (x, context, apply_adain, return_qkv, style_mask), self.parameters(), self.checkpoint)

    def adain(self, content, style, eps=1e-5):
        mean_c = content.mean(dim=1, keepdim=True)
        std_c = content.std(dim=1, keepdim=True) + eps
        mean_s = style.mean(dim=1, keepdim=True)
        std_s = style.std(dim=1, keepdim=True) + eps
        return std_s * (content - mean_c) / std_c + mean_s

    def _forward(self, x, context=None, apply_adain=False, return_qkv=False, style_mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, hw, c = x.shape

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if apply_adain:
            if k.shape[1] % 5 != 0:
                raise ValueError(
                    "StyleFusion attention expects context tokens arranged as "
                    "[current, content-front/back, style-front/back]."
                )

            tokens_per_view = k.shape[1] // 5
            structure_k = k[:, :tokens_per_view, :]
            content_k = k[:, tokens_per_view:3 * tokens_per_view, :]
            style_k = k[:, 3 * tokens_per_view:5 * tokens_per_view, :]

            structure_v = v[:, :tokens_per_view, :]
            content_v = v[:, tokens_per_view:3 * tokens_per_view, :]
            style_v = v[:, 3 * tokens_per_view:5 * tokens_per_view, :]

            # Adaptive Style Modulation: key-only AdaIN. The modulated content
            # keys carry style statistics, while values stay unchanged to keep
            # identity and geometry information stable.
            modulated_content_k = self.adain(content_k, style_k)

            if style_mask is not None:
                if style_mask.dim() == 3:
                    style_mask = style_mask.unsqueeze(1)

                H = W = int(math.sqrt(hw))
                style_mask = F.interpolate(style_mask, size=(H, W), mode='bilinear', align_corners=False)
                style_mask = style_mask.repeat(b * self.heads, 1, 1, 1)
                style_mask = style_mask.flatten(2).permute(0, 2, 1).contiguous()
                style_mask = style_mask.expand(-1, -1, self.dim_head)
                        
                if style_mask.shape != style_k.shape:
                    style_mask = style_mask[:, :style_k.shape[1], :style_k.shape[2]]

                front_tokens = style_k.shape[1] // 2
                modulated_content_k[:, :front_tokens, :] = (
                    modulated_content_k[:, :front_tokens, :] * style_mask
                    + content_k[:, :front_tokens, :] * (1 - style_mask)
                )
                style_k[:, :front_tokens, :] = style_k[:, :front_tokens, :] * style_mask

            style_k = style_k * self.style_attention_scale
            k = torch.cat([structure_k, style_k, modulated_content_k], dim=1)
            v = torch.cat([structure_v, style_v, content_v], dim=1)

        
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)


        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        if return_qkv:
            return self.to_out(out), (q, k, v)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, style_attention_scale=1.0):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, checkpoint=checkpoint, style_attention_scale=style_attention_scale,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, style_attention_scale=style_attention_scale,
                              heads=n_heads, dim_head=d_head, dropout=dropout, checkpoint=checkpoint,)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    # def forward(self, x, context=None, banks=None, attention_mode=None, attn_index=None,uc=False):
    #     return checkpoint(self._forward, (x, context, banks, attention_mode, attn_index,uc), self.parameters(), self.checkpoint)

    def forward(self, x, context=None, banks=None, attention_mode=None, attn_index=None, uc=False, style_attn_banks=None, style_write=False, index=0, style_mask=None):

        if uc or attention_mode is None:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        else:
            x_norm1 = self.norm1(x)
            self_attn1 = None
            self_attention_info = x_norm1
            if attention_mode == 'read':

                reference_info = banks[attn_index] # reference info (input, style)

                if len(reference_info) > 0:
                    # kv will be attention from previous banks
                    #import pdb;pdb.set_trace()

                    input_info, style_info = reference_info.chunk(2, dim=1)

                    # q = torch.cat([x_norm1, style_concat], dim=1)
                    if style_write:
                        k_v = torch.cat([self_attention_info, reference_info], dim=1)
                        self_attn1 = self.attn1(x_norm1, context = k_v, apply_adain=True, style_mask=style_mask)    
                    else:
                        k_v = torch.cat([self_attention_info, input_info], dim=1)
                        self_attn1 = self.attn1(x_norm1, context = k_v)

             
                if self_attn1 is None:
                    self_attn1 = self.attn1(x_norm1, context = self_attention_info)
                x = self_attn1 + x

            elif attention_mode == 'write':
                #bank = []self_attention_info
                #bank.append(self_attention_info)

                banks.append(self_attention_info)

                if self_attn1 is None:

                    self_attn1 = self.attn1(x_norm1, context = self_attention_info)

                x = self_attn1 + x

            else:
                raise NotImplementedError

        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False, style_attention_scale=1.0,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, style_attention_scale=style_attention_scale)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, banks=None, attention_mode=None, attn_index=None,uc=False, style_attn_banks=None, style_write=False, index=0, style_mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], banks=banks, attention_mode=attention_mode, attn_index=attn_index,uc=uc, style_attn_banks=style_attn_banks, style_write=style_write, index=index, style_mask=style_mask)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
