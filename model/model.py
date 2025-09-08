# -*- encoding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import numbers
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def Win_Dila(x, win_size):
    """
    :param x: B, H, W, C
    :param
    :return: y: B, H, W, C
    """
    n_win = win_size * 2
    B, H, W, C = x.shape
    x = x.reshape(-1, (H // n_win), n_win, (W // n_win), n_win, C)  # B x n1 x n_win x n2 x n_win x C
    x = x.permute(0, 1, 3, 2, 4, 5)  # B x n1 x n2 x n_win x n_win x C

    xt = torch.zeros_like(x)
    x0 = x[:, :, :, 0::2, 0::2, :]
    x1 = x[:, :, :, 0::2, 1::2, :]
    x2 = x[:, :, :, 1::2, 0::2, :]
    x3 = x[:, :, :, 1::2, 1::2, :]

    xt[:, :, :, 0:n_win // 2, 0:n_win // 2, :] = x0  # B n/2 n/2 d2c
    xt[:, :, :, 0:n_win // 2, n_win // 2:n_win, :] = x1  # B n/2 n/2 d2c
    xt[:, :, :, n_win // 2:n_win, 0:n_win // 2, :] = x2  # B n/2 n/2 d2c
    xt[:, :, :, n_win // 2:n_win, n_win // 2:n_win, :] = x3  # B n/2 n/2 d2c
    xt = xt.permute(0, 1, 3, 2, 4, 5)
    xt = xt.reshape(-1, H, W, C)

    return xt


def Win_ReDila(x, win_size):
    """
    :param x: B, H, W, C
    :param
    :return: y: B, H, W, C
    """
    n_win = win_size * 2
    B, H, W, C = x.shape
    x = x.reshape(-1, (H // n_win), n_win, (W // n_win), n_win, C)  # B x n1 x n_win x n2 x n_win x C
    x = x.permute(0, 1, 3, 2, 4, 5)  # B x n1 x n2 x n_win x n_win x C

    xt = torch.zeros_like(x)
    xt[:, :, :, 0::2, 0::2, :] = x[:, :, :, 0:n_win // 2, 0:n_win // 2, :]
    xt[:, :, :, 0::2, 1::2, :] = x[:, :, :, 0:n_win // 2, n_win // 2:n_win, :]
    xt[:, :, :, 1::2, 0::2, :] = x[:, :, :, n_win // 2:n_win, 0:n_win // 2, :]
    xt[:, :, :, 1::2, 1::2, :] = x[:, :, :, n_win // 2:n_win, n_win // 2:n_win, :]

    xt = xt.permute(0, 1, 3, 2, 4, 5)
    xt = xt.reshape(-1, H, W, C)

    return xt


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        if L != H * W:
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class DMLA(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert self.num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        x_ = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2).clone()
        # num_dilation, B, H, W, C//num_dilation
        # y = torch.zeros_like(x)
        for i in range(self.num_dilation):
            x_[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
        # x = y
        x_ = x_.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)
        return x_


class Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample_unc(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_unc, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class DMST(nn.Module):
    def __init__(self, dim=32, input_resolution=(16, 16), num_heads=6, window_size=4, dilation=[1, 2, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0., kernel_size=3,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.dilation = dilation
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.cpe_per_block = cpe_per_block
        self.window_size = window_size
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        assert 0 <= self.window_size <= input_resolution[0], "input_resolution should be larger than window_size"
        assert 0 <= self.window_size <= input_resolution[1], "input_resolution should be larger than window_size"

        self.attn1 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.attn = DMLA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm7 = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        attn_windows = self.attn1(x_windows)  # nW*B, window_size*window_size, C
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp1(self.norm2(x)))

        x = self.norm7(x)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm3(x)))
        x = x.view(B, H * W, C)
        x = x + self.drop_path(self.mlp2(self.norm4(x)))

        shortcut = x
        x = self.norm5(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        attn_windows = self.attn2(x_windows)  # nW*B, window_size*window_size, C
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp3(self.norm6(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Dila_QKV(nn.Module):
    def __init__(self, c1, c2, stride=1, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False):
        super(Dila_QKV, self).__init__()
        self.stride = stride
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(c2))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def dila1(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[0],
                                   dilation=self.dilate[0])
        out += residual
        out = self.act(out)

        return out

    def dila2(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[1],
                                   dilation=self.dilate[1])
        out += residual
        out = self.act(out)

        return out

    def dila3(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[2],
                                   dilation=self.dilate[2])
        out += residual
        out = self.act(out)

        return out

    def forward(self, x):
        x1 = self.dila1(x)
        x2 = self.dila2(x)
        x3 = self.dila3(x)

        return x1, x2, x3


class DPEA(nn.Module):
    def __init__(self, dim, num_heads, win, bias):
        super(DPEA, self).__init__()
        self.chan_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim)
        )
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.win = win

        self.qkv = Dila_QKV(dim, dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        x_ = self.chan_embed(x)
        # x_ = x + x_

        # q, k, v = self.qkv(x_)
        q, k, v = self.qkv(x)

        q = q.view(B, C, H // self.win, self.win, W // self.win, self.win)
        q_win = q.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.win * self.win)

        k = k.view(B, C, H // self.win, self.win, W // self.win, self.win)
        k_win = k.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.win * self.win)

        v = v.view(B, C, H // self.win, self.win, W // self.win, self.win)
        v_win = v.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.win * self.win)

        q = rearrange(q_win, 'n (head c) win -> n head c win', head=self.num_heads)
        k = rearrange(k_win, 'n (head c) win -> n head c win', head=self.num_heads)
        v = rearrange(v_win, 'n (head c) win -> n head c win', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'n head c (h w) -> n (head c) h w', w=self.win)
        out = out.view(B, H // self.win, W // self.win, C, self.win, self.win)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

        out = out + x_

        out = self.project_out(out)

        return out


##########################################################################
class DPET(nn.Module):
    def __init__(self, dim, num_heads, win, ffn_expansion_factor, bias, LayerNorm_type):
        super(DPET, self).__init__()

        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DPEA(dim, num_heads, win, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Spa(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=31,
                 embed_dim=96, num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, cpe_per_block=False,
                 use_checkpoint=False, **kwargs):
        '''
        :param patch_size: for the embed conv
        :param in_chans: for the embed conv

        '''
        super(Spa, self).__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

        # split image into non-overlapping patches
        downsample = PatchMerging
        self.downsample1 = downsample(dim=self.embed_dim, input_resolution=(self.img_size, self.img_size),
                                      norm_layer=norm_layer)
        self.downsample2 = downsample(dim=self.embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),
                                      norm_layer=norm_layer)

        self.DMST1 = DMST(dim=self.embed_dim, input_resolution=(self.img_size, self.img_size), num_heads=num_heads[0],
                          window_size=window_size, cpe_per_block=cpe_per_block)
        self.DMST2 = DMST(dim=self.embed_dim * 2, input_resolution=(self.img_size // 2, self.img_size // 2),
                          num_heads=num_heads[1], window_size=window_size, cpe_per_block=cpe_per_block)
        self.DMST3 = DMST(dim=self.embed_dim * 4, input_resolution=(self.img_size // 4, self.img_size // 4),
                          num_heads=num_heads[2], window_size=window_size, cpe_per_block=cpe_per_block)

    def forward(self, x):
        B, _, H, _ = x.shape
        x = self.conv(x)  # B x em x H x W
        x = rearrange(x, 'B C H W -> B (H W) C ')
        Spa1_out = self.DMST1(x)
        Spa1_out_shape = rearrange(Spa1_out, 'B (H W) C -> B C H W', H=H)  # B x C x H x W
        Spa1_out = self.downsample1(Spa1_out)  # B (H/2 W/2) C
        H = H // 2

        Spa2_out = self.DMST2(Spa1_out)
        Spa2_out_shape = rearrange(Spa2_out, 'B (H W) C -> B C H W', H=H)  # B x 2C x H/2 x W/2
        Spa2_out = self.downsample2(Spa2_out)  # B (H/4 W/4) C
        H = H // 2

        Spa3_out = self.DMST3(Spa2_out)
        Spa3_out_shape = rearrange(Spa3_out, 'B (H W) C -> B C H W', H=H)  # B x 4C x H/4 x W/4

        return Spa1_out_shape, Spa2_out_shape, Spa3_out_shape


class Spe(nn.Module):
    def __init__(self, inp_channels=31, dim=32, num_heads=[8, 8, 8], group=8, ffn_expansion_factor=2.66,
                 LayerNorm_type='WithBias', bias=False):
        super(Spe, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.DPET1 = DPET(dim=dim, num_heads=num_heads[0], win=group, ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.up1_2 = Upsample_unc(int(dim))

        self.DPET2 = DPET(dim=dim, num_heads=num_heads[1], win=group, ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.up2_3 = Upsample_unc(int(dim))

        self.DPET3 = DPET(dim=dim, num_heads=num_heads[2], win=group, ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, x):
        emb = self.patch_embed(x)
        Spe_output1 = self.DPET1(emb)  # B x dim x h x w
        Spe_output1_up = self.up1_2(Spe_output1)  # B x dim/2 x 2h x 2w

        Spe_output2 = self.DPET2(Spe_output1_up)  # B x dim/2 x 2h x 2w
        Spe_output2_up = self.up2_3(Spe_output2)  # B x dim/4 x 4h x 4w

        Spe_output3 = self.DPET3(Spe_output2_up)  # B x dim/4 x 4h x 4w

        return Spe_output3, Spe_output2, Spe_output1


class GateMerge(nn.Module):
    def __init__(self, dim, bias):
        super(GateMerge, self).__init__()
        self.ptconv = nn.Conv2d(dim, 2 * dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x_ = self.ptconv(x)
        x1, x2 = self.dwconv(x_).chunk(2, dim=1)
        x_ = F.gelu(x1) * x2
        x_ = self.proj_out(x_)
        return x + x_


class Merge(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans1=34, in_chans2=31, embed_dim=48, num_heads1=[6, 6, 6],
                 window_size=8, group=8, mlp_ratio=4., dim=32,
                 num_heads2=[8, 8, 8], ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False,
                 cpe_per_block=True
                 ):
        super(Merge, self).__init__()
        self.Spa = Spa(img_size=img_size, patch_size=patch_size, in_chans=in_chans1, embed_dim=embed_dim,
                       num_heads=num_heads1, window_size=window_size, mlp_ratio=mlp_ratio, cpe_per_block=cpe_per_block)
        self.Spe = Spe(inp_channels=in_chans2, dim=dim, num_heads=num_heads2, group=group,
                       ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type, bias=bias)
        self.merge1 = nn.Sequential(nn.Conv2d(embed_dim * 4 + dim, dim, kernel_size=1, stride=1, bias=bias),
                                    nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias, groups=dim),
                                    GateMerge(dim, bias),
                                    nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=bias),
                                    nn.LeakyReLU(0.2, bias))

        self.up1 = Upsample_unc(int(dim))

        self.merge2 = nn.Sequential(nn.Conv2d(embed_dim * 2 + dim + dim, dim, kernel_size=1, stride=1, bias=bias),
                                    nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias, groups=dim),
                                    GateMerge(dim, bias),
                                    nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=bias),
                                    nn.LeakyReLU(0.2, bias))

        self.up2 = Upsample_unc(int(dim))

        self.merge3 = nn.Sequential(nn.Conv2d(embed_dim + dim + dim, in_chans2, kernel_size=1, stride=1, bias=bias),
                                    nn.Conv2d(in_chans2, in_chans2, kernel_size=3, padding=1, bias=bias,
                                              groups=in_chans2),
                                    GateMerge(in_chans2, bias),
                                    nn.Conv2d(in_chans2, in_chans2, kernel_size=5, padding=2, bias=bias), )

        self.final_proj = nn.Conv2d(in_chans2, in_chans2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, y):
        F_Spa1, F_Spa2, F_Spa3 = self.Spa(x)
        F_Spe3, F_Spe2, F_Spe1 = self.Spe(y)

        merge1 = self.merge1(torch.cat((F_Spa3, F_Spe1), 1))
        merge2 = self.merge2(torch.cat((F_Spa2, F_Spe2, self.up1(merge1)), 1))
        output = self.merge3(torch.cat((F_Spa1, F_Spe3, self.up2(merge2)), 1))

        output = self.final_proj(output)

        return output

