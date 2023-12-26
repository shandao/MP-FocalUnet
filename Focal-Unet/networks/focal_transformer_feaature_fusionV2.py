import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models import resnet34 as resnet


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


def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # pad feature maps to multiples of window size
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
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


def get_topk_closest_indice(q_windows, k_windows, topk=1):
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])

    if q_windows[0] != k_windows[0]:
        factor = k_windows[0] // q_windows[0]
        coords_h_q = coords_h_q * factor + factor // 2
        coords_w_q = coords_w_q * factor + factor // 2
    else:
        factor = 1

    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k

    relative_position_dists = torch.sqrt(relative_coords[0].float() ** 2 + relative_coords[1].float() ** 2)

    topk = min(topk, relative_position_dists.shape[1])
    topk_score_k, topk_index_k = torch.topk(-relative_position_dists, topk, dim=1)  # B, topK, nHeads
    indice_topk = topk_index_k
    relative_coord_topk = torch.gather(relative_coords, 2, indice_topk.unsqueeze(0).repeat(2, 1, 1))
    return indice_topk, relative_coord_topk.permute(1, 2, 0).contiguous().float(), topk


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm,istif=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        if istif:
            self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
            self.norm = norm_layer(dim)
        else:
            self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
            self.norm = norm_layer(dim//dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2,p2=2 ,c=C //4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, expand_size, shift_size, window_size, window_size_glo, focal_window,
                 focal_level, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none",
                 topK=64):

        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.window_size_glo = window_size_glo
        self.pool_method = pool_method
        self.input_resolution = input_resolution  # NWh, NWw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.nWh, self.nWw = self.input_resolution[0] // self.window_size[0], self.input_resolution[1] // \
                             self.window_size[1]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.topK = topK

        coords_h_window = torch.arange(self.window_size[0]) - self.window_size[0] // 2
        coords_w_window = torch.arange(self.window_size[1]) - self.window_size[1] // 2
        coords_window = torch.stack(torch.meshgrid([coords_h_window, coords_w_window]), dim=-1)  # 2, Wh_q, Ww_q
        self.register_buffer("window_coords", coords_window)

        self.coord2rpb_all = nn.ModuleList()

        self.topks = []
        for k in range(self.focal_level):
            if k == 0:
                range_h = self.input_resolution[0]
                range_w = self.input_resolution[1]
            else:
                range_h = self.nWh
                range_w = self.nWw

            # build relative position range
            topk_closest_indice, topk_closest_coord, topK_updated = get_topk_closest_indice(
                (self.nWh, self.nWw), (range_h, range_w), self.topK)
            self.topks.append(topK_updated)

            if k > 0:
                # scaling the coordinates for pooled windows
                topk_closest_coord = topk_closest_coord * self.window_size[0]
            topk_closest_coord_window = topk_closest_coord.unsqueeze(1) + coords_window.view(-1, 2)[None, :, None, :]

            self.register_buffer("topk_cloest_indice_{}".format(k), topk_closest_indice)
            self.register_buffer("topk_cloest_coords_{}".format(k), topk_closest_coord_window)

            coord2rpb = nn.Sequential(
                nn.Linear(2, head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(head_dim, self.num_heads)
            )
            self.coord2rpb_all.append(coord2rpb)

    def forward(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x_all[0]  #

        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, nW, C

        # partition q map
        q_windows = window_partition(q, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
        ).transpose(1, 2)

        k_all = [];
        v_all = [];
        topKs = [];
        topk_rpbs = []
        for l_k in range(self.focal_level):
            topk_closest_indice = getattr(self, "topk_cloest_indice_{}".format(l_k))
            topk_indice_k = topk_closest_indice.view(1, -1).repeat(B, 1)

            topk_coords_k = getattr(self, "topk_cloest_coords_{}".format(l_k))
            window_coords = getattr(self, "window_coords")

            topk_rpb_k = self.coord2rpb_all[l_k](topk_coords_k)
            topk_rpbs.append(topk_rpb_k)

            if l_k == 0:
                k_k = k.view(B, -1, self.num_heads, C // self.num_heads)
                v_k = v.view(B, -1, self.num_heads, C // self.num_heads)
            else:
                x_k = x_all[l_k]
                qkv_k = self.qkv(x_k).view(B, -1, 3, self.num_heads, C // self.num_heads)
                k_k, v_k = qkv_k[:, :, 1], qkv_k[:, :, 2]

            k_k_selected = torch.gather(k_k, 1, topk_indice_k.view(B, -1, 1).unsqueeze(-1).repeat(1, 1, self.num_heads,
                                                                                                  C // self.num_heads))
            v_k_selected = torch.gather(v_k, 1, topk_indice_k.view(B, -1, 1).unsqueeze(-1).repeat(1, 1, self.num_heads,
                                                                                                  C // self.num_heads))

            k_k_selected = k_k_selected.view(
                (B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)
            v_k_selected = v_k_selected.view(
                (B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)

            k_all.append(k_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            v_all.append(v_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            topKs.append(topk_closest_indice.shape[1])

        k_all = torch.cat(k_all, 2)
        v_all = torch.cat(v_all, 2)

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        attn = (q_windows @ k_all.transpose(-2,
                                            -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size
        window_area = self.window_size[0] * self.window_size[1]
        window_area_whole = k_all.shape[2]

        topk_rpb_cat = torch.cat(topk_rpbs, 2).permute(0, 3, 1, 2).contiguous().unsqueeze(0).repeat(B, 1, 1, 1, 1).view(
            attn.shape)
        attn = attn + topk_rpb_cat

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v_all).transpose(1, 2).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, window_size, unfold_size):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        for k in range(self.focal_level):
            flops += self.num_heads * N * (self.dim // self.num_heads) * self.topks[k]
            # relative position embedding
            if k == 0:
                Nq = N
            else:
                window_size_glo = math.floor(self.window_size[0] / (2 ** (k - 1)))
                Nq = N // (window_size_glo ** 2)
            flops += Nq * self.topks[k] * (
                    2 * (self.dim // self.num_heads) + (self.dim // self.num_heads) * self.num_heads)

        #  x = (attn @ v)
        for k in range(self.focal_level):
            flops += self.num_heads * N * (self.dim // self.num_heads) * self.topks[k]

            # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class FocalTransformerBlock(nn.Module):
    r""" Focal Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none",
                 focal_level=1, focal_window=1, topK=64, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            # self.focal_level = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo, 1))
                    self.pool_layers[-1].weight.data.fill_(1. / (window_size_glo * window_size_glo))
                    self.pool_layers[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers.append(
                        nn.Conv2d(dim, dim, kernel_size=window_size_glo, stride=window_size_glo, groups=dim))

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, expand_size=self.expand_size, shift_size=self.shift_size,
            window_size=to_2tuple(self.window_size),
            window_size_glo=to_2tuple(self.window_size_glo), focal_window=focal_window,
            focal_level=self.focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            pool_method=pool_method, topK=topK)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]

        if self.focal_level > 1 and self.pool_method != "none":
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                pooled_h = math.ceil(H / self.window_size) * (2 ** k)
                pooled_w = math.ceil(W / self.window_size) * (2 ** k)
                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0, 0, 0, 0, pad_t, pad_b))

                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0, 0, pad_l, pad_r))

                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(),
                                                                 window_size_glo)  # B, nw, nw, window_size, window_size, C
                nWh, nWw = x_windows_noreshape.shape[1:3]
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4])  # B, nWh, nWw, C
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(B, nWh, nWw,
                                                                                      C)  # B, nWh, nWw, C
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(B, nWh, nWw, window_size_glo * window_size_glo,
                                                                   C).transpose(3, 4)  # B, nWh, nWw, C, wsize**2
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(
                        -2)  # B, nWh, nWw, C
                elif self.pool_method == "conv":
                    x_windows_noreshape = x_windows_noreshape.view(-1, window_size_glo, window_size_glo, C).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2).contiguous()  # B * nw * nw, C, wsize, wsize
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, nWh, nWw,
                                                                                     C)  # B, nWh, nWw, C

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]

        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)  # nW*B, window_size*window_size, C

        attn_windows = attn_windows[:, :self.window_size ** 2]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(
            self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W

        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size, self.window_size, self.focal_window)

        if self.pool_method != "none" and self.focal_level > 1:
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                nW_glo = nW * (2 ** k)
                # (sub)-window pooling
                flops += nW_glo * self.dim * window_size_glo * window_size_glo
                # qkv for global levels
                # NOTE: in our implementation, we pass the pooled window embedding to qkv embedding layer,
                # but theoritically, we only need to compute k and v.
                flops += nW_glo * self.dim * 3 * self.dim

                # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        img_size (tuple[int]): Resolution of input feature.
        in_chans (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=nn.LayerNorm,
                 use_pre_norm=False, is_stem=False):
        super().__init__()
        self.input_resolution = img_size
        self.dim = in_chans
        self.reduction = nn.Linear(4 * in_chans, 2 * in_chans, bias=False)
        self.norm = norm_layer(4 * in_chans)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none",
                 focal_level=1, focal_window=1, topK=64, use_conv_embed=False, use_shift=False, use_pre_norm=False,
                 downsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1

        # build blocks
        self.blocks = nn.ModuleList([
            FocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, window_size=window_size,
                                  shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                  expand_size=0 if (i % 2 == expand_factor) else expand_size,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop,
                                  attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer,
                                  pool_method=pool_method,
                                  focal_level=focal_level,
                                  focal_window=focal_window,
                                  topK=topK,
                                  use_layerscale=use_layerscale,
                                  layerscale_value=layerscale_value)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size=2, in_chans=dim, embed_dim=2 * dim,
                use_conv_embed=use_conv_embed, norm_layer=norm_layer, use_pre_norm=use_pre_norm,
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = x.view(x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1,
                                                                                                   2).contiguous()
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none",
                 focal_level=1, focal_window=1, topK=64, use_conv_embed=False, use_shift=False, use_pre_norm=False,
                 upsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1

        # build blocks
        self.blocks = nn.ModuleList([
            FocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, window_size=window_size,
                                  shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                  expand_size=0 if (i % 2 == expand_factor) else expand_size,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop,
                                  attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer,
                                  pool_method=pool_method,
                                  focal_level=focal_level,
                                  focal_window=focal_window,
                                  topK=topK,
                                  use_layerscale=use_layerscale,
                                  layerscale_value=layerscale_value)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x)
            else:
                x1 = blk(x)

        if self.upsample is not None:
            # x = x.view(x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1,
            #                                                                                        2).contiguous()
            x = self.upsample(x1)
        return x, x1

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False,
                 norm_layer=None, use_pre_norm=True, is_stem=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm
        self.use_conv_embed = use_conv_embed

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7;
                padding = 2;
                stride = 4
            else:
                kernel_size = 3;
                padding = 1;
                stride = 2
            self.kernel_size = kernel_size
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class ds_PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=8, in_chans=3, embed_dim=96, use_conv_embed=False,
                 norm_layer=None, use_pre_norm=True, is_stem=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm
        self.use_conv_embed = use_conv_embed

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 15;
                padding = 4;
                stride = 8
            else:
                kernel_size = 3;
                padding = 1;
                stride = 2
            self.kernel_size = kernel_size
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        if self.use_conv_embed:
            flops = Ho * Wo * self.embed_dim * self.in_chans * (self.kernel_size ** 2)
        else:
            flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, number, hidden_size, dropout_rate=0.2):
        super(Embeddings, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, number, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Transformer_Attention, self).__init__()
        # self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, }


class Transformer_Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Transformer_Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Transformer_Mlp(hidden_size, mlp_dim, dropout_rate)
        self.attn = Transformer_Attention(hidden_size, num_heads, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, mlp_dim, dropout_rate, num_heads, attention_dropout_rate):
        super(Encoder, self).__init__()
        # self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self, number, hidden_size, num_layers, mlp_dim, dropout_rate, num_heads, attention_dropout_rate):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(number, hidden_size)
        self.encoder = Encoder(hidden_size, num_layers, mlp_dim, dropout_rate, num_heads, attention_dropout_rate)

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoded = self.encoder(embedding_output)
        return encoded


class AVG(nn.Module):
    def __init__(self):
        super(AVG, self).__init__()

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        # x = torch.flatten(x, 2).permute(0, 2, 1)
        return x


class LFR(nn.Module):
    def __init__(self, channel_high, channel_low, out_channel, c_mid=64, scale=2, k_up=5, k_enc=3, r_2=4,drop_rate=0):
        super().__init__()
        self.scale = scale
        self.inter_channel = channel_high

        self.conv_reduce = ConvBNReLU(channel_high, self.inter_channel, kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv1 = ConvBNReLU(channel_high, c_mid, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2 = ConvBNReLU(channel_low*2, c_mid, kernel_size=1, stride=1, padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid * 2, k_up ** 2, kernel_size=k_enc, stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)

        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)
        self.fc1 = nn.Conv2d(channel_high, channel_high // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel_high // r_2, channel_high, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        # self.fusion = BiFusion_block(ch_1=channel_high, ch_2=channel_low, r_2=r_2, ch_int=self.inter_channel, ch_out=out_channel, drop_rate=drop_rate)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channel + channel_low, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        # self.conv_out_fusion = nn.Sequential(
        #     nn.Conv2d(self.inter_channel + channel_low, out_channel, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(),
        # )

    def forward(self, x_h, x_l):
        b, N, C = x_h.size()
        h, w = int(math.sqrt(N)), int(math.sqrt(N))
        x_h = x_h.permute(0, 2, 1).view(b, C, h, w)
        # b, c, h, w = x_h.size()
        h_, w_ = h * self.scale, w * self.scale
        x_l = x_l.permute(0, 2, 1).view(b, -1, h_, w_)
        x1 = F.interpolate(self.conv1(x_h), size=(h_, w_), mode="bilinear", align_corners=True)  # b * m * h * w
        # x3 = F.interpolate(x_h, size=(h_, w_), mode="bilinear", align_corners=True)
        g_in = x_l
        g_c = x_l.mean((2, 3), keepdim=True)
        g_c = self.fc1(g_c)
        g_c = self.relu(g_c)
        g_c = self.fc2(g_c)
        g_c = self.sigmoid(g_c) * g_in
        x2 = self.conv2(torch.cat((g_c, x_l), dim=1))

        # x_in = x_l
        # # x_c = x
        # x = self.compress_c(x_l)
        # x = self.spatial_c(x)
        # x = self.sigmoid_c(x) * x_in
        # x2 = self.conv2(x)
        W = self.enc(torch.cat((x1, x2), dim=1))  # b * 100 * h * w
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        x_h_reduce = self.conv_reduce(x_h)

        X = self.unfold(
            F.interpolate(x_h_reduce, size=(h_, w_), mode="bilinear", align_corners=True))  # b * 25c * h_ * w_
        X = X.view(b, self.inter_channel, -1, h_, w_)  # b * c * 25 * h_ * w_
        X = torch.mul(W.unsqueeze(1), X).sum(dim=2)
        y = self.conv_out(torch.cat((X, x_l), dim=1))
        # y1 = self.fusion(x3, x_l)
        return y



class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, use_bn=True, use_relu=True, coord=False):
        super(ConvBNReLU, self).__init__()
        self.coord = coord
        if coord:
            c_in += 2

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)

        if use_bn:
            self.bn = nn.BatchNorm2d(c_out)
        else:
            self.bn = None
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        if self.coord:
            x = self.concat_grid(x)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def concat_grid(self, input):
        b, c, out_h, out_w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2).permute(2, 0, 1)  #(2, h, w)
        grid = grid.repeat(b, 1, 1, 1).type_as(input).to(input.device)
        input = torch.cat((grid, input), dim=1)
        return input

class TIF(nn.Module):
    def __init__(self, dim, x_number, y_number,img_size):
        super(TIF, self).__init__()
        self.avg = AVG()

        self.x_transformer = Transformer(x_number+1, dim, 2, 1024, 0.1, 8, 0.0)
        self.y_transformer = Transformer(y_number+1, dim, 2, 1024, 0.1, 8, 0.0)
        # self.pool = nn.MaxPool2d(2)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.res = DoubleConv(2 * dim, dim)
        self.patch_expand = PatchExpand(img_size,dim,istif=True)
        self.concat_linear = nn.Linear(2*dim,dim)

    def forward(self, x, y):
        x1 = x
        y1 = y
        x = self.avg(x)
        # B1, C1, H1, W1 = x1.shape
        # x1 = x1.view(B1, C1, -1).permute(0, 2, 1)

        y = self.avg(y)
        # B, C, H, W = y1.shape
        # y1 = y1.view(B, C, -1).permute(0, 2, 1)
        y1 = torch.cat((x, y1), dim=1)
        x1 = torch.cat((y, x1), dim=1)
        x = self.x_transformer(x1)
        x = x[:, 1:, :]
        # x = x.permute(0, 2, 1).contiguous().view(B1, C1, H1, W1)
        y = self.y_transformer(y1)
        y = y[:, 1:, :]
        # y = y.permute(0, 2, 1).contiguous().view(B, C, H, W)
        y=self.patch_expand(y)
        # y = self.pool(y)
        x = torch.cat((x, y), dim=-1)
        x = self.concat_linear(x)
        # x = self.res(x)
        return x


class FocalTransformerSys(nn.Module):
    r""" Swin Transformer
           A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
             https://arxiv.org/pdf/2103.14030

       Args:
           img_size (int | tuple(int)): Input image size. Default 224
           patch_size (int | tuple(int)): Patch size. Default: 4
           in_chans (int): Number of input image channels. Default: 3
           num_classes (int): Number of classes for classification head. Default: 1000
           embed_dim (int): Patch embedding dimension. Default: 96
           depths (tuple(int)): Depth of each Swin Transformer layer.
           num_heads (tuple(int)): Number of attention heads in different layers.
           window_size (int): Window size. Default: 7
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
           qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
           qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
           drop_rate (float): Dropout rate. Default: 0
           attn_drop_rate (float): Attention dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
       """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], depths_decoder=[1, 6, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, use_shift=False, focal_stages=[0, 1, 2, 3], focal_levels=[2, 2, 2, 2],
                 focal_windows=[7, 5, 3, 1], focal_topK=128, focal_pool="fc", expand_sizes=[3, 3, 3, 3],
                 expand_layer="all",
                 use_conv_embed=True, use_layerscale=False, layerscale_value=1e-4, use_pre_norm=False,
                 final_upsample="expand_first", pretrained=True, **kwargs):
        super().__init__()

        print(
            "FocalTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.Fuse_up = []
        self.resnet_up = []

        self.resnet = resnet()
        # device = torch.device("cpu")
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.drop = nn.Dropout2d(drop_rate)
        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        # self.up_c = BiFusion_block(ch_1=384, ch_2=384, r_2=4, ch_int=384, ch_out=384, drop_rate=drop_rate / 2)
        # # self.Fuse_up.append(self.up_c)
        # self.up_c_1_1 = BiFusion_block(ch_1=192, ch_2=192, r_2=2, ch_int=192, ch_out=192, drop_rate=drop_rate / 2)
        # self.up_c_1_2 = Up(in_ch1=384, out_ch=192, in_ch2=192, attn=True)
        # # self.Fuse_up.append(self.up_c_1_1)
        # self.up_c_2_1 = BiFusion_block(ch_1=96, ch_2=96, r_2=1, ch_int=96, ch_out=96, drop_rate=drop_rate / 2)
        # self.up_c_2_2 = Up(192, 96, 96, attn=True)
        # # self.Fuse_up.append(self.up_c_2_1)
        self.conv1 = DoubleConv(256, 384)
        self.conv2 = DoubleConv(128, 192)
        self.conv3 = DoubleConv(64, 96)
        # self.up_c = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        #
        # self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate / 2)
        # self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)
        #
        # self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate / 2)
        # self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            use_conv_embed=use_conv_embed, is_stem=True,
            norm_layer=norm_layer if self.patch_norm else None)

        self.ds_patch_embed = ds_PatchEmbed(
            img_size=to_2tuple(img_size), patch_size=patch_size*2, in_chans=in_chans, embed_dim=embed_dim,
            use_conv_embed=use_conv_embed, is_stem=True,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        ds_num_patches = self.ds_patch_embed.num_patches
        ds_patches_resolution = self.ds_patch_embed.patches_resolution
        self.ds_patches_resolution = ds_patches_resolution

        self.donv_up1 = LFR(384, 384, 384, c_mid=384, r_2=4,drop_rate=drop_rate / 2)
        self.donv_up2 = LFR(192, 192, 192, c_mid=384, r_2=2,drop_rate=drop_rate / 2)
        self.donv_up3 = LFR(96, 96, 96, c_mid=384, r_2=1,drop_rate=drop_rate / 2)
        # self.donv_up4 = LFR(96, 96, 96, c_mid=384)

        # self.tif = nn.ModuleList()
        # for i in range(self.num_layers-2):
        #     i=i+1
        #     tif = TIF (self.embed_dim*(2**i),(patches_resolution[0]//(2**i))*(patches_resolution[1]//(2**i)),(patches_resolution[0]//(2**(i+1))*(patches_resolution[1]//(2**(i+1)))),(patches_resolution[0]//(2**(i+1)),patches_resolution[1]//(2**(i+1))))
        #     self.tif.append(tif)
        # self.tif1 = TIF(96, 3136, 784)
        # self.tif2 = TIF(192, 784, 192)
        # self.tif3 = TIF(384, 192, 96)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               pool_method=focal_pool if i_layer in focal_stages else "none",
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer],
                               focal_window=focal_windows[i_layer],
                               topK=focal_topK,
                               expand_size=expand_sizes[i_layer],
                               expand_layer=expand_layer,
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift,
                               use_pre_norm=use_pre_norm,
                               use_checkpoint=use_checkpoint,
                               use_layerscale=use_layerscale,
                               layerscale_value=layerscale_value)
            self.layers.append(layer)

        self.ds_layers = nn.ModuleList()
        for ds_i_layer in range(self.num_layers-1):
            layer = BasicLayer(dim=int(embed_dim * 2 ** ds_i_layer),
                               input_resolution=(ds_patches_resolution[0] // (2 ** ds_i_layer),
                                                 ds_patches_resolution[1] // (2 ** ds_i_layer)),
                               depth=depths[ds_i_layer],
                               num_heads=num_heads[ds_i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:ds_i_layer]):sum(depths[:ds_i_layer + 1])],
                               norm_layer=norm_layer,
                               pool_method=focal_pool if ds_i_layer in focal_stages else "none",
                               downsample=PatchEmbed if (ds_i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[ds_i_layer],
                               focal_window=focal_windows[ds_i_layer],
                               topK=focal_topK,
                               expand_size=expand_sizes[ds_i_layer],
                               expand_layer=expand_layer,
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift,
                               use_pre_norm=use_pre_norm,
                               use_checkpoint=use_checkpoint,
                               use_layerscale=use_layerscale,
                               layerscale_value=layerscale_value)
            self.ds_layers.append(layer)


        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(3 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop=drop_rate,
                                         attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         pool_method=focal_pool if i_layer in focal_stages else "none",
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         focal_level=focal_levels[(self.num_layers - 1 - i_layer)],
                                         focal_window=focal_windows[(self.num_layers - 1 - i_layer)],
                                         topK=focal_topK,
                                         expand_size=expand_sizes[(self.num_layers - 1 - i_layer)],
                                         expand_layer=expand_layer,
                                         use_conv_embed=use_conv_embed,
                                         use_shift=use_shift,
                                         use_pre_norm=use_pre_norm,
                                         use_checkpoint=use_checkpoint,
                                         use_layerscale=use_layerscale,
                                         layerscale_value=layerscale_value)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    def ds_forward_features(self, x):
        x = self.ds_patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for ds_layer in self.ds_layers:
            x_downsample.append(x)
            x = ds_layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample, resnet_up=None):
        x_upsample = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                B, N, C = x.size()
                H, W = int(math.sqrt(N)), int(math.sqrt(N))
                # x_upsample.append(x.permute(0, 2, 1).view(B, C, H, W).contiguous())
                # x = self.Fuse_up[inx-1](x_downsample[3 - inx].permute(0, 2, 1).view(B, C, H, W), x.permute(0, 2, 1).view(B, C, H, W))
                x = torch.cat(
                    [x, x_downsample[3 - inx].view(B, C, N).permute(0, 2, 1).contiguous(), resnet_up[3 - inx].view(B, C, N).permute(0, 2, 1).contiguous()], -1)
                x = self.concat_back_dim[inx](x)
                # x = x.view(B, C, N).permute(0, 2, 1).contiguous()
                x, x1 = layer_up(x)
                x_upsample.append(x1.permute(0, 2, 1).view(B, C, H, W).contiguous())

        x = self.norm_up(x)  # B L C

        return x, x_upsample

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    # def ds_tif(self,x_downsample,ds_x_downsample):
    #     x_tif = []
    #     for inx, tif in enumerate(self.tif):
    #         x = tif (x_downsample[inx+1],ds_x_downsample[inx+1])
    #         x_tif.append(x)
    #     return x_tif




    def forward(self, x):
        g = x
        resnet_up = []
        # self.resnet.to("cuda:0")
        x_u = self.resnet.conv1(x)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)
        x_u_2_2 = self.conv3(x_u_2)
        resnet_up.append(x_u_2_2)

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)
        x_u_2_1 = self.conv2(x_u_1)
        resnet_up.append(x_u_2_1)
        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)
        x_u_2_0 = self.conv1(x_u)
        resnet_up.append(x_u_2_0)
        x, x_downsample = self.forward_features(x)
        ds_x, ds_x_downsample = self.ds_forward_features(g)
        # y=self.tif1(x_downsample[0],ds_x_downsample[0])
        fusion =[]
        tif_downsample =self.donv_up3(ds_x_downsample[0],x_downsample[0])
        # tif_downsample = self.ds_tif(x_downsample,ds_x_downsample)
        fusion.append(tif_downsample)
        tif_downsample = self.donv_up2(ds_x_downsample[1], x_downsample[1])
        # tif_downsample = self.ds_tif(x_downsample,ds_x_downsample)
        fusion.append(tif_downsample)
        tif_downsample = self.donv_up1(ds_x_downsample[2], x_downsample[2])
        # tif_downsample = self.ds_tif(x_downsample,ds_x_downsample)
        fusion.append(tif_downsample)

        # x_downsample[1]=tif_downsample[0]
        # x_downsample[2] = tif_downsample[1]
        # x, x_upsample = self.forward_up_features(x, x_downsample, resnet_up)
        x, x_upsample = self.forward_up_features(x, fusion, resnet_up)
        # x_c = self.up_c(x_u, self.conv1(x_upsample[0]))
        #
        # x_c_1_1 = self.up_c_1_1(x_u_1, self.conv2(x_upsample[1]))
        # x_c_1 = self.up_c_1_2(x_c, x_c_1_1)
        #
        # x_c_2_1 = self.up_c_2_1(x_u_2, self.conv3(x_upsample[2]))
        # x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)

        map_1 = self.up_x4(x)
        # map_2 = F.interpolate(self.final_x(x_u), scale_factor=16, mode='bilinear')
        # map_3 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear')

        # return map_1, map_2, map_3

        return map_1


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.fc1_c = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu_c = nn.ReLU(inplace=True)
        self.fc2_c = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid_c = nn.Sigmoid()

        # spatial attention for F_l
        # self.compress = ChannelPool()
        # self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        # self.compress_c = ChannelPool()
        # self.spatial_c = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        # self.W_g_c = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        # self.W_x_c = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        # self.W_c = Conv(ch_int, ch_int, 3, bn=True, relu=True)


        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        # self.residual_c = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.conv_out = nn.Sequential(
            nn.Conv2d(ch_out*3, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # W_g_c = self.W_g_c(g)
        # W_x_c = self.W_x_c(x)
        # bp_c = self.W_c(W_g_c * W_x_c)

        # spatial attention for cnn branch
        # g_in = g
        # g_c = g
        # g = self.compress(g)
        # g = self.spatial(g)
        # g = self.sigmoid(g) * g_in
        #
        # x_in = x
        # x_c = x
        # x = self.compress_c(x)
        # x = self.spatial_c(x)
        # x = self.sigmoid_c(x) * x_in
        #
        # fuse_s = self.residual(torch.cat([g, x, bp], 1))
        # g = g.mean((2, 3), keepdim=True)
        # g = self.fc1(g)
        # g = self.relu(g)
        # g = self.fc2(g)
        # g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x_c = x.mean((2, 3), keepdim=True)
        x_c = self.fc1_c(x_c)
        x_c = self.relu_c(x_c)
        x_c = self.fc2_c(x_c)
        x_c = self.sigmoid_c(x_c) * x_in

        g_in = g
        g_c = g.mean((2, 3), keepdim=True)
        g_c = self.fc1(g_c)
        g_c = self.relu(g_c)
        g_c = self.fc2(g_c)
        g_c = self.sigmoid(g_c) * g_in

        fuse = self.residual(torch.cat([g_c, x_c, bp], 1))
        # fuse = self.conv_out(torch.cat([g_c, x_c, bp], dim=1))

        return fuse


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))
