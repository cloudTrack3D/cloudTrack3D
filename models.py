from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def extract_fp_dict(out: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a combined DataParallel-friendly output dictionary into (clean_dict, fp_dict)."""
    clean = {
        "cloud_logit": out["cloud_logit"],
        "cloud_prob": out["cloud_prob"],
        "cloud_top": out["cloud_top"],
        "cloud_bot": out["cloud_bot"],
    }
    fp = {
        "cloud_logit": out["fp_cloud_logit"],
        "cloud_prob": out["fp_cloud_prob"],
        "cloud_top": out["fp_cloud_top"],
        "cloud_bot": out["fp_cloud_bot"],
    }
    return clean, fp


class RadiometricContextStem(nn.Module):
    """
    Computes fast GPU radiometric, temporal motion, and ERA5 thermodynamic context features:
      - Satellite-only:
          * `in_channels == 3` (1f IR): 7 channels
          * `in_channels == 4` (1f IR+VIS): 10 channels
          * `in_channels == 5` (3f IR): 12 channels
          * `in_channels == 8` (3f IR+VIS): 19 channels
      - ERA5-only (`in_channels == 20`: [T(9), RH(9), lat, lon]):
          * 20 + 6 thermodynamic profile summary channels -> 26 channels
      - Satellite + ERA5 (`in_channels in {21, 22, 23, 26}`: [sat_plus_coord(3|4|5|8), T(9), RH(9)]):
          * satellite stem channels (7|10|12|19) + 18 ERA5 channels + 6 ERA5 summary channels
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        if in_channels == 3:
            self.out_channels = 7
        elif in_channels == 4:
            self.out_channels = 10
        elif in_channels == 5:
            self.out_channels = 12
        elif in_channels == 8:
            self.out_channels = 19
        elif in_channels == 20:
            self.out_channels = 26
        elif in_channels in (21, 22, 23, 26):
            sat_ch = in_channels - 18
            sat_out = {3: 7, 4: 10, 5: 12, 8: 19}[sat_ch]
            self.out_channels = sat_out + 18 + 6
        else:
            self.out_channels = in_channels + 4

    @staticmethod
    def _era5_summary(era5_18: torch.Tensor) -> list[torch.Tensor]:
        temp = era5_18[:, 0:9]
        rh = era5_18[:, 9:18]
        rh_max = rh.max(dim=1, keepdim=True).values
        rh_mean = rh.mean(dim=1, keepdim=True)
        rh_upper = rh[:, 0:3].mean(dim=1, keepdim=True)
        rh_lower = rh[:, 6:9].mean(dim=1, keepdim=True)
        dt_low = temp[:, 8:9] - temp[:, 3:4]
        dt_high = temp[:, 3:4] - temp[:, 0:1]
        return [rh_max, rh_mean, rh_upper, rh_lower, dt_low, dt_high]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == 20:
            era5_extras = self._era5_summary(x[:, 0:18])
            return torch.cat([x] + era5_extras, dim=1)

        has_era5 = c in (21, 22, 23, 26)
        sat_ch = (c - 18) if has_era5 else c
        x_sat = x[:, :sat_ch]

        ir = x_sat[:, 0:1]
        mean_5 = F.avg_pool2d(F.pad(ir, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
        anom_5 = ir - mean_5
        # Warm clear-sky background estimate in 17x17 neighborhood (minimum IR brightness = warmest pixel)
        warm_bg = -F.max_pool2d(F.pad(-ir, (8, 8, 8, 8), mode="reflect"), kernel_size=17, stride=1)
        cold_diff = (ir - warm_bg).clamp(min=0.0)
        extras = [mean_5, anom_5, warm_bg, cold_diff]

        if sat_ch == 4:
            vis_0 = x_sat[:, 1:2]
            vis_mean_5 = F.avg_pool2d(F.pad(vis_0, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
            vis_anom_5 = vis_0 - vis_mean_5
            extras.extend([vis_mean_5, vis_anom_5])

        if sat_ch >= 5:
            ir_15 = x_sat[:, 1:2]
            ir_30 = x_sat[:, 2:3]
            extras.extend([ir - ir_15, ir_15 - ir_30, ir - ir_30])

        if sat_ch == 8:
            vis_0 = x_sat[:, 3:4]
            vis_15 = x_sat[:, 4:5]
            vis_30 = x_sat[:, 5:6]
            vis_mean_5 = F.avg_pool2d(F.pad(vis_0, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
            vis_anom_5 = vis_0 - vis_mean_5
            extras.extend([vis_mean_5, vis_anom_5, vis_0 - vis_15, vis_0 - vis_30])

        if has_era5:
            era5_extras = self._era5_summary(x[:, sat_ch : sat_ch + 18])
            return torch.cat([x] + extras + era5_extras, dim=1)

        return torch.cat([x] + extras, dim=1)


class ConvBlock(nn.Module):
    """Residual double-convolution block with BatchNorm and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class MultiScaleBottleneck(nn.Module):
    """Dilated multi-scale context module at the U-Net bottleneck."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        mid = ch // 4
        self.b1 = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(ch, mid, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(ch, mid, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(ch, mid, 3, padding=8, dilation=8, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True)
        )
        self.proj = nn.Sequential(nn.Conv2d(mid * 4, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return x + self.proj(out)


class MultiTaskCloudUNet(nn.Module):
    """
    Dense pixel-wise U-Net with radiometric context stem and direct skip connection to heads.
    Multi-GPU (DataParallel) compatible with `need_fp=True`.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32, fp_drop_rate: float = 0.5) -> None:
        super().__init__()
        self.rad_stem = RadiometricContextStem(in_channels=in_channels)
        stem_ch = self.rad_stem.out_channels

        self.enc1 = ConvBlock(stem_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = MultiScaleBottleneck(base_ch * 8)
        self.fp_dropout = nn.Dropout2d(fp_drop_rate)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.SiLU(inplace=True),
        )
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
        )
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
        )
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        head_in_ch = base_ch + stem_ch
        self.cls_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, kernel_size=1),
        )

    def _decode(
        self,
        x_rad: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        feat = torch.cat([d1, x_rad], dim=1)
        cloud_logit = self.cls_head(feat)
        cloud_prob = torch.sigmoid(cloud_logit)

        h_raw = torch.sigmoid(self.height_head(feat))
        cloud_top = h_raw[:, 0:1]
        cloud_bot = cloud_top * h_raw[:, 1:2]

        return {
            "cloud_logit": cloud_logit,
            "cloud_prob": cloud_prob,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
        }

    def forward(self, x: torch.Tensor, need_fp: bool = False) -> dict[str, torch.Tensor]:
        x_rad = self.rad_stem(x)
        e1 = self.enc1(x_rad)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(e4)

        out = self._decode(x_rad, e1, e2, e3, b)
        if not need_fp:
            return out

        out_fp = self._decode(
            x_rad,
            self.fp_dropout(e1),
            self.fp_dropout(e2),
            self.fp_dropout(e3),
            self.fp_dropout(b),
        )
        out["fp_cloud_logit"] = out_fp["cloud_logit"]
        out["fp_cloud_prob"] = out_fp["cloud_prob"]
        out["fp_cloud_top"] = out_fp["cloud_top"]
        out["fp_cloud_bot"] = out_fp["cloud_bot"]
        return out


# ==============================================================================
# SegViT Architecture (Vision Transformer + Attention-to-Mask ATM Decoder)
# ==============================================================================


class MixFFN(nn.Module):
    """SegFormer/SegViT Mix-FFN with 3x3 grouped convolution for position-aware local continuity."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 3.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=16, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        x_2d = x.transpose(1, 2).reshape(B, -1, H, W)
        x_2d = self.act(self.bn(self.dwconv(x_2d)))
        x = x_2d.flatten(2).transpose(1, 2)
        x = self.drop(self.fc2(x))
        return x


class ViTBlock(nn.Module):
    """Vision Transformer Encoder Block with Pre-Norm Multi-Head Self-Attention and MixFFN."""

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MixFFN(embed_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        xn = self.norm1(x)
        qkv = self.qkv(xn).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(attn_out)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class ATMBlock(nn.Module):
    """
    SegViT Attention-to-Mask (ATM) Decoder Block:
    Learnable task query tokens attend to spatial ViT token features (Z_l) via cross-attention.
    Produces both updated query embeddings and spatial attention masks (B, num_queries, H_t, W_t)
    weighted by the dynamically refined query embeddings.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_mask_q = nn.LayerNorm(embed_dim)
        self.mask_proj_q = nn.Linear(embed_dim, embed_dim)
        self.mask_proj_k = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(
        self, queries: torch.Tensor, spatial_tokens: torch.Tensor, H_t: int, W_t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = spatial_tokens.shape
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(spatial_tokens)

        q_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm, need_weights=False)
        queries = queries + q_out
        queries = queries + self.ffn(self.norm_ffn(queries))

        # Compute Attention-to-Mask (ATM) spatial similarity maps in FP32 to prevent FP16 overflow
        q_embed = self.mask_proj_q(self.norm_mask_q(queries)).float() * self.scale  # (B, K, C)
        k_embed = self.mask_proj_k(kv_norm).float()                                 # (B, N, C)
        sim_masks = torch.einsum("bkc,bnc->bkn", q_embed, k_embed).to(queries.dtype)
        sim_masks = sim_masks.reshape(B, -1, H_t, W_t)
        return queries, sim_masks


class MultiTaskSegViT(nn.Module):
    """
    SegViT (Segmentation Vision Transformer) with Multi-Layer Attention-to-Mask (ATM) Decoder
    and High-Resolution Convolutional Detail Fusion for dense pixel-wise cloud detection and
    cloud top/base height estimation.

    Architecture:
      1. RadiometricContextStem -> 7 physical input channels
      2. Hierarchical Local Detail Encoder at 1x (H, W), 1/2x (H/2, W/2), 1/4x (H/4, W/4)
      3. Patch Embedding at 1/8x (32x32 = 1024 tokens for 256x256 input) + 6 ViT Blocks
      4. SegViT ATM Decoder extracting multi-stage attention masks from layers [2, 4, 6]
         using K=32 learnable semantic/height task queries
      5. Progressive upsampling fusion with 1/4x, 1/2x, and 1x high-res detail maps + raw input skip
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 48,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        num_queries: int = 32,
        fp_drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.rad_stem = RadiometricContextStem(in_channels=in_channels)
        stem_ch = self.rad_stem.out_channels

        # High-resolution detail stages (1x, 1/2x, 1/4x)
        self.stem_1x = ConvBlock(stem_ch, base_ch)
        self.down_2x = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
            ConvBlock(base_ch * 2, base_ch * 2),
        )
        self.down_4x = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.SiLU(inplace=True),
            ConvBlock(base_ch * 4, base_ch * 4),
        )

        # Patch tokenization to 1/8x resolution for Vision Transformer
        self.patch_embed = nn.Conv2d(base_ch * 4, embed_dim, kernel_size=3, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, 32, 32) * 0.02)

        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.out_indices = (depth // 3 - 1, 2 * depth // 3 - 1, depth - 1)

        # SegViT Learnable Task Queries + Multi-Stage ATM Blocks
        self.task_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.atm_blocks = nn.ModuleList([ATMBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(3)])

        # Fuse the 3 ATM mask stages + top ViT spatial features at 1/8x resolution
        atm_total_ch = embed_dim + 3 * num_queries
        self.atm_proj = ConvBlock(atm_total_ch, base_ch * 4)

        self.fp_dropout = nn.Dropout2d(fp_drop_rate)

        # Progressive decoder fusing 1/8x ATM output with 1/4x, 1/2x, and 1x spatial detail
        self.up_4x = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.SiLU(inplace=True),
        )
        self.dec_4x = ConvBlock(base_ch * 8, base_ch * 2)

        self.up_2x = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
        )
        self.dec_2x = ConvBlock(base_ch * 4, base_ch)

        self.up_1x = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
        )
        self.dec_1x = ConvBlock(base_ch * 2, base_ch)

        head_in_ch = base_ch + stem_ch
        self.cls_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, kernel_size=1),
        )

    def _decode(
        self,
        x_rad: torch.Tensor,
        f_1x: torch.Tensor,
        f_2x: torch.Tensor,
        f_4x: torch.Tensor,
        vit_stages: list[torch.Tensor],
        H_t: int,
        W_t: int,
    ) -> dict[str, torch.Tensor]:
        B = x_rad.shape[0]
        queries = self.task_queries.expand(B, -1, -1)

        atm_masks = []
        for stage_feat, atm in zip(vit_stages, self.atm_blocks):
            queries, mask_s = atm(queries, stage_feat, H_t, W_t)
            atm_masks.append(mask_s)

        top_spatial = self.norm_out(vit_stages[-1]).transpose(1, 2).reshape(B, -1, H_t, W_t)
        atm_concat = torch.cat([top_spatial] + atm_masks, dim=1)
        f_8x = self.atm_proj(atm_concat)

        d_4x = self.dec_4x(torch.cat([self.up_4x(f_8x), f_4x], dim=1))
        d_2x = self.dec_2x(torch.cat([self.up_2x(d_4x), f_2x], dim=1))
        d_1x = self.dec_1x(torch.cat([self.up_1x(d_2x), f_1x], dim=1))

        feat = torch.cat([d_1x, x_rad], dim=1)
        cloud_logit = self.cls_head(feat)
        cloud_prob = torch.sigmoid(cloud_logit)

        h_raw = torch.sigmoid(self.height_head(feat))
        cloud_top = h_raw[:, 0:1]
        cloud_bot = cloud_top * h_raw[:, 1:2]

        return {
            "cloud_logit": cloud_logit,
            "cloud_prob": cloud_prob,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
        }

    def forward(self, x: torch.Tensor, need_fp: bool = False) -> dict[str, torch.Tensor]:
        x_rad = self.rad_stem(x)
        f_1x = self.stem_1x(x_rad)
        f_2x = self.down_2x(f_1x)
        f_4x = self.down_4x(f_2x)

        patch_feat = self.patch_embed(f_4x)
        B, C, H_t, W_t = patch_feat.shape
        if self.pos_embed.shape[2:] != (H_t, W_t):
            pos = F.interpolate(self.pos_embed, size=(H_t, W_t), mode="bilinear", align_corners=False)
        else:
            pos = self.pos_embed
        tokens = (patch_feat + pos).flatten(2).transpose(1, 2)

        vit_stages = []
        for idx, blk in enumerate(self.blocks):
            tokens = blk(tokens, H_t, W_t)
            if idx in self.out_indices:
                vit_stages.append(tokens)

        out = self._decode(x_rad, f_1x, f_2x, f_4x, vit_stages, H_t, W_t)
        if not need_fp:
            return out

        # Apply feature perturbation (spatial Dropout2d) to multi-scale features & ViT stages
        fp_stages = [
            self.fp_dropout(s.transpose(1, 2).reshape(B, C, H_t, W_t)).flatten(2).transpose(1, 2)
            for s in vit_stages
        ]
        out_fp = self._decode(
            x_rad,
            self.fp_dropout(f_1x),
            self.fp_dropout(f_2x),
            self.fp_dropout(f_4x),
            fp_stages,
            H_t,
            W_t,
        )
        out["fp_cloud_logit"] = out_fp["cloud_logit"]
        out["fp_cloud_prob"] = out_fp["cloud_prob"]
        out["fp_cloud_top"] = out_fp["cloud_top"]
        out["fp_cloud_bot"] = out_fp["cloud_bot"]
        return out


# ==============================================================================
# Pretrained Foundation Models: DINOv3 (ViT-B/16) & SAM (SAM-2 Hiera + Mask Query Decoder)
# ==============================================================================


class FoundationInputAdapter(nn.Module):
    """
    Adapts 1-frame (`in_channels=3`) or 3-frame (`in_channels=8`, IR 10.5 + VIS 0.6 at 0, -15, -30m)
    satellite inputs into a 3-channel tensor suitable for RGB-pretrained foundation backbones (DINOv3 / SAM).
    Starts as an exact physical 3-channel mapping at initialization (via zero-initialized residual conv)
    and learns optimal spatio-temporal channel mixing during fine-tuning.
    """

    def __init__(self, in_channels: int, stem_ch: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.adapter = nn.Sequential(
            nn.Conv2d(stem_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=True),
        )
        # Zero-init the final 1x1 projection so at step 0 the pretrained backbone sees clean imagery
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, x_rad: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == 20:
            # ERA5-only: map [max_RH, mean_RH, T_500] into [0, 1] RGB channels
            rh_max = x[:, 9:18].max(dim=1, keepdim=True).values
            rh_mean = x[:, 9:18].mean(dim=1, keepdim=True)
            t_500 = x[:, 3:4]
            base_rgb = torch.cat([rh_max, rh_mean, t_500], dim=1)
        else:
            sat_ch = (c - 18) if c in (21, 22, 23, 26) else c
            if sat_ch >= 8:
                # 3-frame IR + VIS: map [ir_t0, vis_t0, ir_t15] as base RGB channels
                base_rgb = torch.cat([x[:, 0:1], x[:, 3:4], x[:, 1:2]], dim=1)
            elif sat_ch >= 5:
                # 3-frame IR: map [ir_t0, ir_t15, ir_t30] as base RGB channels
                base_rgb = x[:, 0:3]
            elif sat_ch == 4:
                # 1-frame IR + VIS: map [ir_t0, vis_t0, ir_t0] as base RGB channels
                base_rgb = torch.cat([x[:, 0:1], x[:, 1:2], x[:, 0:1]], dim=1)
            else:
                # 1-frame IR: replicate ir_t0 across R, G, B
                base_rgb = x[:, 0:1].repeat(1, 3, 1, 1)

        rgb = (base_rgb + torch.tanh(self.adapter(x_rad)) * 0.25).clamp(0.0, 1.0)
        return (rgb - self.mean) / self.std


class MultiTaskDINOv3(nn.Module):
    """
    Pretrained Meta DINOv3 (ViT-Base/16 pretrained on LVD-1689M) fine-tuned for dense pixel-wise
    cloud detection and cloud top/base height estimation.
    Combines:
      - `FoundationInputAdapter` supporting both 1-frame still (`t=0`) and 3-frame (`t=0, -15, -30m`) inputs
      - Pretrained `vit_base_patch16_dinov3.lvd1689m` extracting multi-layer representations (layers 3, 7, 11)
      - DPT-style multi-stage feature pyramid + high-resolution local detail skip branch (`1x, 1/2x, 1/4x`)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 48,
        fp_drop_rate: float = 0.5,
        pretrained: bool = True,
        use_rad_skip: bool = True,
        enforce_ordering: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.use_rad_skip = use_rad_skip
        self.enforce_ordering = enforce_ordering
        self.rad_stem = RadiometricContextStem(in_channels=in_channels)
        stem_ch = self.rad_stem.out_channels
        self.input_adapter = FoundationInputAdapter(in_channels=in_channels, stem_ch=stem_ch)

        self.backbone = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=pretrained,
            features_only=True,
            img_size=256,
        )
        embed_dim = 768

        # High-resolution convolutional detail branch for sub-pixel lidar track alignment
        self.stem_1x = ConvBlock(stem_ch, base_ch)
        self.down_2x = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
            ConvBlock(base_ch * 2, base_ch * 2),
        )
        self.down_4x = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.SiLU(inplace=True),
            ConvBlock(base_ch * 4, base_ch * 4),
        )

        # Fuse the 3 multi-layer DINOv3 feature maps (3 * 768 -> base_ch * 4 at 16x16)
        self.dino_proj = nn.Sequential(
            nn.Conv2d(embed_dim * 3, base_ch * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.SiLU(inplace=True),
            ConvBlock(base_ch * 4, base_ch * 4),
        )

        self.fp_dropout = nn.Dropout2d(fp_drop_rate)

        # Progressive upsampling from 16x16 (1/16x) -> 64x64 (1/4x) -> 128x128 (1/2x) -> 256x256 (1x)
        self.up_4x = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.SiLU(inplace=True),
        )
        self.dec_4x = ConvBlock(base_ch * 8, base_ch * 2)

        self.up_2x = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
        )
        self.dec_2x = ConvBlock(base_ch * 4, base_ch)

        self.up_1x = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
        )
        self.dec_1x = ConvBlock(base_ch * 2, base_ch)

        head_in_ch = base_ch + stem_ch
        self.cls_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, kernel_size=1),
        )

    def _decode(
        self,
        x_rad: torch.Tensor,
        f_1x: torch.Tensor,
        f_2x: torch.Tensor,
        f_4x: torch.Tensor,
        f_dino: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.use_rad_skip:
            f_4x = torch.zeros_like(f_4x)
            f_2x = torch.zeros_like(f_2x)
            f_1x = torch.zeros_like(f_1x)
            x_rad = torch.zeros_like(x_rad)

        d_4x = self.dec_4x(torch.cat([self.up_4x(f_dino), f_4x], dim=1))
        d_2x = self.dec_2x(torch.cat([self.up_2x(d_4x), f_2x], dim=1))
        d_1x = self.dec_1x(torch.cat([self.up_1x(d_2x), f_1x], dim=1))

        feat = torch.cat([d_1x, x_rad], dim=1)
        cloud_logit = self.cls_head(feat)
        cloud_prob = torch.sigmoid(cloud_logit)

        h_raw = torch.sigmoid(self.height_head(feat))
        cloud_top = h_raw[:, 0:1]
        if self.enforce_ordering:
            cloud_bot = cloud_top * h_raw[:, 1:2]
        else:
            cloud_bot = h_raw[:, 1:2]
        return {
            "cloud_logit": cloud_logit,
            "cloud_prob": cloud_prob,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
        }

    def forward(self, x: torch.Tensor, need_fp: bool = False) -> dict[str, torch.Tensor]:
        x_rad = self.rad_stem(x)
        f_1x = self.stem_1x(x_rad)
        f_2x = self.down_2x(f_1x)
        f_4x = self.down_4x(f_2x)

        rgb_norm = self.input_adapter(x, x_rad)
        dino_feats = self.backbone(rgb_norm)
        f_dino = self.dino_proj(torch.cat(dino_feats, dim=1))

        out = self._decode(x_rad, f_1x, f_2x, f_4x, f_dino)
        if not need_fp:
            return out

        out_fp = self._decode(
            x_rad,
            self.fp_dropout(f_1x),
            self.fp_dropout(f_2x),
            self.fp_dropout(f_4x),
            self.fp_dropout(f_dino),
        )
        out["fp_cloud_logit"] = out_fp["cloud_logit"]
        out["fp_cloud_prob"] = out_fp["cloud_prob"]
        out["fp_cloud_top"] = out_fp["cloud_top"]
        out["fp_cloud_bot"] = out_fp["cloud_bot"]
        return out


class MultiTaskSAM(nn.Module):
    """
    Pretrained Meta SAM (Segment Anything Model 2 Hiera-Small pretrained on SA-1B + SA-V)
    with SAM Prompt/Query Cross-Attention Mask Decoder (`ATMBlock`) and multi-scale FPN fusion.
    Supports both 1-frame still (`t=0`) and 3-frame (`t=0, -15, -30m`) inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 48,
        num_queries: int = 32,
        fp_drop_rate: float = 0.5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.rad_stem = RadiometricContextStem(in_channels=in_channels)
        stem_ch = self.rad_stem.out_channels
        self.input_adapter = FoundationInputAdapter(in_channels=in_channels, stem_ch=stem_ch)

        # SAM-2 Hiera hierarchical backbone: outputs [64x64 (96), 32x32 (192), 16x16 (384), 8x8 (768)]
        self.backbone = timm.create_model(
            "sam2_hiera_small.fb_r896_2pt1",
            pretrained=pretrained,
            features_only=True,
        )

        self.stem_1x = ConvBlock(stem_ch, base_ch)
        self.down_2x = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
            ConvBlock(base_ch * 2, base_ch * 2),
        )

        # Project SAM multi-scale pyramid levels to unified decoder dimensions
        self.lat_32x = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True))
        self.lat_16x = nn.Sequential(nn.Conv2d(384, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True))
        self.lat_8x = nn.Sequential(nn.Conv2d(192, base_ch * 4, 1, bias=False), nn.BatchNorm2d(base_ch * 4), nn.SiLU(inplace=True))
        self.lat_4x = nn.Sequential(nn.Conv2d(96, base_ch * 2, 1, bias=False), nn.BatchNorm2d(base_ch * 2), nn.SiLU(inplace=True))

        # SAM Prompt/Mask-Token Cross-Attention Decoder at 16x16 resolution
        self.prompt_queries = nn.Parameter(torch.randn(1, num_queries, 256) * 0.02)
        self.sam_mask_decoder = ATMBlock(embed_dim=256, num_heads=8)
        self.sam_fuse_16x = ConvBlock(256 + num_queries, base_ch * 4)

        self.fp_dropout = nn.Dropout2d(fp_drop_rate)

        self.dec_8x = ConvBlock(base_ch * 8, base_ch * 2)
        self.dec_4x = ConvBlock(base_ch * 4, base_ch * 2)
        self.dec_2x = ConvBlock(base_ch * 4, base_ch)
        self.dec_1x = ConvBlock(base_ch * 2, base_ch)

        head_in_ch = base_ch + stem_ch
        self.cls_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(head_in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, kernel_size=1),
        )

    def _decode(
        self,
        x_rad: torch.Tensor,
        f_1x: torch.Tensor,
        f_2x: torch.Tensor,
        s_4x: torch.Tensor,
        s_8x: torch.Tensor,
        s_16x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, C, H16, W16 = s_16x.shape
        tokens_16x = s_16x.flatten(2).transpose(1, 2)
        queries = self.prompt_queries.expand(B, -1, -1)
        _, mask_maps = self.sam_mask_decoder(queries, tokens_16x, H16, W16)
        d_16x = self.sam_fuse_16x(torch.cat([s_16x, mask_maps], dim=1))

        up_8x = F.interpolate(d_16x, size=s_8x.shape[2:], mode="bilinear", align_corners=False)
        d_8x = self.dec_8x(torch.cat([up_8x, s_8x], dim=1))

        up_4x = F.interpolate(d_8x, size=s_4x.shape[2:], mode="bilinear", align_corners=False)
        d_4x = self.dec_4x(torch.cat([up_4x, s_4x], dim=1))

        up_2x = F.interpolate(d_4x, size=f_2x.shape[2:], mode="bilinear", align_corners=False)
        d_2x = self.dec_2x(torch.cat([up_2x, f_2x], dim=1))

        up_1x = F.interpolate(d_2x, size=f_1x.shape[2:], mode="bilinear", align_corners=False)
        d_1x = self.dec_1x(torch.cat([up_1x, f_1x], dim=1))

        feat = torch.cat([d_1x, x_rad], dim=1)
        cloud_logit = self.cls_head(feat)
        cloud_prob = torch.sigmoid(cloud_logit)

        h_raw = torch.sigmoid(self.height_head(feat))
        cloud_top = h_raw[:, 0:1]
        cloud_bot = cloud_top * h_raw[:, 1:2]
        return {
            "cloud_logit": cloud_logit,
            "cloud_prob": cloud_prob,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
        }

    def forward(self, x: torch.Tensor, need_fp: bool = False) -> dict[str, torch.Tensor]:
        x_rad = self.rad_stem(x)
        f_1x = self.stem_1x(x_rad)
        f_2x = self.down_2x(f_1x)

        rgb_norm = self.input_adapter(x, x_rad)
        c1, c2, c3, c4 = [f.contiguous() for f in self.backbone(rgb_norm)]

        s_4x = self.lat_4x(c1)
        s_8x = self.lat_8x(c2)
        s_32x_up = F.interpolate(self.lat_32x(c4), size=c3.shape[2:], mode="bilinear", align_corners=False)
        s_16x = (self.lat_16x(c3) + s_32x_up).contiguous()

        out = self._decode(x_rad, f_1x, f_2x, s_4x, s_8x, s_16x)
        if not need_fp:
            return out

        out_fp = self._decode(
            x_rad,
            self.fp_dropout(f_1x),
            self.fp_dropout(f_2x),
            self.fp_dropout(s_4x),
            self.fp_dropout(s_8x),
            self.fp_dropout(s_16x),
        )
        out["fp_cloud_logit"] = out_fp["cloud_logit"]
        out["fp_cloud_prob"] = out_fp["cloud_prob"]
        out["fp_cloud_top"] = out_fp["cloud_top"]
        out["fp_cloud_bot"] = out_fp["cloud_bot"]
        return out


def get_optimizer_param_groups(model: nn.Module, lr: float, weight_decay: float = 1e-4) -> list[dict]:
    """
    Use a 0.15x lower learning rate on pretrained foundation model backbones (`backbone.*`)
    so pretrained DINOv3 / SAM representations are preserved while adapters and decoders learn at `lr`.
    """
    base_model = unwrap_model(model)
    if not hasattr(base_model, "backbone"):
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]

    backbone_ids = {id(p) for p in base_model.backbone.parameters()}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids]
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    return [
        {"params": backbone_params, "lr": lr * 0.15, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr, "weight_decay": weight_decay},
    ]


def build_model(
    arch: str = "segvit",
    in_channels: int = 3,
    base_ch: int = 48,
    fp_drop_rate: float = 0.5,
    pretrained: bool = True,
) -> nn.Module:
    arch = arch.lower()
    if arch == "unet":
        return MultiTaskCloudUNet(in_channels=in_channels, base_ch=base_ch, fp_drop_rate=fp_drop_rate)
    elif arch == "segvit":
        return MultiTaskSegViT(
            in_channels=in_channels,
            base_ch=base_ch,
            embed_dim=256,
            depth=6,
            num_heads=8,
            num_queries=32,
            fp_drop_rate=fp_drop_rate,
        )
    elif arch == "dinov3":
        return MultiTaskDINOv3(
            in_channels=in_channels,
            base_ch=base_ch,
            fp_drop_rate=fp_drop_rate,
            pretrained=pretrained,
        )
    elif arch == "sam":
        return MultiTaskSAM(
            in_channels=in_channels,
            base_ch=base_ch,
            fp_drop_rate=fp_drop_rate,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from ['unet', 'segvit', 'dinov3', 'sam'].")


class EMATeacher(nn.Module):
    """
    Exponential Moving Average (EMA) Teacher model for Semi-Supervised Learning.
    Handles unwrapped inner models cleanly when Student is wrapped in nn.DataParallel.
    """

    def __init__(self, student: nn.Module, decay: float = 0.996) -> None:
        super().__init__()
        self.decay = decay
        base_student = unwrap_model(student)
        self.model = copy.deepcopy(base_student)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module, step: int | None = None) -> None:
        base_student = unwrap_model(student)
        decay = self.decay
        if step is not None:
            decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for t_param, s_param in zip(self.model.parameters(), base_student.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)
        for t_buf, s_buf in zip(self.model.buffers(), base_student.buffers()):
            t_buf.data.copy_(s_buf.data)
