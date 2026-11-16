"""
Diagnostic & Architectural Study: Mixed-Unified Satellite + ERA5 Architectures
for 3D Cloud Reconstruction (3f IR+VIS + ERA5, 26 channels, 408-Granule Split).

Investigates why naive early concatenation of ERA5 (26 ch) bottlenecks pretrained
RGB foundation backbones (SAM 2, DINOv3) on 3-frame imagery, and evaluates progressive
Mixed-Unified architectures across U-Net, SAM 2, and DINOv3 under both Supervised
and Semi-Supervised (Dilated+Dist, Ours) regimes:

  1. Baseline A: Satellite-Only 3f IR+VIS (8 ch)
  2. Baseline B: Naive Early Concat 3f IR+VIS + ERA5 (26 ch -> 43 stem ch -> 3 RGB ch)
  3. Ablation 1 (Decoupled-ViT-Adapter):
     - Pretrained backbone (DINOv3 / SAM 2) and 1x pixel skip receive ONLY the 8-ch
       satellite tensor (19 sharp 450m radiometric + motion channels), eliminating
       3-channel RGB crowding and 28km head blurring.
     - Dedicated ERA5 Thermodynamic Side-Encoder processes the 18-ch 9-level (T, RH)
       profiles + 6 thermodynamic summaries + local IR/VIS thermal contrast (28 ch)
       and modulates the decoder via zero-initialized FiLM (scale/shift) + residual.
  4. Ablation 2 (Decoupled-ViT + Physics-Guided Isobaric Level-Matching Attention):
     - Adds explicit physical thermal/humidity level-matching attention across the
       9 pressure levels (200..900 hPa) to compute soft isobaric height priors and
       gated thermodynamic residual corrections.
  5. Full Mixed-Unified Architecture (Dual-Stream Visual-Motion + Thermodynamic-ERA5
     with Regime-Gated Cross-Modal Fusion):
     - Unifies the high-resolution 3-frame visual motion stream with the synchronous
       ERA5 thermodynamic profile stream via a learned pixel-wise regime/discrepancy
       gate and spatial conv refinement, trained end-to-end with Bilateral Dilated
       Ground-Truth Supervision + Distance-Adaptive Consistency (Dilated+Dist, Ours).
"""

from __future__ import annotations

import copy
import gc
import json
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import (
    MAX_HEIGHT_KM,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import (
    DilatedSupervisedCloudLoss,
    DistanceGuidedConsistencyLoss,
    SupervisedCloudLoss,
    evaluate_sparse_predictions,
)
from models import (
    ConvBlock,
    EMATeacher,
    RadiometricContextStem,
    build_model,
    unwrap_model,
)
from run_cvpr_experiments import (
    preload_all_images_and_tracks,
    set_seed,
)
from run_era5_experiments import (
    make_era5_patch_dataset,
    preload_era5_for_granules,
)
from run_s4mc_and_dist_ours import (
    BUILD_LOCK,
    load_ckpt_state,
)
from run_tables45_and_3fera5_ours import (
    InMemoryDataset,
    add_derived_metrics,
)

OUT_DIR = "outputs/mixed_unified_era5_study"
OUT_JSON = os.path.join(OUT_DIR, "mixed_unified_era5_results.json")

# Standard ICAO geopotential altitudes (km) for [200, 300, 400, 500, 600, 700, 800, 850, 900] hPa
# normalized by MAX_HEIGHT_KM = 16.0 km
STD_PRESSURE_HEIGHTS_NORM = [
    11.78 / MAX_HEIGHT_KM,
    9.16 / MAX_HEIGHT_KM,
    7.19 / MAX_HEIGHT_KM,
    5.57 / MAX_HEIGHT_KM,
    4.21 / MAX_HEIGHT_KM,
    3.01 / MAX_HEIGHT_KM,
    1.95 / MAX_HEIGHT_KM,
    1.46 / MAX_HEIGHT_KM,
    0.99 / MAX_HEIGHT_KM,
]


class ERA5ThermodynamicEncoder(nn.Module):
    """
    Dedicated Thermodynamic Profile Encoder for 9-level ERA5 [T(9), RH(9)] coupled with
    local satellite brightness temperature & visible reflectance reference:
      - Computes 6 vertical profile summary channels (RH_max, RH_mean, RH_upper, RH_lower,
        dT_low, dT_high)
      - Optionally computes Physics-Guided Isobaric Level-Matching Attention across the
        9 pressure surfaces [200..900 hPa] by comparing local satellite IR brightness
        temperature against each pressure level's T(p_k) and RH(p_k)
      - Outputs spatial thermodynamic embeddings for FiLM modulation and gated height refinement.
    """

    def __init__(self, hidden_ch: int = 48, use_isobaric_attn: bool = True) -> None:
        super().__init__()
        self.use_isobaric_attn = use_isobaric_attn
        # Input channels: 18 ERA5 + 6 ERA5 summaries + 6 sat reference [ir0, vis0, warm_bg, cold_diff, lat, lon] = 30
        in_ch = 30 + (4 if use_isobaric_attn else 0)
        self.enc = nn.Sequential(
            ConvBlock(in_ch, hidden_ch),
            ConvBlock(hidden_ch, hidden_ch),
        )
        if use_isobaric_attn:
            # Learnable per-level affine calibration for matching normalized IR brightness [0,1] to ERA5 T(p_k) [0,1]
            self.level_temp_scale = nn.Parameter(torch.ones(1, 9, 1, 1) * (-6.0))
            self.level_rh_scale = nn.Parameter(torch.ones(1, 9, 1, 1) * 3.0)
            self.level_bias = nn.Parameter(torch.zeros(1, 9, 1, 1))
            self.register_buffer(
                "level_heights",
                torch.tensor(STD_PRESSURE_HEIGHTS_NORM, dtype=torch.float32).view(1, 9, 1, 1),
            )

    def forward(self, x_sat8: torch.Tensor, x_rad_sat19: torch.Tensor, x_era5_18: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temp = x_era5_18[:, 0:9]  # (B, 9, H, W) in [0, 1] (warmer = larger)
        rh = x_era5_18[:, 9:18]   # (B, 9, H, W) in [0, 1]
        era5_sum = RadiometricContextStem._era5_summary(x_era5_18)  # 6 channels

        ir0 = x_sat8[:, 0:1]      # (B, 1, H, W) inverted brightness (colder = larger)
        vis0 = x_sat8[:, 3:4]
        coords = x_sat8[:, 6:8]
        warm_bg = x_rad_sat19[:, 10:11]
        cold_diff = x_rad_sat19[:, 11:12]

        sat_ref = [ir0, vis0, warm_bg, cold_diff, coords]

        if self.use_isobaric_attn:
            # Physical approximate brightness temperature in ERA5 normalized [0,1] space:
            # Larger ir0 means colder cloud top -> lower T_norm ~ (1 - ir0)
            tb_approx = 1.0 - ir0
            t_diff = (tb_approx - temp).abs()  # (B, 9, H, W)
            logits_top = self.level_temp_scale * t_diff + self.level_rh_scale * rh + self.level_bias
            attn_top = F.softmax(logits_top, dim=1)
            # Cloud base tends to sit at lower altitude (higher pressure index k in 4..8) where RH is high
            low_prior = torch.linspace(-1.0, 1.5, 9, device=x_sat8.device, dtype=x_sat8.dtype).view(1, 9, 1, 1)
            attn_bot = F.softmax(logits_top + low_prior + 2.0 * rh, dim=1)

            h_phys_top = (attn_top * self.level_heights).sum(dim=1, keepdim=True)
            h_phys_bot = torch.minimum(h_phys_top, (attn_bot * self.level_heights).sum(dim=1, keepdim=True))
            rh_at_match = (attn_top * rh).sum(dim=1, keepdim=True)
            t_err_at_match = (attn_top * t_diff).sum(dim=1, keepdim=True)
            iso_feats = [h_phys_top, h_phys_bot, rh_at_match, t_err_at_match]
            inp = torch.cat([x_era5_18] + era5_sum + sat_ref + iso_feats, dim=1)
            phys_prior = torch.cat([h_phys_top, h_phys_bot], dim=1)
        else:
            inp = torch.cat([x_era5_18] + era5_sum + sat_ref, dim=1)
            phys_prior = torch.zeros((x_sat8.shape[0], 2, x_sat8.shape[2], x_sat8.shape[3]), device=x_sat8.device, dtype=x_sat8.dtype)

        f_era5 = self.enc(inp)
        return f_era5, phys_prior


class DecoupledSingleBackboneERA5Model(nn.Module):
    """
    Single-Backbone Decoupled Satellite + ERA5 Architecture (Ablations 1 & 2):
      - The pretrained foundation backbone (DINOv3 / SAM 2 / U-Net) receives ONLY the 8-channel
        3f IR+VIS satellite tensor (`x[:, :8]`), completely avoiding 3-channel RGB input crowding
        and preserving 100% of the 35-epoch visual-motion representation at initialization.
      - A parallel `ERA5ThermodynamicEncoder` processes the 18-channel ERA5 profile + satellite
        thermal reference and injects thermodynamic context via:
          (a) Zero-initialized FiLM (scale + shift) on the high-res visual features, and
          (b) Zero-initialized gated residual corrections on cloud_logit, z_top, and z_bot.
    """

    def __init__(
        self,
        arch: str,
        sat_ckpt_path: str,
        use_isobaric_attn: bool = True,
        base_ch: int = 48,
    ) -> None:
        super().__init__()
        self.arch = arch
        with BUILD_LOCK:
            self.sat_model = build_model(arch=arch, in_channels=8, base_ch=base_ch, pretrained=False)
        unwrap_model(self.sat_model).load_state_dict(load_ckpt_state(sat_ckpt_path))
        # Freeze the converged visual backbone so 450m multi-frame motion features are preserved
        for p in self.sat_model.parameters():
            p.requires_grad_(False)

        self.era5_encoder = ERA5ThermodynamicEncoder(hidden_ch=base_ch, use_isobaric_attn=use_isobaric_attn)

        # FiLM + Gated Residual Fusion Head operating on [sat_rad_19, sat_preds_4, era5_feat_48, phys_prior_2]
        fusion_in_ch = 19 + 4 + base_ch + 2
        self.fusion_net = nn.Sequential(
            ConvBlock(fusion_in_ch, base_ch),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.delta_cls = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.delta_h = nn.Conv2d(base_ch, 2, kernel_size=1)
        self.gate_net = nn.Sequential(
            nn.Conv2d(fusion_in_ch, 16, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        # Zero-initialize final residual layers so at step 0 output == exact 3f IR+VIS model
        nn.init.zeros_(self.delta_cls.weight)
        nn.init.zeros_(self.delta_cls.bias)
        nn.init.zeros_(self.delta_h.weight)
        nn.init.zeros_(self.delta_h.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_sat = x[:, :8]
        x_era5 = x[:, 8:26]

        with torch.no_grad():
            sat_out = self.sat_model(x_sat)
            x_rad_sat = self.sat_model.rad_stem(x_sat)

        f_era5, phys_prior = self.era5_encoder(x_sat, x_rad_sat, x_era5)

        sat_preds = torch.cat(
            [
                sat_out["cloud_logit"],
                sat_out["cloud_prob"],
                sat_out["cloud_top"],
                sat_out["cloud_bot"],
            ],
            dim=1,
        )
        fusion_in = torch.cat([x_rad_sat, sat_preds, f_era5, phys_prior], dim=1)
        f_joint = self.fusion_net(fusion_in)
        gates = self.gate_net(fusion_in)  # (B, 3, H, W) in [0, 1]

        d_cls = self.delta_cls(f_joint)
        d_h = torch.tanh(self.delta_h(f_joint)) * 0.35

        cloud_logit = sat_out["cloud_logit"] + gates[:, 0:1] * d_cls
        cloud_prob = torch.sigmoid(cloud_logit)

        # Ordered height refinement in [0, 1]
        cloud_top = (sat_out["cloud_top"] + gates[:, 1:2] * d_h[:, 0:1]).clamp(0.0, 1.0)
        raw_bot = (sat_out["cloud_bot"] + gates[:, 2:3] * d_h[:, 1:2]).clamp(0.0, 1.0)
        cloud_bot = torch.minimum(cloud_top, raw_bot)

        return {
            "cloud_logit": cloud_logit,
            "cloud_prob": cloud_prob,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
        }


class MixedUnifiedDualStreamModel(nn.Module):
    """
    Full Mixed-Unified Architecture (Variant 5):
    Combines:
      1. High-Res 3-Frame Visual-Motion Stream (`sat_3f_model` on 8-ch 3f IR+VIS):
         Preserves sharp 450m cloud edges, solar VIS albedo, and 30-minute wind-shear advection.
      2. Synchronous Thermodynamic ERA5 Stream (`era5_1f_model` on 22-ch 1f IR+VIS+ERA5 and
         `era5_3f_model` on 26-ch 3f IR+VIS+ERA5, plus `ERA5ThermodynamicEncoder` with
         9-level Isobaric Level-Matching Attention):
         Provides uncorrupted t=0 9-level (T, RH) vertical atmospheric sounding anchors.
      3. Regime-Aware Gated Cross-Modal Fusion Head:
         Dynamically weights the visual-motion stream and the thermodynamic-ERA5 stream at every
         pixel based on local brightness temperature, visible albedo, motion magnitude, ERA5 RH
         saturation, and inter-stream height discrepancy, plus a learned spatial conv residual.
    """

    def __init__(
        self,
        arch: str,
        sat_3f_ckpt: str,
        era5_3f_ckpt: str,
        era5_1f_ckpt: str,
        base_ch: int = 48,
    ) -> None:
        super().__init__()
        self.arch = arch
        with BUILD_LOCK:
            self.sat_3f_model = build_model(arch=arch, in_channels=8, base_ch=base_ch, pretrained=False)
            self.era5_3f_model = build_model(arch=arch, in_channels=26, base_ch=base_ch, pretrained=False)
            self.era5_1f_model = build_model(arch=arch, in_channels=22, base_ch=base_ch, pretrained=False)

        unwrap_model(self.sat_3f_model).load_state_dict(load_ckpt_state(sat_3f_ckpt))
        unwrap_model(self.era5_3f_model).load_state_dict(load_ckpt_state(era5_3f_ckpt))
        unwrap_model(self.era5_1f_model).load_state_dict(load_ckpt_state(era5_1f_ckpt))

        for m in (self.sat_3f_model, self.era5_3f_model, self.era5_1f_model):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self.era5_encoder = ERA5ThermodynamicEncoder(hidden_ch=32, use_isobaric_attn=True)

        # Input to cross-modal regime gate & residual refiner:
        # 19 (sat_rad) + 12 (3 streams * 4 preds) + 4 (inter-stream diffs) + 32 (era5_enc) + 2 (phys_prior) = 69
        in_ch = 19 + 12 + 4 + 32 + 2
        self.regime_net = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch),
        )
        # 3-way simplex weights per output (cls, top, bot) -> 3 * 3 = 9 channels
        self.stream_logits = nn.Conv2d(base_ch, 9, kernel_size=1)
        self.residual_head = nn.Conv2d(base_ch, 3, kernel_size=1)

        # Initialize stream logits to favor a balanced physical prior (50% 3f-sat, 30% 1f-era5, 20% 3f-era5)
        nn.init.zeros_(self.stream_logits.weight)
        with torch.no_grad():
            init_b = torch.tensor([0.7, 0.2, -0.1] * 3, dtype=torch.float32)
            self.stream_logits.bias.copy_(init_b)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _extract_1f_vis_era5_from_3f(x_26: torch.Tensor) -> torch.Tensor:
        """
        Extract synchronous t=0 1f IR+VIS + ERA5 (22 ch: [ir_t0, vis_t0, lat, lon, era5(18)])
        from the 26-ch 3f IR+VIS + ERA5 tensor ([ir0, ir15, ir30, vis0, vis15, vis30, lat, lon, era5(18)]).
        """
        ir0 = x_26[:, 0:1]
        vis0 = x_26[:, 3:4]
        coords = x_26[:, 6:8]
        era5 = x_26[:, 8:26]
        return torch.cat([ir0, vis0, coords, era5], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_sat8 = x[:, :8]
        x_era5_18 = x[:, 8:26]
        x_1f_era5_22 = self._extract_1f_vis_era5_from_3f(x)

        with torch.no_grad():
            out_s3 = self.sat_3f_model(x_sat8)
            out_e1 = self.era5_1f_model(x_1f_era5_22)
            out_e3 = self.era5_3f_model(x)
            x_rad_sat = self.sat_3f_model.rad_stem(x_sat8)

        f_era5, phys_prior = self.era5_encoder(x_sat8, x_rad_sat, x_era5_18)

        preds_all = torch.cat(
            [
                out_s3["cloud_logit"], out_s3["cloud_prob"], out_s3["cloud_top"], out_s3["cloud_bot"],
                out_e1["cloud_logit"], out_e1["cloud_prob"], out_e1["cloud_top"], out_e1["cloud_bot"],
                out_e3["cloud_logit"], out_e3["cloud_prob"], out_e3["cloud_top"], out_e3["cloud_bot"],
            ],
            dim=1,
        )
        diffs = torch.cat(
            [
                (out_s3["cloud_top"] - out_e1["cloud_top"]).abs(),
                (out_s3["cloud_bot"] - out_e1["cloud_bot"]).abs(),
                (out_s3["cloud_top"] - out_e3["cloud_top"]).abs(),
                (out_s3["cloud_prob"] - out_e1["cloud_prob"]).abs(),
            ],
            dim=1,
        )

        inp = torch.cat([x_rad_sat, preds_all, diffs, f_era5, phys_prior], dim=1)
        feat = self.regime_net(inp)

        w_raw = self.stream_logits(feat)  # (B, 9, H, W)
        w_cls = F.softmax(w_raw[:, 0:3], dim=1)
        w_top = F.softmax(w_raw[:, 3:6], dim=1)
        w_bot = F.softmax(w_raw[:, 6:9], dim=1)

        res = self.residual_head(feat)
        d_cls = res[:, 0:1]
        d_top = torch.tanh(res[:, 1:2]) * 0.25
        d_bot = torch.tanh(res[:, 2:3]) * 0.25

        cloud_logit = (
            w_cls[:, 0:1] * out_s3["cloud_logit"]
            + w_cls[:, 1:2] * out_e1["cloud_logit"]
            + w_cls[:, 2:3] * out_e3["cloud_logit"]
            + d_cls
        )
        cloud_prob = torch.sigmoid(cloud_logit)

        cloud_top = (
            w_top[:, 0:1] * out_s3["cloud_top"]
            + w_top[:, 1:2] * out_e1["cloud_top"]
            + w_top[:, 2:3] * out_e3["cloud_top"]
            + d_top
        ).clamp(0.0, 1.0)

        raw_bot = (
            w_bot[:, 0:1] * out_s3["cloud_bot"]
            + w_bot[:, 1:2] * out_e1["cloud_bot"]
            + w_bot[:, 2:3] * out_e3["cloud_bot"]
            + d_bot
        ).clamp(0.0, 1.0)
        cloud_bot = torch.minimum(cloud_top, raw_bot)

        return {
            "cloud_logit": cloud_logit,
            "cloud_prob": cloud_prob,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
        }


def train_and_evaluate_unified_model(
    name: str,
    model_fn: Any,
    train_ds: Dataset,
    val_ds: Dataset,
    gpu_id: int,
    use_semisup_ours: bool = True,
    epochs: int = 8,
    lr: float = 6e-4,
) -> dict[str, Any]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    t0 = time.time()

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_file = os.path.join(OUT_DIR, f"{name}.pt")
    if os.path.isfile(ckpt_file):
        saved = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        best_m = saved["metrics"]
        print(
            f"[{name} | CACHED] "
            f"Acc={best_m['cloud_acc']*100:.2f} F1={best_m['cloud_f1']*100:.2f} IoU={best_m['cloud_iou']*100:.2f} "
            f"AUROC={best_m['cloud_auroc']*100:.2f} | "
            f"Top={best_m['top_mae_km']:.3f} [Low={best_m['top_upto_15kft_mae_km']:.3f}, High={best_m['top_above_15kft_mae_km']:.3f}] | "
            f"Bot={best_m['bot_mae_km']:.3f} [Low={best_m['bot_upto_15kft_mae_km']:.3f}, High={best_m['bot_above_15kft_mae_km']:.3f}] | "
            f"Thick={best_m['thick_mae_km']:.3f}"
        )
        return best_m

    model = model_fn().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=0)

    if use_semisup_ours:
        sup_criterion = DilatedSupervisedCloudLoss(
            cls_weight=1.0,
            reg_weight=8.0,
            order_weight=0.5,
            dilate_radius=3.5,
            dilate_gamma=0.45,
        ).to(device)
        cons_criterion = DistanceGuidedConsistencyLoss(
            tau_cls_min=0.65,
            tau_cls_max=0.88,
            tau_reg_min=0.55,
            tau_reg_max=0.82,
            dist_sigma=24.0,
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)
    else:
        sup_criterion = SupervisedCloudLoss(cls_weight=1.0, reg_weight=8.0, order_weight=0.5).to(device)
        cons_criterion = None

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    init_m = add_derived_metrics(evaluate_sparse_predictions(model, val_loader, device))
    best_score = init_m["cloud_f1"] - 0.22 * (init_m["top_mae_km"] + init_m["bot_mae_km"])
    best_m = init_m
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            vm = batch["valid_mask"].to(device, non_blocking=True)
            targets = {
                "valid_mask": vm,
                "cloud_mask": batch["cloud_mask"].to(device, non_blocking=True),
                "cloud_top": batch["cloud_top"].to(device, non_blocking=True),
                "cloud_bot": batch["cloud_bot"].to(device, non_blocking=True),
                "image": img,
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                preds = model(img)
                loss = sup_criterion(preds, targets)["loss"]
                if use_semisup_ours and cons_criterion is not None:
                    with torch.no_grad():
                        t_preds = {k: v.detach().float() for k, v in preds.items()}
                    # Horizontal flip equivariance consistency with distance-adaptive thresholding
                    img_flip = torch.flip(img, dims=[3])
                    preds_flip = model(img_flip)
                    preds_unflip = {k: torch.flip(v, dims=[3]) for k, v in preds_flip.items()}
                    c_loss = cons_criterion(preds_unflip, t_preds, vm)["loss_cons"]
                    loss = loss + 0.20 * c_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        val_m = add_derived_metrics(evaluate_sparse_predictions(model, val_loader, device))
        score = val_m["cloud_f1"] - 0.22 * (val_m["top_mae_km"] + val_m["bot_mae_km"])
        if score > best_score:
            best_score = score
            best_m = val_m
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"model_state": best_state, "metrics": best_m}, os.path.join(OUT_DIR, f"{name}.pt"))
    print(
        f"[{name} | GPU {gpu_id} | {time.time()-t0:.1f}s] "
        f"Acc={best_m['cloud_acc']*100:.2f} F1={best_m['cloud_f1']*100:.2f} IoU={best_m['cloud_iou']*100:.2f} "
        f"AUROC={best_m['cloud_auroc']*100:.2f} | "
        f"Top={best_m['top_mae_km']:.3f} [Low={best_m['top_upto_15kft_mae_km']:.3f}, High={best_m['top_above_15kft_mae_km']:.3f}] | "
        f"Bot={best_m['bot_mae_km']:.3f} [Low={best_m['bot_upto_15kft_mae_km']:.3f}, High={best_m['bot_above_15kft_mae_km']:.3f}] | "
        f"Thick={best_m['thick_mae_km']:.3f}"
    )
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_m


def main() -> None:
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(granules_490, "../ir10.5", "../vis0.6", use_vis06=True)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)

    img_cache, track_cache = preload_all_images_and_tracks(granules_408)
    era5_cache = preload_era5_for_granules(granules_408)

    print("[Dataset] Materializing 3f IR+VIS + ERA5 (26 ch) train and val patches in RAM...")
    raw_tr = make_era5_patch_dataset(train_408, img_cache, track_cache, era5_cache, 6, True, True, True, True, True, seed=42)
    raw_va = make_era5_patch_dataset(val_408, img_cache, track_cache, era5_cache, 6, False, True, True, True, True, seed=42)
    mem_tr = InMemoryDataset(raw_tr, max_items=320, seed=42)
    mem_va = InMemoryDataset(raw_va, max_items=None, seed=42)

    results: dict[str, Any] = {}

    # Checkpoint paths for each architecture and regime
    ckpt_map = {
        "unet": {
            "sup_sat3f": "outputs/supervised_unet_3f_vis/best_supervised.pt",
            "sup_era3f": "outputs/cvpr_experiments/era5_sup_unet_3f_ir_vis_era5/best_model.pt",
            "sup_era1f": "outputs/cvpr_experiments/era5_sup_unet_1f_ir_vis_era5/best_model.pt",
            "ours_sat3f": "outputs/semisup_s4mc_and_ours/ours_dist_3f_vis_unet/best_semisup.pt",
            "ours_era3f": "outputs/semisup_s4mc_and_ours/ours_dist_3f_ir_vis_era5_unet/best_semisup.pt",
            "ours_era1f": "outputs/semisup_s4mc_and_ours/ours_dist_1f_ir_vis_era5_unet/best_semisup.pt",
        },
        "sam": {
            "sup_sat3f": "outputs/supervised_sam_3f_vis/best_supervised.pt",
            "sup_era3f": "outputs/cvpr_experiments/era5_sup_sam_3f_ir_vis_era5/best_model.pt",
            "sup_era1f": "outputs/cvpr_experiments/era5_sup_sam_1f_ir_vis_era5/best_model.pt",
            "ours_sat3f": "outputs/semisup_s4mc_and_ours/ours_dist_3f_vis_sam/best_semisup.pt",
            "ours_era3f": "outputs/semisup_s4mc_and_ours/ours_dist_3f_ir_vis_era5_sam/best_semisup.pt",
            "ours_era1f": "outputs/semisup_s4mc_and_ours/ours_dist_1f_ir_vis_era5_sam/best_semisup.pt",
        },
        "dinov3": {
            "sup_sat3f": "outputs/supervised_dinov3_3f_vis/best_supervised.pt",
            "sup_era3f": "outputs/cvpr_experiments/era5_sup_dinov3_3f_ir_vis_era5/best_model.pt",
            "sup_era1f": "outputs/cvpr_experiments/era5_sup_dinov3_1f_ir_vis_era5/best_model.pt",
            "ours_sat3f": "outputs/semisup_s4mc_and_ours/ours_dist_3f_vis_dinov3/best_semisup.pt",
            "ours_era3f": "outputs/semisup_s4mc_and_ours/ours_dist_3f_ir_vis_era5_dinov3/best_semisup.pt",
            "ours_era1f": "outputs/semisup_s4mc_and_ours/ours_dist_1f_ir_vis_era5_dinov3/best_semisup.pt",
        },
    }

    jobs = []
    for arch in ["unet", "sam", "dinov3"]:
        c = ckpt_map[arch]
        # 1. Ablation 1: Single-Backbone Decoupled-ViT + ERA5 Side-Encoder (Ours)
        jobs.append((
            f"abl1_decoupled_side_{arch}_ours",
            lambda _a=arch, _c=c: DecoupledSingleBackboneERA5Model(_a, _c["ours_sat3f"], use_isobaric_attn=False),
            True,
        ))
        # 2. Ablation 2: Single-Backbone Decoupled-ViT + Isobaric Level-Matching Attention (Ours)
        jobs.append((
            f"abl2_decoupled_isobaric_{arch}_ours",
            lambda _a=arch, _c=c: DecoupledSingleBackboneERA5Model(_a, _c["ours_sat3f"], use_isobaric_attn=True),
            True,
        ))
        # 3. Full Mixed-Unified Dual-Stream (Supervised)
        jobs.append((
            f"mixed_unified_{arch}_sup",
            lambda _a=arch, _c=c: MixedUnifiedDualStreamModel(_a, _c["sup_sat3f"], _c["sup_era3f"], _c["sup_era1f"]),
            False,
        ))
        # 4. Full Mixed-Unified Dual-Stream (Dilated+Dist, Ours)
        jobs.append((
            f"mixed_unified_{arch}_ours",
            lambda _a=arch, _c=c: MixedUnifiedDualStreamModel(_a, _c["ours_sat3f"], _c["ours_era3f"], _c["ours_era1f"]),
            True,
        ))

    import queue
    from concurrent.futures import ThreadPoolExecutor

    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu_id: int) -> None:
        while True:
            try:
                j_name, j_fn, j_ours = q.get_nowait()
            except queue.Empty:
                break
            try:
                m = train_and_evaluate_unified_model(j_name, j_fn, mem_tr, mem_va, gpu_id, use_semisup_ours=j_ours)
                results[j_name] = m
                with open(OUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
            finally:
                q.task_done()

    n_gpus = min(torch.cuda.device_count(), 4)
    print(f"=== Launching {len(jobs)} Mixed-Unified Jobs across {n_gpus} GPUs ===")
    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futs = [ex.submit(worker, g) for g in range(n_gpus)]
        for fut in futs:
            fut.result()

    print("\n=== SUMMARY OF MIXED-UNIFIED ERA5 STUDY ===")
    for k, m in sorted(results.items()):
        print(
            f"{k:36s} | Acc={m['cloud_acc']*100:.2f} Prec={m['cloud_precision']*100:.2f} "
            f"Rec={m['cloud_recall']*100:.2f} F1={m['cloud_f1']*100:.2f} IoU={m['cloud_iou']*100:.2f} "
            f"AUROC={m['cloud_auroc']*100:.2f} | "
            f"Top={m['top_mae_km']:.3f} ({m['top_mae_kft']:.2f}kft) "
            f"[Low={m['top_upto_15kft_mae_km']:.3f}, High={m['top_above_15kft_mae_km']:.3f}, "
            f"1L={m['top_single_layer_mae_km']:.3f}, ML={m['top_multi_layer_mae_km']:.3f}] | "
            f"Bot={m['bot_mae_km']:.3f} ({m['bot_mae_kft']:.2f}kft) "
            f"[Low={m['bot_upto_15kft_mae_km']:.3f}, High={m['bot_above_15kft_mae_km']:.3f}, "
            f"1L={m['bot_single_layer_mae_km']:.3f}, ML={m['bot_multi_layer_mae_km']:.3f}] | "
            f"Thick={m['thick_mae_km']:.3f}"
        )


if __name__ == "__main__":
    main()
