from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset import MAX_HEIGHT_KM

# 15,000 feet in kilometers (15,000 * 0.3048 m = 4.572 km); also track 5.0 km threshold
THRESHOLD_15KFT_KM = 4.572
THRESHOLD_5KM = 5.0
KM_TO_KFT = 3.28084  # 1 km = 3.28084 kft


class SupervisedCloudLoss(nn.Module):
    """
    Sparse-pixel supervised loss computed strictly on pixels where `valid_mask == 1` (ICESat-2 beams):
      1. Classification Loss: Calibrated Binary Cross-Entropy on `valid_mask == 1`.
      2. Height Regression Loss: L1 + MSE loss on `cloud_top` and `cloud_bot` strictly on
         labeled cloudy pixels (`valid_mask == 1` & `cloud_mask == 1`).
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 8.0,
        order_weight: float = 0.5,
        pos_weight: float = 1.35,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.order_weight = order_weight
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logit = preds["cloud_logit"].squeeze(1)   # (B, H, W)
        pred_top = preds["cloud_top"].squeeze(1)  # (B, H, W)
        pred_bot = preds["cloud_bot"].squeeze(1)  # (B, H, W)

        valid_mask = batch["valid_mask"]          # (B, H, W)
        cloud_mask = batch["cloud_mask"]          # (B, H, W)
        gt_top = batch["cloud_top"]               # (B, H, W)
        gt_bot = batch["cloud_bot"]               # (B, H, W)

        # 1. Sparse Classification Loss on valid_mask == 1
        valid_bool = valid_mask > 0.5
        if valid_bool.any():
            v_logit = logit[valid_bool]
            v_target = cloud_mask[valid_bool]
            bce = F.binary_cross_entropy_with_logits(
                v_logit,
                v_target,
                pos_weight=torch.tensor(self.pos_weight, device=logit.device, dtype=logit.dtype),
                reduction="none",
            )
            if self.focal_gamma > 0:
                prob = torch.sigmoid(v_logit)
                p_t = v_target * prob + (1.0 - v_target) * (1.0 - prob)
                focal_factor = (1.0 - p_t).pow(self.focal_gamma)
                loss_cls = (focal_factor * bce).mean()
            else:
                loss_cls = bce.mean()
        else:
            loss_cls = logit.sum() * 0.0

        # 2. Sparse Height Regression Loss (L1 + MSE) on valid_mask == 1 AND cloud_mask == 1
        cloudy_bool = valid_bool & (cloud_mask > 0.5) & (gt_top > 0.0)
        if cloudy_bool.any():
            p_t, g_t = pred_top[cloudy_bool], gt_top[cloudy_bool]
            p_b, g_b = pred_bot[cloudy_bool], gt_bot[cloudy_bool]
            loss_top = 0.5 * F.l1_loss(p_t, g_t) + 0.5 * F.mse_loss(p_t, g_t)
            loss_bot = 0.5 * F.l1_loss(p_b, g_b) + 0.5 * F.mse_loss(p_b, g_b)
            loss_reg = 0.5 * (loss_top + loss_bot)
        else:
            loss_top = pred_top.sum() * 0.0
            loss_bot = pred_bot.sum() * 0.0
            loss_reg = pred_top.sum() * 0.0

        # 3. Physical Ordering Constraint (cloud_top >= cloud_bot everywhere)
        loss_order = F.relu(pred_bot - pred_top).mean()

        total_loss = (
            self.cls_weight * loss_cls
            + self.reg_weight * loss_reg
            + self.order_weight * loss_order
        )
        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_top": loss_top.detach(),
            "loss_bot": loss_bot.detach(),
            "loss_order": loss_order.detach(),
        }


class SemiSupervisedConsistencyLoss(nn.Module):
    """
    Class-Balanced UniMatch / Mean Teacher consistency loss on UNLABELED pixels (`valid_mask == 0`):
      - Prevents majority-class (clear-sky) confirmation bias by balancing confident cloudy
        and confident clear-sky pseudo-labels (`pos_mask` and `neg_mask`).
      - Uses sharpened soft targets so Teacher probability calibration is preserved.
      - Applies L1 + MSE height regression consistency on confident cloudy unlabeled pixels.
    """

    def __init__(
        self,
        conf_thresh_cls: float = 0.80,
        conf_thresh_reg: float = 0.70,
        cls_cons_weight: float = 1.0,
        reg_cons_weight: float = 8.0,
    ) -> None:
        super().__init__()
        self.conf_thresh_cls = conf_thresh_cls
        self.conf_thresh_reg = conf_thresh_reg
        self.cls_cons_weight = cls_cons_weight
        self.reg_cons_weight = reg_cons_weight

    def forward(
        self,
        student_preds: dict[str, torch.Tensor],
        teacher_targets: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        s_logit = student_preds["cloud_logit"].squeeze(1)
        s_top = student_preds["cloud_top"].squeeze(1)
        s_bot = student_preds["cloud_bot"].squeeze(1)

        t_prob = teacher_targets["cloud_prob"].squeeze(1).detach()
        t_top = teacher_targets["cloud_top"].squeeze(1).detach()
        t_bot = teacher_targets["cloud_bot"].squeeze(1).detach()

        unlabeled = (valid_mask < 0.5).float()

        # Separate confident positive (cloudy) and confident negative (clear) masks
        pos_mask = unlabeled * (t_prob >= self.conf_thresh_cls).float()
        neg_mask = unlabeled * ((1.0 - t_prob) >= self.conf_thresh_cls).float()
        cls_mask = pos_mask + neg_mask

        # Gently sharpened soft target in [0, 1] to preserve calibration around 0.5
        t_sharp = torch.clamp((t_prob - 0.5) * 1.4 + 0.5, 0.0, 1.0)
        bce_map = F.binary_cross_entropy_with_logits(s_logit, t_sharp, reduction="none")

        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        if pos_count > 0 and neg_count > 0:
            loss_pos = (bce_map * pos_mask).sum() / pos_count
            loss_neg = (bce_map * neg_mask).sum() / neg_count
            # Class-balanced combination matching ~1.35 positive weight
            loss_cons_cls = 0.57 * loss_pos + 0.43 * loss_neg
        elif cls_mask.sum() > 0:
            loss_cons_cls = (bce_map * cls_mask).sum() / cls_mask.sum().clamp(min=1.0)
        else:
            loss_cons_cls = s_logit.sum() * 0.0

        # Height regression consistency (L1 + MSE) on confident cloudy unlabeled pixels
        reg_mask = unlabeled * (t_prob >= self.conf_thresh_reg).float()
        reg_count = reg_mask.sum()
        if reg_count > 0:
            top_l1 = (torch.abs(s_top - t_top) * reg_mask).sum() / reg_count
            top_mse = (((s_top - t_top) ** 2) * reg_mask).sum() / reg_count
            bot_l1 = (torch.abs(s_bot - t_bot) * reg_mask).sum() / reg_count
            bot_mse = (((s_bot - t_bot) ** 2) * reg_mask).sum() / reg_count
            loss_cons_reg = 0.25 * (top_l1 + top_mse + bot_l1 + bot_mse)
        else:
            loss_cons_reg = s_top.sum() * 0.0

        total_cons = self.cls_cons_weight * loss_cons_cls + self.reg_cons_weight * loss_cons_reg
        return {
            "loss_cons": total_cons,
            "loss_cons_cls": loss_cons_cls.detach(),
            "loss_cons_reg": loss_cons_reg.detach(),
            "cls_mask_ratio": cls_mask.mean().detach(),
            "reg_mask_ratio": reg_mask.mean().detach(),
        }


@torch.no_grad()
def compute_gt_distance_and_nearest(
    valid_mask: torch.Tensor, max_dist: float = 362.0, chunk_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact 2-pass separable 2D Euclidean Distance & Nearest-GT Coordinate Transform on GPU.
    Computes for every pixel (r, c):
      1. dist_map: (B, H, W) Euclidean distance (in pixels) to the nearest GT lidar pixel (`valid_mask > 0.5`).
      2. nearest_r: (B, H, W) row index r*(r, c) of the nearest GT lidar pixel.
      3. nearest_c: (B, H, W) column index c*(r, c) of the nearest GT lidar pixel.
    Processed in mini-batches of `chunk_size` to keep temporary GPU memory minimal.
    """
    if valid_mask.dim() == 4:
        mask = (valid_mask.squeeze(1) > 0.5)
    else:
        mask = (valid_mask > 0.5)

    B, H, W = mask.shape
    device = mask.device
    inf_val = 1e6

    cols = torch.arange(W, device=device, dtype=torch.float32)
    delta_w = (cols.unsqueeze(1) - cols.unsqueeze(0)) ** 2  # (W, W)
    rows = torch.arange(H, device=device, dtype=torch.float32)
    delta_h = (rows.unsqueeze(1) - rows.unsqueeze(0)) ** 2  # (H, H)

    dist_list, r_list, c_list = [], [], []
    for i in range(0, B, chunk_size):
        m_chunk = mask[i : i + chunk_size]
        b_c = m_chunk.shape[0]
        init_cost = torch.where(
            m_chunk,
            torch.zeros((b_c, H, W), device=device, dtype=torch.float32),
            torch.full((b_c, H, W), inf_val, device=device, dtype=torch.float32),
        )
        # Pass 1: 1D parabolic lower envelope along columns (W)
        cost_col = init_cost.unsqueeze(2) + delta_w.unsqueeze(0).unsqueeze(0)
        d2_col, argmin_c = cost_col.min(dim=-1)

        # Pass 2: 1D parabolic lower envelope along rows (H)
        cost_row = d2_col.permute(0, 2, 1).unsqueeze(2) + delta_h.unsqueeze(0).unsqueeze(0)
        d2_full_t, argmin_r_t = cost_row.min(dim=-1)

        d2_full = d2_full_t.permute(0, 2, 1)
        nearest_r = argmin_r_t.permute(0, 2, 1)
        nearest_c = torch.gather(argmin_c, dim=1, index=nearest_r)
        dist_map = torch.sqrt(d2_full.clamp(min=0.0, max=max_dist * max_dist))

        dist_list.append(dist_map)
        r_list.append(nearest_r)
        c_list.append(nearest_c)

    return torch.cat(dist_list, dim=0), torch.cat(r_list, dim=0), torch.cat(c_list, dim=0)


@torch.no_grad()
def compute_gt_distance_map(valid_mask: torch.Tensor, max_dist: float = 362.0) -> torch.Tensor:
    """
    Exact 2-pass separable 2D Euclidean Distance Transform on GPU.
    Computes for every pixel (r, c) its Euclidean distance (in pixels) to the nearest
    ground-truth lidar pixel (`valid_mask > 0.5`).
    """
    dist_map, _, _ = compute_gt_distance_and_nearest(valid_mask, max_dist=max_dist)
    return dist_map


class DilatedSupervisedCloudLoss(nn.Module):
    """
    Bilateral Dilated Ground-Truth Supervision with Decayed Loss Weight (w_dil < 1):
    Trains on both exact ICESat-2 ground-truth pixels (d_GT == 0, weight = 1.0) and
    nearby unlabeled pixels within a dilation band (0 < d_GT(r, c) <= r_dil) by propagating
    the nearest ground-truth label (r*, c*) weighted by a spatial-radiometric kernel strictly < 1:
        w_dil(r, c) = gamma_dil * exp(-d_GT(r, c)^2 / (2 * sigma_s^2)) * exp(-||I(r, c) - I(r*, c*)||^2 / (2 * sigma_r^2)) < 1
    where gamma_dil in (0, 1) ensures dilated pixels have strictly lower weight than exact labeled pixels
    and the bilateral radiometric term prevents label bleeding across cloud boundaries.
    """

    def __init__(
        self,
        dilate_radius: float = 3.5,
        dilate_gamma: float = 0.45,
        sigma_spatial: float = 2.0,
        sigma_radio: float = 0.12,
        cls_weight: float = 1.0,
        reg_weight: float = 8.0,
        order_weight: float = 0.5,
        pos_weight: float = 1.35,
    ) -> None:
        super().__init__()
        self.dilate_radius = dilate_radius
        self.dilate_gamma = dilate_gamma
        self.sigma_spatial = sigma_spatial
        self.sigma_radio = sigma_radio
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.order_weight = order_weight
        self.pos_weight = pos_weight

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logit = preds["cloud_logit"].squeeze(1)
        pred_top = preds["cloud_top"].squeeze(1)
        pred_bot = preds["cloud_bot"].squeeze(1)

        valid_mask = batch["valid_mask"]
        cloud_mask = batch["cloud_mask"]
        gt_top = batch["cloud_top"]
        gt_bot = batch["cloud_bot"]

        B, H, W = valid_mask.shape
        device = logit.device

        dist_map, nearest_r, nearest_c = compute_gt_distance_and_nearest(valid_mask)
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)

        near_cloud = cloud_mask[b_idx, nearest_r, nearest_c]
        near_top = gt_top[b_idx, nearest_r, nearest_c]
        near_bot = gt_bot[b_idx, nearest_r, nearest_c]

        w_spatial = torch.exp(-(dist_map ** 2) / (2.0 * (self.sigma_spatial ** 2)))
        if image is not None:
            n_rad_ch = min(2, image.shape[1])
            img_rad = image[:, :n_rad_ch]
            b_idx_c = b_idx.unsqueeze(1).expand(B, n_rad_ch, H, W)
            ch_idx = torch.arange(n_rad_ch, device=device).view(1, n_rad_ch, 1, 1).expand(B, n_rad_ch, H, W)
            nr_c = nearest_r.unsqueeze(1).expand(B, n_rad_ch, H, W)
            nc_c = nearest_c.unsqueeze(1).expand(B, n_rad_ch, H, W)
            near_img_rad = img_rad[b_idx_c, ch_idx, nr_c, nc_c]
            rad_diff2 = ((img_rad - near_img_rad) ** 2).mean(dim=1)
            w_radio = torch.exp(-rad_diff2 / (2.0 * (self.sigma_radio ** 2)))
        else:
            w_radio = torch.ones_like(w_spatial)

        w_dil = self.dilate_gamma * w_spatial * w_radio  # strictly <= dilate_gamma < 1.0
        is_exact = valid_mask > 0.5
        is_dilated = (~is_exact) & (dist_map <= self.dilate_radius) & (w_radio >= 0.25)

        sup_weight = torch.where(
            is_exact,
            torch.ones_like(w_dil),
            torch.where(is_dilated, w_dil, torch.zeros_like(w_dil)),
        )
        target_cls = torch.where(is_exact, cloud_mask, near_cloud)
        target_top = torch.where(is_exact, gt_top, near_top)
        target_bot = torch.where(is_exact, gt_bot, near_bot)

        active_cls = sup_weight > 0.0
        if active_cls.any():
            bce = F.binary_cross_entropy_with_logits(
                logit[active_cls],
                target_cls[active_cls],
                pos_weight=torch.tensor(self.pos_weight, device=device, dtype=logit.dtype),
                reduction="none",
            )
            w_c = sup_weight[active_cls]
            loss_cls = (bce * w_c).sum() / w_c.sum().clamp(min=1.0)
        else:
            loss_cls = logit.sum() * 0.0

        active_reg = active_cls & (target_cls > 0.5) & (target_top > 0.0)
        if active_reg.any():
            p_t, g_t = pred_top[active_reg], target_top[active_reg]
            p_b, g_b = pred_bot[active_reg], target_bot[active_reg]
            w_r = sup_weight[active_reg]
            l_top = 0.85 * torch.abs(p_t - g_t) + 0.15 * ((p_t - g_t) ** 2)
            l_bot = 0.85 * torch.abs(p_b - g_b) + 0.15 * ((p_b - g_b) ** 2)
            loss_top = (l_top * w_r).sum() / w_r.sum().clamp(min=1.0)
            loss_bot = (l_bot * w_r).sum() / w_r.sum().clamp(min=1.0)
            loss_reg = 0.5 * (loss_top + loss_bot)
        else:
            loss_top = pred_top.sum() * 0.0
            loss_bot = pred_bot.sum() * 0.0
            loss_reg = pred_top.sum() * 0.0

        loss_order = F.relu(pred_bot - pred_top).mean()
        total_loss = (
            self.cls_weight * loss_cls
            + self.reg_weight * loss_reg
            + self.order_weight * loss_order
        )
        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_top": loss_top.detach(),
            "loss_bot": loss_bot.detach(),
            "loss_order": loss_order.detach(),
        }


class S4MCConsistencyLoss(nn.Module):
    """
    S4MC: Semi-Supervised Semantic Segmentation via Marginal Contextual Information
    (Kimhi et al., 2023/2024).
    Refines per-pixel teacher probabilities using the event-union probability upper bound
    over the 3x3 spatial neighborhood N(j, k) with Manhattan distance decay
    beta_{l,m} = exp(-0.5 * (|l - j| + |m - k|)):
        p_tilde_c(x_{j,k}) = max_{(l,m) in N(j,k) \\ {(j,k)}} [ p_c(x_{j,k}) + beta_{l,m} * p_c(x_{l,m}) - p_c(x_{j,k}) * beta_{l,m} * p_c(x_{l,m}) ]
    """

    def __init__(
        self,
        conf_thresh_cls: float = 0.85,
        conf_thresh_reg: float = 0.78,
        cls_cons_weight: float = 1.0,
        reg_cons_weight: float = 8.0,
    ) -> None:
        super().__init__()
        self.conf_thresh_cls = conf_thresh_cls
        self.conf_thresh_reg = conf_thresh_reg
        self.cls_cons_weight = cls_cons_weight
        self.reg_cons_weight = reg_cons_weight

        # Precompute 8-neighbor 3x3 depthwise kernels weighted by beta_{l,m} = exp(-0.5 * d_1)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        kernel = torch.zeros((8, 1, 3, 3), dtype=torch.float32)
        for idx, (dr, dc) in enumerate(offsets):
            d_manhattan = abs(dr) + abs(dc)
            beta = float(np.exp(-0.5 * d_manhattan))
            kernel[idx, 0, dr + 1, dc + 1] = beta
        self.register_buffer("neighbor_kernel", kernel)

    def _marginal_context_prob(self, prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prob: (B, H, W) class probability in [0, 1].
        Returns:
            p_tilde: (B, H, W) S4MC marginal contextual probability in [0, 1].
        """
        x = F.pad(prob.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        weighted_neighbors = F.conv2d(x, self.neighbor_kernel.to(prob.dtype))  # (B, 8, H, W)
        max_beta_neighbor = weighted_neighbors.max(dim=1).values  # (B, H, W)
        p_tilde = prob + (1.0 - prob) * max_beta_neighbor
        return p_tilde.clamp(0.0, 1.0)

    def forward(
        self,
        student_preds: dict[str, torch.Tensor],
        teacher_targets: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        s_logit = student_preds["cloud_logit"].squeeze(1)
        s_top = student_preds["cloud_top"].squeeze(1)
        s_bot = student_preds["cloud_bot"].squeeze(1)

        t_prob = teacher_targets["cloud_prob"].squeeze(1).detach()
        t_top = teacher_targets["cloud_top"].squeeze(1).detach()
        t_bot = teacher_targets["cloud_bot"].squeeze(1).detach()

        unlabeled = (valid_mask < 0.5).float()

        # Compute S4MC marginal contextual probabilities for cloud and clear-sky classes
        p_tilde_pos = self._marginal_context_prob(t_prob)
        p_tilde_neg = self._marginal_context_prob(1.0 - t_prob)

        pos_mask = unlabeled * ((t_prob >= 0.5) & (p_tilde_pos >= self.conf_thresh_cls)).float()
        neg_mask = unlabeled * ((t_prob < 0.5) & (p_tilde_neg >= self.conf_thresh_cls)).float()
        cls_mask = pos_mask + neg_mask

        t_sharp = torch.clamp((t_prob - 0.5) * 1.4 + 0.5, 0.0, 1.0)
        bce_map = F.binary_cross_entropy_with_logits(s_logit, t_sharp, reduction="none")

        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        if pos_count > 0 and neg_count > 0:
            loss_pos = (bce_map * pos_mask).sum() / pos_count
            loss_neg = (bce_map * neg_mask).sum() / neg_count
            loss_cons_cls = 0.57 * loss_pos + 0.43 * loss_neg
        elif cls_mask.sum() > 0:
            loss_cons_cls = (bce_map * cls_mask).sum() / cls_mask.sum().clamp(min=1.0)
        else:
            loss_cons_cls = s_logit.sum() * 0.0

        # Height regression consistency using S4MC context-refined cloud probability
        reg_mask = unlabeled * ((t_prob >= 0.5) & (p_tilde_pos >= self.conf_thresh_reg)).float()
        reg_count = reg_mask.sum()
        if reg_count > 0:
            top_l1 = (torch.abs(s_top - t_top) * reg_mask).sum() / reg_count
            top_mse = (((s_top - t_top) ** 2) * reg_mask).sum() / reg_count
            bot_l1 = (torch.abs(s_bot - t_bot) * reg_mask).sum() / reg_count
            bot_mse = (((s_bot - t_bot) ** 2) * reg_mask).sum() / reg_count
            loss_cons_reg = 0.25 * (top_l1 + top_mse + bot_l1 + bot_mse)
        else:
            loss_cons_reg = s_top.sum() * 0.0

        total_cons = self.cls_cons_weight * loss_cons_cls + self.reg_cons_weight * loss_cons_reg
        return {
            "loss_cons": total_cons,
            "loss_cons_cls": loss_cons_cls.detach(),
            "loss_cons_reg": loss_cons_reg.detach(),
            "cls_mask_ratio": cls_mask.mean().detach(),
            "reg_mask_ratio": reg_mask.mean().detach(),
        }


class DistanceGuidedConsistencyLoss(nn.Module):
    """
    Ours: Ground-Truth Distance-Adaptive Semi-Supervised Consistency Loss.
    Determines the pixel-wise pseudo-label confidence threshold dynamically from the
    Euclidean distance d_GT(r, c) to the nearest ground-truth ICESat-2 lidar pixel:
        tau_c(r, c) = tau_{c,min} + (tau_{c,max} - tau_{c,min}) * (1 - exp(-d_GT(r, c) / sigma_d))
        tau_r(r, c) = tau_{r,min} + (tau_{r,max} - tau_{r,min}) * (1 - exp(-d_GT(r, c) / sigma_d))
    Near the sparse ground-truth lidar track (d_GT -> 0), the teacher is strongly anchored by
    ground-truth supervision, enabling a lower threshold tau_min to recruit informative boundary
    and transitional pixels; far from the track (d_GT >> sigma_d), a stricter threshold tau_max
    suppresses far-field confirmation bias. Also applies distance-aware proximity weighting
    w_dist(r, c) = 1 + alpha_d * exp(-d_GT(r, c) / sigma_d).
    """

    def __init__(
        self,
        tau_cls_min: float = 0.65,
        tau_cls_max: float = 0.88,
        tau_reg_min: float = 0.55,
        tau_reg_max: float = 0.82,
        dist_sigma: float = 24.0,
        prox_alpha: float = 0.50,
        cls_cons_weight: float = 1.0,
        reg_cons_weight: float = 8.0,
    ) -> None:
        super().__init__()
        self.tau_cls_min = tau_cls_min
        self.tau_cls_max = tau_cls_max
        self.tau_reg_min = tau_reg_min
        self.tau_reg_max = tau_reg_max
        self.dist_sigma = dist_sigma
        self.prox_alpha = prox_alpha
        self.cls_cons_weight = cls_cons_weight
        self.reg_cons_weight = reg_cons_weight

    def forward(
        self,
        student_preds: dict[str, torch.Tensor],
        teacher_targets: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        s_logit = student_preds["cloud_logit"].squeeze(1)
        s_top = student_preds["cloud_top"].squeeze(1)
        s_bot = student_preds["cloud_bot"].squeeze(1)

        t_prob = teacher_targets["cloud_prob"].squeeze(1).detach()
        t_top = teacher_targets["cloud_top"].squeeze(1).detach()
        t_bot = teacher_targets["cloud_bot"].squeeze(1).detach()

        unlabeled = (valid_mask < 0.5).float()

        # Compute exact Euclidean distance (in pixels) to nearest GT lidar pixel
        dist_map = compute_gt_distance_map(valid_mask)
        exp_decay = torch.exp(-dist_map / max(self.dist_sigma, 1.0))
        dist_factor = 1.0 - exp_decay  # 0 at GT track, -> 1 far from GT track

        # Spatially varying confidence thresholds determined by distance from GT
        tau_cls_map = self.tau_cls_min + (self.tau_cls_max - self.tau_cls_min) * dist_factor
        tau_reg_map = self.tau_reg_min + (self.tau_reg_max - self.tau_reg_min) * dist_factor

        # Proximity weight: higher confidence weight near the ground-truth lidar track
        w_prox = 1.0 + self.prox_alpha * exp_decay

        pos_mask = unlabeled * (t_prob >= tau_cls_map).float()
        neg_mask = unlabeled * ((1.0 - t_prob) >= tau_cls_map).float()
        cls_mask = pos_mask + neg_mask

        t_sharp = torch.clamp((t_prob - 0.5) * 1.4 + 0.5, 0.0, 1.0)
        bce_map = F.binary_cross_entropy_with_logits(s_logit, t_sharp, reduction="none") * w_prox

        pos_weight_sum = (pos_mask * w_prox).sum()
        neg_weight_sum = (neg_mask * w_prox).sum()
        if pos_weight_sum > 0 and neg_weight_sum > 0:
            loss_pos = (bce_map * pos_mask).sum() / pos_weight_sum
            loss_neg = (bce_map * neg_mask).sum() / neg_weight_sum
            loss_cons_cls = 0.57 * loss_pos + 0.43 * loss_neg
        elif cls_mask.sum() > 0:
            loss_cons_cls = (bce_map * cls_mask).sum() / (cls_mask * w_prox).sum().clamp(min=1.0)
        else:
            loss_cons_cls = s_logit.sum() * 0.0

        # Distance-thresholded height regression consistency
        reg_mask = unlabeled * (t_prob >= tau_reg_map).float()
        reg_weight = reg_mask * w_prox
        reg_weight_sum = reg_weight.sum()
        if reg_weight_sum > 0:
            top_l1 = (torch.abs(s_top - t_top) * reg_weight).sum() / reg_weight_sum
            top_mse = (((s_top - t_top) ** 2) * reg_weight).sum() / reg_weight_sum
            bot_l1 = (torch.abs(s_bot - t_bot) * reg_weight).sum() / reg_weight_sum
            bot_mse = (((s_bot - t_bot) ** 2) * reg_weight).sum() / reg_weight_sum
            loss_cons_reg = 0.25 * (top_l1 + top_mse + bot_l1 + bot_mse)
        else:
            loss_cons_reg = s_top.sum() * 0.0

        total_cons = self.cls_cons_weight * loss_cons_cls + self.reg_cons_weight * loss_cons_reg
        return {
            "loss_cons": total_cons,
            "loss_cons_cls": loss_cons_cls.detach(),
            "loss_cons_reg": loss_cons_reg.detach(),
            "cls_mask_ratio": cls_mask.mean().detach(),
            "reg_mask_ratio": reg_mask.mean().detach(),
        }


def _compute_subset_error_stats(pred_km: np.ndarray, gt_km: np.ndarray, prefix: str) -> dict[str, float]:
    if len(gt_km) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mae_km": 0.0,
            f"{prefix}_mae_m": 0.0,
            f"{prefix}_mae_kft": 0.0,
            f"{prefix}_rmse_km": 0.0,
            f"{prefix}_rmse_m": 0.0,
            f"{prefix}_rmse_kft": 0.0,
            f"{prefix}_bias_km": 0.0,
        }
    err = pred_km - gt_km
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    return {
        f"{prefix}_count": int(len(gt_km)),
        f"{prefix}_mae_km": mae,
        f"{prefix}_mae_m": mae * 1000.0,
        f"{prefix}_mae_kft": mae * KM_TO_KFT,
        f"{prefix}_rmse_km": rmse,
        f"{prefix}_rmse_m": rmse * 1000.0,
        f"{prefix}_rmse_kft": rmse * KM_TO_KFT,
        f"{prefix}_bias_km": bias,
    }


@torch.no_grad()
def evaluate_sparse_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate dense pixel predictions strictly against sparse labeled ICESat-2 pixels (`valid_mask == 1`).
    Computes:
      1. Cloud Detection metrics: Accuracy, Precision, Recall, F1, AUROC.
      2. Overall Cloud Top & Base Height metrics: MAE, RMSE, R2, Pearson r.
      3. Category Breakdown by Altitude:
         - Clouds up to 15 kft (<= 4.572 km, and <= 5.0 km)
         - Clouds above 15 kft (> 4.572 km, and > 5.0 km)
    """
    model.eval()
    all_probs, all_targets = [], []
    all_pred_top_km, all_gt_top_km = [], []
    all_pred_bot_km, all_gt_bot_km = [], []
    all_n_layers, all_surf_km = [], []
    total_pixels_checked = 0
    total_order_violations = 0

    for batch in dataloader:
        img = batch["image"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"] > 0.5
        cloud_mask = batch["cloud_mask"]
        gt_top = batch["cloud_top"]
        gt_bot = batch["cloud_bot"]
        n_layers = batch.get("n_layers")
        surf_h = batch.get("surface_height")

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(img)
        prob = out["cloud_prob"].float().squeeze(1).cpu()
        pred_top_raw = out["cloud_top"].float().squeeze(1).cpu()
        pred_bot_raw = out["cloud_bot"].float().squeeze(1).cpu()

        total_pixels_checked += int(pred_top_raw.numel())
        total_order_violations += int((pred_bot_raw > pred_top_raw + 1e-4).sum().item())

        pred_top = pred_top_raw
        pred_bot = torch.minimum(pred_bot_raw, pred_top_raw)

        if valid_mask.any():
            all_probs.append(prob[valid_mask].numpy())
            all_targets.append(cloud_mask[valid_mask].numpy())

        cloudy_mask = valid_mask & (cloud_mask > 0.5) & (gt_top > 0.0)
        if cloudy_mask.any():
            all_pred_top_km.append((pred_top[cloudy_mask] * MAX_HEIGHT_KM).numpy())
            all_gt_top_km.append((gt_top[cloudy_mask] * MAX_HEIGHT_KM).numpy())
            all_pred_bot_km.append((pred_bot[cloudy_mask] * MAX_HEIGHT_KM).numpy())
            all_gt_bot_km.append((gt_bot[cloudy_mask] * MAX_HEIGHT_KM).numpy())
            if n_layers is not None:
                all_n_layers.append(n_layers[cloudy_mask].numpy())
            if surf_h is not None:
                all_surf_km.append(surf_h[cloudy_mask].numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    preds_bin = (probs >= 0.5).astype(np.float32)
    tp = float(np.sum((preds_bin == 1) & (targets == 1)))
    fp = float(np.sum((preds_bin == 1) & (targets == 0)))
    fn = float(np.sum((preds_bin == 0) & (targets == 1)))
    tn = float(np.sum((preds_bin == 0) & (targets == 0)))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-8)
    auroc = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5

    pred_top_km = np.concatenate(all_pred_top_km)
    gt_top_km = np.concatenate(all_gt_top_km)
    pred_bot_km = np.concatenate(all_pred_bot_km)
    gt_bot_km = np.concatenate(all_gt_bot_km)

    def reg_metrics(pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float]:
        err = pred - gt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        corr = float(np.corrcoef(pred, gt)[0, 1]) if len(pred) > 2 else 0.0
        res = {
            f"{prefix}_mae_km": mae,
            f"{prefix}_mae_m": mae * 1000.0,
            f"{prefix}_mae_kft": mae * KM_TO_KFT,
            f"{prefix}_rmse_km": rmse,
            f"{prefix}_rmse_m": rmse * 1000.0,
            f"{prefix}_rmse_kft": rmse * KM_TO_KFT,
            f"{prefix}_bias_km": bias,
            f"{prefix}_r2": r2,
            f"{prefix}_pearson_r": corr,
        }
        # Breakdown by 15 kft (4.572 km)
        low_15kft = gt <= THRESHOLD_15KFT_KM
        high_15kft = gt > THRESHOLD_15KFT_KM
        res.update(_compute_subset_error_stats(pred[low_15kft], gt[low_15kft], f"{prefix}_upto_15kft"))
        res.update(_compute_subset_error_stats(pred[high_15kft], gt[high_15kft], f"{prefix}_above_15kft"))

        # Breakdown by 5.0 km (~16.4 kft)
        low_5km = gt <= THRESHOLD_5KM
        high_5km = gt > THRESHOLD_5KM
        res.update(_compute_subset_error_stats(pred[low_5km], gt[low_5km], f"{prefix}_upto_5km"))
        res.update(_compute_subset_error_stats(pred[high_5km], gt[high_5km], f"{prefix}_above_5km"))
        return res

    cloud_iou = tp / max(tp + fp + fn, 1.0)
    clear_iou = tn / max(tn + fp + fn, 1.0)
    miou = 0.5 * (cloud_iou + clear_iou)

    metrics = {
        "n_eval_pixels": int(len(targets)),
        "n_eval_cloudy_pixels": int(len(gt_top_km)),
        "cloud_acc": acc,
        "cloud_precision": prec,
        "cloud_recall": rec,
        "cloud_f1": f1,
        "cloud_iou": cloud_iou,
        "clear_iou": clear_iou,
        "miou": miou,
        "cloud_auroc": auroc,
        "order_violation_pct": 100.0 * total_order_violations / max(total_pixels_checked, 1),
    }
    metrics.update(reg_metrics(pred_top_km, gt_top_km, "top"))
    metrics.update(reg_metrics(pred_bot_km, gt_bot_km, "bot"))

    # Cloud Geometric Thickness (top - bot) metrics
    pred_thick_km = np.clip(pred_top_km - pred_bot_km, 0.0, MAX_HEIGHT_KM)
    gt_thick_km = np.clip(gt_top_km - gt_bot_km, 0.0, MAX_HEIGHT_KM)
    metrics.update(reg_metrics(pred_thick_km, gt_thick_km, "thick"))

    # Single-layer (n_layers == 1) vs Multi-layer (n_layers >= 2) breakdown
    if all_n_layers:
        nl = np.concatenate(all_n_layers)
        single_m = nl == 1
        multi_m = nl >= 2
        metrics.update(_compute_subset_error_stats(pred_top_km[single_m], gt_top_km[single_m], "top_single_layer"))
        metrics.update(_compute_subset_error_stats(pred_top_km[multi_m], gt_top_km[multi_m], "top_multi_layer"))
        metrics.update(_compute_subset_error_stats(pred_bot_km[single_m], gt_bot_km[single_m], "bot_single_layer"))
        metrics.update(_compute_subset_error_stats(pred_bot_km[multi_m], gt_bot_km[multi_m], "bot_multi_layer"))

    # Marine (surface_height <= 0.05 km) vs Land (surface_height > 0.05 km) breakdown
    if all_surf_km:
        sf = np.concatenate(all_surf_km)
        marine_m = sf <= 0.05
        land_m = sf > 0.05
        metrics.update(_compute_subset_error_stats(pred_top_km[marine_m], gt_top_km[marine_m], "top_marine"))
        metrics.update(_compute_subset_error_stats(pred_top_km[land_m], gt_top_km[land_m], "top_land"))
        metrics.update(_compute_subset_error_stats(pred_bot_km[marine_m], gt_bot_km[marine_m], "bot_marine"))
        metrics.update(_compute_subset_error_stats(pred_bot_km[land_m], gt_bot_km[land_m], "bot_land"))

    return metrics
