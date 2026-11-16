from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_weak_augmentation(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Apply random geometric weak augmentations (horizontal flip, vertical flip, 90-deg rotations)
    consistently to both the input image and all 2D target maps.
    """
    img = batch["image"]
    valid_mask = batch["valid_mask"]
    cloud_mask = batch["cloud_mask"]
    cloud_top = batch["cloud_top"]
    cloud_bot = batch["cloud_bot"]

    # Random horizontal flip
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[3])
        valid_mask = torch.flip(valid_mask, dims=[2])
        cloud_mask = torch.flip(cloud_mask, dims=[2])
        cloud_top = torch.flip(cloud_top, dims=[2])
        cloud_bot = torch.flip(cloud_bot, dims=[2])

    # Random vertical flip
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[2])
        valid_mask = torch.flip(valid_mask, dims=[1])
        cloud_mask = torch.flip(cloud_mask, dims=[1])
        cloud_top = torch.flip(cloud_top, dims=[1])
        cloud_bot = torch.flip(cloud_bot, dims=[1])

    # Random 90-degree rotation (k in {0, 1, 2, 3})
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        img = torch.rot90(img, k=k, dims=[2, 3])
        valid_mask = torch.rot90(valid_mask, k=k, dims=[1, 2])
        cloud_mask = torch.rot90(cloud_mask, k=k, dims=[1, 2])
        cloud_top = torch.rot90(cloud_top, k=k, dims=[1, 2])
        cloud_bot = torch.rot90(cloud_bot, k=k, dims=[1, 2])

    return {
        "image": img,
        "valid_mask": valid_mask,
        "cloud_mask": cloud_mask,
        "cloud_top": cloud_top,
        "cloud_bot": cloud_bot,
    }


def apply_strong_photometric_augmentation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply strong photometric & spatial context perturbations to all radiometric channels (`x[:, :-2]`,
    e.g. IR 10.5 um and VIS 0.6 um across 1 or 3 temporal frames) of a weakly-augmented input tensor `x`
    of shape (B, C, H, W).
    Applies identical per-sample contrast/brightness/gamma/CutOut across the temporal sequence so
    inter-frame motion differences remain physically coherent, and leaves coordinate channels (`x[:, -2:]`) intact.
    """
    B, C, H, W = x.shape
    device = x.device
    if C == 20:
        # ERA5-only input ([T(9), RH(9), lat, lon]): preserve physical calibration of T and RH,
        # applying only mild sensor noise and local spatial CutOut on the 18 ERA5 channels.
        era5 = x[:, :18].clone()
        mean_era5 = era5.mean(dim=(2, 3), keepdim=True)
        noise_std = torch.empty((B, 1, 1, 1), device=device).uniform_(0.0, 0.015)
        era5 = (era5 + torch.randn_like(era5) * noise_std).clamp(0.0, 1.0)
        for b in range(B):
            if torch.rand(1).item() < 0.5:
                bh = int(torch.randint(12, 36, (1,)).item())
                bw = int(torch.randint(12, 36, (1,)).item())
                r0 = int(torch.randint(0, max(H - bh, 1), (1,)).item())
                c0 = int(torch.randint(0, max(W - bw, 1), (1,)).item())
                era5[b, :, r0 : r0 + bh, c0 : c0 + bw] = mean_era5[b, :, :, :]
        return torch.cat([era5, x[:, 18:]], dim=1)

    n_rad = max((C - 20) if C > 20 else (C - 2), 1)
    rad = x[:, :n_rad].clone()

    # 1. Random multiplicative contrast + additive brightness jitter per sample
    contrast = torch.empty((B, 1, 1, 1), device=device).uniform_(0.78, 1.25)
    brightness = torch.empty((B, 1, 1, 1), device=device).uniform_(-0.06, 0.06)
    mean_rad = rad.mean(dim=(2, 3), keepdim=True)
    rad = (rad - mean_rad) * contrast + mean_rad + brightness
    rad = rad.clamp(0.0, 1.0)

    # 2. Random non-linear gamma adjustment per sample
    gamma = torch.empty((B, 1, 1, 1), device=device).uniform_(0.80, 1.25)
    rad = rad.clamp(min=1e-5).pow(gamma).clamp(0.0, 1.0)

    # 3. Random 3x3 Gaussian blur on ~50% of samples
    if torch.rand(1).item() < 0.5:
        kernel = (
            torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                device=device,
                dtype=x.dtype,
            ).view(1, 1, 3, 3)
            / 16.0
        ).repeat(n_rad, 1, 1, 1)
        rad_blur = F.conv2d(F.pad(rad, (1, 1, 1, 1), mode="reflect"), kernel, groups=n_rad)
        blur_mask = (torch.rand((B, 1, 1, 1), device=device) < 0.6).float()
        rad = rad * (1.0 - blur_mask) + rad_blur * blur_mask

    # 4. Random Gaussian sensor noise
    noise_std = torch.empty((B, 1, 1, 1), device=device).uniform_(0.0, 0.025)
    rad = (rad + torch.randn_like(rad) * noise_std).clamp(0.0, 1.0)

    # 5. Random CutOut (small local occlusion forcing neighborhood context usage)
    for b in range(B):
        if torch.rand(1).item() < 0.5:
            bh = int(torch.randint(12, 36, (1,)).item())
            bw = int(torch.randint(12, 36, (1,)).item())
            r0 = int(torch.randint(0, max(H - bh, 1), (1,)).item())
            c0 = int(torch.randint(0, max(W - bw, 1), (1,)).item())
            rad[b, :, r0 : r0 + bh, c0 : c0 + bw] = mean_rad[b, :, :, :]

    if C > n_rad:
        return torch.cat([rad, x[:, n_rad:]], dim=1)
    return rad


def apply_cutmix_pair(
    x_strong: torch.Tensor,
    teacher_preds: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    p: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """
    UniMatch-style spatial CutMix between rolled batch pairs for strong consistency regularization.
    Mixes both the strong student input `x_strong` and the teacher pseudo-label maps so pixel-wise
    correspondence is preserved.
    """
    if torch.rand(1).item() >= p or x_strong.shape[0] < 2:
        return x_strong, teacher_preds, valid_mask

    B, _, H, W = x_strong.shape
    device = x_strong.device
    box_mask = torch.zeros((B, 1, H, W), device=device, dtype=x_strong.dtype)

    for b in range(B):
        bh = int(torch.randint(H // 4, H // 2 + 1, (1,)).item())
        bw = int(torch.randint(W // 4, W // 2 + 1, (1,)).item())
        r0 = int(torch.randint(0, H - bh + 1, (1,)).item())
        c0 = int(torch.randint(0, W - bw + 1, (1,)).item())
        box_mask[b, 0, r0 : r0 + bh, c0 : c0 + bw] = 1.0

    x_rolled = torch.roll(x_strong, shifts=1, dims=0)
    x_mixed = x_strong * (1.0 - box_mask) + x_rolled * box_mask

    t_mixed = {}
    for k, v in teacher_preds.items():
        v_rolled = torch.roll(v, shifts=1, dims=0)
        t_mixed[k] = v * (1.0 - box_mask) + v_rolled * box_mask

    box_2d = box_mask.squeeze(1)
    vm_rolled = torch.roll(valid_mask, shifts=1, dims=0)
    vm_mixed = valid_mask * (1.0 - box_2d) + vm_rolled * box_2d

    return x_mixed, t_mixed, vm_mixed
