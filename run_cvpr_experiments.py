from __future__ import annotations

import datetime
import gc
import json
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from augmentations import (
    apply_cutmix_pair,
    apply_strong_photometric_augmentation,
    apply_weak_augmentation,
)
from dataset import (
    DEFAULT_BBOX,
    DEFAULT_IMAGE_SHAPE,
    MAX_HEIGHT_KM,
    CloudPatchDataset,
    MatchedGranule,
    discover_matched_granules,
    extract_granule_tracks,
    filter_granules_with_30min_window,
    get_3frame_filenames,
    load_ir_image,
    split_granules_chronologically,
)
from losses import (
    KM_TO_KFT,
    THRESHOLD_5KM,
    THRESHOLD_15KFT_KM,
    SemiSupervisedConsistencyLoss,
    SupervisedCloudLoss,
    _compute_subset_error_stats,
    evaluate_sparse_predictions,
)
from models import (
    EMATeacher,
    MultiTaskDINOv3,
    RadiometricContextStem,
    build_model,
    extract_fp_dict,
    get_optimizer_param_groups,
    unwrap_model,
)

BUILD_LOCK = threading.Lock()
JSON_LOCK = threading.Lock()


def set_batchnorm_eval(model: nn.Module, is_eval: bool = True) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if is_eval:
                m.eval()
            else:
                m.train()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preload_all_images_and_tracks(
    granules_all: list[MatchedGranule],
    ir_dir: str = "../ir10.5",
    vis_dir: str = "../vis0.6",
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    t0 = time.time()
    paths: set[str] = set()
    for g in granules_all:
        for fn in get_3frame_filenames(g.img_name):
            p_ir = g.img_path if fn == g.img_name else os.path.join(ir_dir, fn)
            if os.path.isfile(p_ir):
                paths.add(p_ir)
            p_vis = os.path.join(vis_dir, fn)
            if os.path.isfile(p_vis):
                paths.add(p_vis)

    sorted_paths = sorted(paths)
    print(f"[Preload] Loading {len(sorted_paths)} full-res 4096x4096 images in parallel across 24 threads...")
    with ThreadPoolExecutor(max_workers=24) as ex:
        imgs = list(ex.map(load_ir_image, sorted_paths))
    img_cache = dict(zip(sorted_paths, imgs))

    print(f"[Preload] Extracting tracks for {len(granules_all)} granules...")
    track_cache: dict[str, dict[str, np.ndarray]] = {}
    for g in granules_all:
        track_cache[g.granule_id] = extract_granule_tracks(
            h5_path=g.h5_path,
            bbox=DEFAULT_BBOX,
            img_shape=DEFAULT_IMAGE_SHAPE,
            cloud_attr_mode="cloud_only",
        )
    print(f"[Preload] Done in {time.time() - t0:.1f}s ({len(img_cache)} images, {len(track_cache)} tracks).")
    return img_cache, track_cache


def make_fast_patch_dataset(
    granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    patches_per_granule: int,
    is_train: bool,
    use_3frames: bool,
    use_vis06: bool,
    allowed_beams: tuple[int, ...] = (1, 2, 3),
    seed: int = 42,
) -> CloudPatchDataset:
    ds = CloudPatchDataset.__new__(CloudPatchDataset)
    ds.granules = granules
    ds.patch_size = 256
    ds.patches_per_granule = patches_per_granule
    ds.bbox = DEFAULT_BBOX
    ds.cloud_attr_mode = "cloud_only"
    ds.is_train = is_train
    ds.rng = np.random.default_rng(seed)
    ds.max_cached_images = 100000
    ds.use_3frames = use_3frames
    ds.use_vis06 = use_vis06
    ds.use_era5 = False
    ds.use_satellite = True
    ds._era5_cache = {}
    ds.ir_dir = "../ir10.5"
    ds.vis_dir = "../vis0.6"
    ds.allowed_beams = tuple(allowed_beams)
    ds._img_cache = img_cache

    ds.track_cache = []
    ds.valid_granule_indices = []
    for idx, g in enumerate(granules):
        tr = track_cache[g.granule_id]
        if ds.allowed_beams != (1, 2, 3) and len(tr["row"]) > 0:
            beam_mask = np.isin(tr["beam"], ds.allowed_beams)
            tr = {k: v[beam_mask] for k, v in tr.items()}
        ds.track_cache.append(tr)
        if len(tr["row"]) > 0:
            ds.valid_granule_indices.append(idx)
    return ds


def save_json(data: dict[str, Any], path: str) -> None:
    with JSON_LOCK:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def train_and_eval_supervised_single_gpu(
    name: str,
    model_fn: Callable[[], nn.Module],
    train_ds: CloudPatchDataset,
    val_ds: CloudPatchDataset,
    out_dir: str,
    gpu_id: int,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    order_weight: float = 0.5,
) -> dict[str, Any]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.pt")

    with BUILD_LOCK:
        model = model_fn()

    if os.path.isfile(ckpt_path):
        print(f"[GPU {gpu_id} | Supervised: {name}] Loading cached checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(ckpt["model_state"])
        model = model.to(device)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
        res = evaluate_sparse_predictions(model, val_loader, device)
        del model
        torch.cuda.empty_cache()
        return res

    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    criterion = SupervisedCloudLoss(cls_weight=1.0, reg_weight=8.0, order_weight=order_weight).to(device)
    param_groups = get_optimizer_param_groups(model, lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_score = -1e9
    best_metrics: dict[str, Any] = {}
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = apply_weak_augmentation(batch)
            img = batch["image"].to(device, non_blocking=True)
            targets = {
                "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
                "cloud_mask": batch["cloud_mask"].to(device, non_blocking=True),
                "cloud_top": batch["cloud_top"].to(device, non_blocking=True),
                "cloud_bot": batch["cloud_bot"].to(device, non_blocking=True),
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                preds = model(img)
                loss = criterion(preds, targets)["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        if epoch >= max(epochs - 4, 1) or epoch % 3 == 0:
            val_metrics = evaluate_sparse_predictions(model, val_loader, device)
            score = val_metrics["cloud_f1"] - 0.12 * (val_metrics["top_mae_km"] + val_metrics["bot_mae_km"])
            if score > best_score:
                best_score = score
                best_metrics = val_metrics
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": unwrap_model(model).state_dict(),
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  [GPU {gpu_id} | {name}] Ep {epoch:02d}/{epochs:02d} ({time.time()-t_start:.0f}s) | "
                f"Best Top MAE={best_metrics.get('top_mae_km', 0):.3f} km | "
                f"Bot MAE={best_metrics.get('bot_mae_km', 0):.3f} km | "
                f"F1={best_metrics.get('cloud_f1', 0):.4f}"
            )

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_metrics


def train_and_eval_semisup_single_gpu(
    name: str,
    model_fn: Callable[[], nn.Module],
    init_ckpt_path: str | None,
    train_ds: CloudPatchDataset,
    val_ds: CloudPatchDataset,
    out_dir: str,
    gpu_id: int,
    epochs: int = 10,
    batch_size: int = 12,
    lr: float = 2e-4,
    reg_cons_weight: float = 8.0,
    use_fp_stream: bool = True,
) -> dict[str, Any]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_semisup.pt")

    with BUILD_LOCK:
        model = model_fn()

    if os.path.isfile(ckpt_path):
        print(f"[GPU {gpu_id} | Semi-Sup: {name}] Loading cached checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("teacher_state", ckpt.get("model_state")))
        unwrap_model(model).load_state_dict(state)
        model = model.to(device)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
        res = evaluate_sparse_predictions(model, val_loader, device)
        del model
        torch.cuda.empty_cache()
        return res

    if init_ckpt_path and os.path.isfile(init_ckpt_path):
        ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("teacher_state")))
        unwrap_model(model).load_state_dict(state)

    model = model.to(device)
    teacher = EMATeacher(model, decay=0.996).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    sup_criterion = SupervisedCloudLoss(cls_weight=1.0, reg_weight=8.0, order_weight=0.5).to(device)
    cons_criterion = SemiSupervisedConsistencyLoss(
        conf_thresh_cls=0.80,
        conf_thresh_reg=0.70,
        cls_cons_weight=1.0,
        reg_cons_weight=reg_cons_weight,
    ).to(device)

    param_groups = get_optimizer_param_groups(model, lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_score = -1e9
    best_metrics: dict[str, Any] = {}
    global_step = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ramp = min(1.0, epoch / 3.0)
        cons_weight = 0.35 * ramp

        for batch in train_loader:
            batch = apply_weak_augmentation(batch)
            x_weak = batch["image"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            targets = {
                "valid_mask": valid_mask,
                "cloud_mask": batch["cloud_mask"].to(device, non_blocking=True),
                "cloud_top": batch["cloud_top"].to(device, non_blocking=True),
                "cloud_bot": batch["cloud_bot"].to(device, non_blocking=True),
            }

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                t_out = teacher.model(x_weak)
                t_out = {k: v.float() for k, v in t_out.items()}

            x_s1 = apply_strong_photometric_augmentation(x_weak)
            x_s2 = apply_strong_photometric_augmentation(x_weak)
            x_s1, t_out_s1, vm_s1 = apply_cutmix_pair(x_s1, t_out, valid_mask, p=0.5)
            x_s2, t_out_s2, vm_s2 = apply_cutmix_pair(x_s2, t_out, valid_mask, p=0.5)

            optimizer.zero_grad(set_to_none=True)

            # Stream 1: Weak view + optional Feature Perturbation (immediate backward frees activations)
            set_batchnorm_eval(model, is_eval=False)
            with torch.amp.autocast("cuda", enabled=True):
                if use_fp_stream:
                    out_w_comb = model(x_weak, need_fp=True)
                    preds_w, preds_fp = extract_fp_dict(out_w_comb)
                    sup_loss = sup_criterion(preds_w, targets)["loss"]
                    c_fp = cons_criterion(preds_fp, t_out, valid_mask)["loss_cons"]
                    loss_s1 = sup_loss + cons_weight * 0.2 * c_fp
                else:
                    preds_w = model(x_weak, need_fp=False)
                    sup_loss = sup_criterion(preds_w, targets)["loss"]
                    loss_s1 = sup_loss
            scaler.scale(loss_s1).backward()
            del preds_w

            # Stream 2: Strong view 1
            set_batchnorm_eval(model, is_eval=True)
            with torch.amp.autocast("cuda", enabled=True):
                preds_s1 = model(x_s1, need_fp=False)
                w_s = 0.4 if use_fp_stream else 0.5
                loss_s2 = cons_weight * w_s * cons_criterion(preds_s1, t_out_s1, vm_s1)["loss_cons"]
            scaler.scale(loss_s2).backward()
            del preds_s1

            # Stream 3: Strong view 2
            with torch.amp.autocast("cuda", enabled=True):
                preds_s2 = model(x_s2, need_fp=False)
                loss_s3 = cons_weight * w_s * cons_criterion(preds_s2, t_out_s2, vm_s2)["loss_cons"]
            scaler.scale(loss_s3).backward()
            del preds_s2
            set_batchnorm_eval(model, is_eval=False)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            teacher.update(model, step=None)

        scheduler.step()
        if epoch >= max(epochs - 4, 1) or epoch % 3 == 0:
            t_metrics = evaluate_sparse_predictions(teacher.model, val_loader, device)
            s_metrics = evaluate_sparse_predictions(model, val_loader, device)
            t_score = t_metrics["cloud_f1"] - 0.12 * (t_metrics["top_mae_km"] + t_metrics["bot_mae_km"])
            s_score = s_metrics["cloud_f1"] - 0.12 * (s_metrics["top_mae_km"] + s_metrics["bot_mae_km"])
            cur_metrics = t_metrics if t_score >= s_score else s_metrics
            cur_score = max(t_score, s_score)
            if cur_score > best_score:
                best_score = cur_score
                best_metrics = cur_metrics
                torch.save(
                    {
                        "epoch": epoch,
                        "teacher_state": teacher.model.state_dict(),
                        "student_state": unwrap_model(model).state_dict(),
                        "val_metrics": best_metrics,
                    },
                    ckpt_path,
                )
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  [GPU {gpu_id} | {name}] Ep {epoch:02d}/{epochs:02d} ({time.time()-t_start:.0f}s) | "
                f"Best Top MAE={best_metrics.get('top_mae_km', 0):.3f} km | "
                f"Bot MAE={best_metrics.get('bot_mae_km', 0):.3f} km | "
                f"F1={best_metrics.get('cloud_f1', 0):.4f}"
            )

    del model, teacher, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_metrics


def run_jobs_across_4_gpus(jobs: list[Callable[[int], None]]) -> None:
    if not jobs:
        return
    n_gpus = max(torch.cuda.device_count(), 1)
    q: queue.Queue[Callable[[int], None]] = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu_id: int) -> None:
        torch.cuda.set_device(gpu_id)
        while True:
            try:
                job_fn = q.get_nowait()
            except queue.Empty:
                break
            try:
                job_fn(gpu_id)
            finally:
                q.task_done()

    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futures = [ex.submit(worker, g_id) for g_id in range(n_gpus)]
        for f in futures:
            f.result()


def main() -> None:
    out_json = "outputs/cvpr_experiments/cvpr_all_results.json"
    all_res: dict[str, Any] = {}
    if os.path.isfile(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            all_res = json.load(f)

    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(granules_490, "../ir10.5", "../vis0.6", use_vis06=True)

    train_490, val_490 = split_granules_chronologically(granules_490, val_ratio=0.2)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)

    img_cache, track_cache = preload_all_images_and_tracks(granules_490)

    # Synoptic-Block Split setup (>= 48h synoptic weather buffer between train and test blocks)
    sorted_408 = sorted(granules_408, key=lambda g: g.timestamp_utc)
    n_tot = len(sorted_408)
    n_blocks = 5
    block_sz = n_tot // n_blocks
    syn_train, syn_val = [], []
    buffer_sec = 48.0 * 3600.0
    for b_i in range(n_blocks):
        blk = sorted_408[b_i * block_sz : (b_i + 1) * block_sz if b_i < n_blocks - 1 else n_tot]
        split_pt = int(len(blk) * 0.78)
        val_blk = blk[split_pt:]
        val_start_t = val_blk[0].timestamp_utc
        val_end_t = val_blk[-1].timestamp_utc
        syn_val.extend(val_blk)
        for g in blk[:split_pt]:
            if abs(g.timestamp_utc - val_start_t) >= buffer_sec and abs(g.timestamp_utc - val_end_t) >= buffer_sec:
                syn_train.append(g)

    all_res.setdefault("modality_temporal_408", {})
    all_res.setdefault("component_ablations", {})
    all_res.setdefault("beam_sparsity_study", {})
    all_res.setdefault("synoptic_block_split", {})

    mod_res = all_res["modality_temporal_408"]
    abl_res = all_res["component_ablations"]
    beam_res = all_res["beam_sparsity_study"]
    syn_res = all_res["synoptic_block_split"]

    # Copy already-computed 3f IR+VIS results into modality_temporal_408 and beam_sparsity_study
    for k_copy in (
        "Supervised DINOv3 (3f IR+VIS)",
        "Semi-Sup DINOv3 (3f IR+VIS)",
        "Supervised SAM (3f IR+VIS)",
        "Semi-Sup SAM (3f IR+VIS)",
    ):
        if k_copy not in mod_res and k_copy in all_res.get("main_16_models", {}):
            mod_res[k_copy] = all_res["main_16_models"][k_copy]
    beam_res["Supervised DINOv3 (3 Beams, 0.075% Pixels)"] = all_res["main_16_models"]["Supervised DINOv3 (3f IR+VIS)"]
    beam_res["Semi-Sup DINOv3 (3 Beams, 0.075% Pixels)"] = all_res["main_16_models"]["Semi-Sup DINOv3 (3f IR+VIS)"]
    save_json(all_res, out_json)

    # =========================================================================
    # WAVE 1: Run all Supervised Ablation Models in parallel across 4 GPUs
    # =========================================================================
    sup_jobs: list[Callable[[int], None]] = []

    # 3. Modality & Temporal Disentanglement (Supervised)
    modality_configs = [
        ("1f IR (408 Split)", False, False, 3),
        ("3f IR Only (408 Split)", True, False, 5),
        ("1f IR+VIS (408 Split)", False, True, 4),
    ]
    for mod_tag, use_3f, use_vis, in_ch in modality_configs:
        for arch_label, arch_code in [("DINOv3", "dinov3"), ("SAM", "sam")]:
            sup_key = f"Supervised {arch_label} ({mod_tag})"
            sup_dir = f"outputs/cvpr_experiments/sup_{arch_code}_{in_ch}ch_408"
            if sup_key not in mod_res:
                def _make_sup_mod_job(
                    _k=sup_key, _ac=arch_code, _ic=in_ch, _u3=use_3f, _uv=use_vis, _sd=sup_dir
                ) -> Callable[[int], None]:
                    def _run(gpu_id: int) -> None:
                        tr_ds = make_fast_patch_dataset(train_408, img_cache, track_cache, 6, True, _u3, _uv, seed=42 + gpu_id)
                        va_ds = make_fast_patch_dataset(val_408, img_cache, track_cache, 6, False, _u3, _uv, seed=42)
                        res = train_and_eval_supervised_single_gpu(
                            _k,
                            lambda: build_model(arch=_ac, in_channels=_ic, base_ch=48, pretrained=True),
                            tr_ds,
                            va_ds,
                            _sd,
                            gpu_id=gpu_id,
                            epochs=10,
                            batch_size=16,
                        )
                        mod_res[_k] = res
                        save_json(all_res, out_json)
                    return _run
                sup_jobs.append(_make_sup_mod_job())

    # 4a, 4b, 4c: Component Ablations (Supervised)
    sup_abl_specs = [
        (
            "Supervised DINOv3 (Scratch / Random Init)",
            "outputs/cvpr_experiments/abl_dinov3_scratch",
            lambda: MultiTaskDINOv3(in_channels=8, base_ch=48, pretrained=False),
            0.5,
        ),
        (
            "Supervised DINOv3 (w/o Radiometric Skip & Stem)",
            "outputs/cvpr_experiments/abl_dinov3_noskip",
            lambda: MultiTaskDINOv3(in_channels=8, base_ch=48, pretrained=True, use_rad_skip=False),
            0.5,
        ),
        (
            "Supervised DINOv3 (w/o Structural Ordering Head)",
            "outputs/cvpr_experiments/abl_dinov3_noorder",
            lambda: MultiTaskDINOv3(in_channels=8, base_ch=48, pretrained=True, enforce_ordering=False),
            0.0,
        ),
    ]
    for a_key, a_dir, m_fn, ord_w in sup_abl_specs:
        if a_key not in abl_res:
            def _make_abl_job(_k=a_key, _d=a_dir, _fn=m_fn, _ow=ord_w) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_fast_patch_dataset(train_408, img_cache, track_cache, 6, True, True, True, seed=100 + gpu_id)
                    va_ds = make_fast_patch_dataset(val_408, img_cache, track_cache, 6, False, True, True, seed=42)
                    res = train_and_eval_supervised_single_gpu(
                        _k, _fn, tr_ds, va_ds, _d, gpu_id=gpu_id, epochs=10, batch_size=16, order_weight=_ow
                    )
                    abl_res[_k] = res
                    save_json(all_res, out_json)
                return _run
            sup_jobs.append(_make_abl_job())

    # 5: Beam Sparsity Study (Supervised 1-Beam & 2-Beams)
    for b_label, beams_tuple in [("1 Beam (0.025% Pixels)", (2,)), ("2 Beams (0.050% Pixels)", (1, 2))]:
        sup_b_key = f"Supervised DINOv3 ({b_label})"
        sup_b_dir = f"outputs/cvpr_experiments/sup_dinov3_beams_{len(beams_tuple)}"
        if sup_b_key not in beam_res:
            def _make_beam_sup_job(_k=sup_b_key, _d=sup_b_dir, _bt=beams_tuple) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_fast_patch_dataset(train_408, img_cache, track_cache, 6, True, True, True, allowed_beams=_bt, seed=200 + gpu_id)
                    va_ds = make_fast_patch_dataset(val_408, img_cache, track_cache, 6, False, True, True, seed=42)
                    res = train_and_eval_supervised_single_gpu(
                        _k,
                        lambda: MultiTaskDINOv3(in_channels=8, base_ch=48, pretrained=True),
                        tr_ds,
                        va_ds,
                        _d,
                        gpu_id=gpu_id,
                        epochs=10,
                        batch_size=16,
                    )
                    beam_res[_k] = res
                    save_json(all_res, out_json)
                return _run
            sup_jobs.append(_make_beam_sup_job())

    # 6: Synoptic-Block Split (Supervised U-Net, SAM, DINOv3)
    for arch_label, arch_code in [("U-Net", "unet"), ("SAM", "sam"), ("DINOv3", "dinov3")]:
        s_key = f"Supervised {arch_label} (Synoptic-Split 3f IR+VIS)"
        s_dir = f"outputs/cvpr_experiments/synoptic_sup_{arch_code}"
        if s_key not in syn_res:
            def _make_syn_sup_job(_k=s_key, _ac=arch_code, _d=s_dir) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_fast_patch_dataset(syn_train, img_cache, track_cache, 6, True, True, True, seed=300 + gpu_id)
                    va_ds = make_fast_patch_dataset(syn_val, img_cache, track_cache, 6, False, True, True, seed=42)
                    res = train_and_eval_supervised_single_gpu(
                        _k,
                        lambda: build_model(arch=_ac, in_channels=8, base_ch=48, pretrained=True),
                        tr_ds,
                        va_ds,
                        _d,
                        gpu_id=gpu_id,
                        epochs=10,
                        batch_size=16,
                    )
                    syn_res[_k] = res
                    save_json(all_res, out_json)
                return _run
            sup_jobs.append(_make_syn_sup_job())

    print(f"\n=== WAVE 1: Launching {len(sup_jobs)} Supervised Ablation Jobs across 4 GPUs in parallel ===")
    run_jobs_across_4_gpus(sup_jobs)

    # =========================================================================
    # WAVE 2: Run all Semi-Supervised Ablation Models in parallel across 4 GPUs
    # =========================================================================
    semi_jobs: list[Callable[[int], None]] = []

    # 3. Modality & Temporal Disentanglement (Semi-Supervised)
    for mod_tag, use_3f, use_vis, in_ch in modality_configs:
        for arch_label, arch_code in [("DINOv3", "dinov3"), ("SAM", "sam")]:
            semi_key = f"Semi-Sup {arch_label} ({mod_tag})"
            sup_dir = f"outputs/cvpr_experiments/sup_{arch_code}_{in_ch}ch_408"
            semi_dir = f"outputs/cvpr_experiments/semisup_{arch_code}_{in_ch}ch_408"
            if semi_key not in mod_res:
                def _make_semi_mod_job(
                    _k=semi_key, _ac=arch_code, _ic=in_ch, _u3=use_3f, _uv=use_vis, _sd=sup_dir, _md=semi_dir
                ) -> Callable[[int], None]:
                    def _run(gpu_id: int) -> None:
                        tr_ds = make_fast_patch_dataset(train_408, img_cache, track_cache, 6, True, _u3, _uv, seed=400 + gpu_id)
                        va_ds = make_fast_patch_dataset(val_408, img_cache, track_cache, 6, False, _u3, _uv, seed=42)
                        res = train_and_eval_semisup_single_gpu(
                            _k,
                            lambda: build_model(arch=_ac, in_channels=_ic, base_ch=48, pretrained=False),
                            os.path.join(_sd, "best_model.pt"),
                            tr_ds,
                            va_ds,
                            _md,
                            gpu_id=gpu_id,
                            epochs=10,
                            batch_size=12,
                        )
                        mod_res[_k] = res
                        save_json(all_res, out_json)
                    return _run
                semi_jobs.append(_make_semi_mod_job())

    # 4d, 4e: Semi-Supervised Component Ablations
    sup_dino_3f_ckpt = "outputs/supervised_dinov3_3f_vis/best_supervised.pt"
    semi_abl_specs = [
        (
            "Semi-Sup DINOv3 (w/o Height Consistency)",
            "outputs/cvpr_experiments/abl_dinov3_no_regcons",
            0.0,
            True,
        ),
        (
            "Semi-Sup DINOv3 (w/o Feature Perturbation)",
            "outputs/cvpr_experiments/abl_dinov3_no_fp",
            8.0,
            False,
        ),
    ]
    for sa_key, sa_dir, reg_cw, use_fp in semi_abl_specs:
        if sa_key not in abl_res:
            def _make_semi_abl_job(_k=sa_key, _d=sa_dir, _rcw=reg_cw, _fp=use_fp) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_fast_patch_dataset(train_408, img_cache, track_cache, 6, True, True, True, seed=500 + gpu_id)
                    va_ds = make_fast_patch_dataset(val_408, img_cache, track_cache, 6, False, True, True, seed=42)
                    res = train_and_eval_semisup_single_gpu(
                        _k,
                        lambda: MultiTaskDINOv3(in_channels=8, base_ch=48, pretrained=False),
                        sup_dino_3f_ckpt,
                        tr_ds,
                        va_ds,
                        _d,
                        gpu_id=gpu_id,
                        epochs=10,
                        batch_size=12,
                        reg_cons_weight=_rcw,
                        use_fp_stream=_fp,
                    )
                    abl_res[_k] = res
                    save_json(all_res, out_json)
                return _run
            semi_jobs.append(_make_semi_abl_job())

    # 5: Beam Sparsity Study (Semi-Supervised 1-Beam & 2-Beams)
    for b_label, beams_tuple in [("1 Beam (0.025% Pixels)", (2,)), ("2 Beams (0.050% Pixels)", (1, 2))]:
        semi_b_key = f"Semi-Sup DINOv3 ({b_label})"
        sup_b_dir = f"outputs/cvpr_experiments/sup_dinov3_beams_{len(beams_tuple)}"
        semi_b_dir = f"outputs/cvpr_experiments/semisup_dinov3_beams_{len(beams_tuple)}"
        if semi_b_key not in beam_res:
            def _make_beam_semi_job(_k=semi_b_key, _sd=sup_b_dir, _md=semi_b_dir, _bt=beams_tuple) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_fast_patch_dataset(train_408, img_cache, track_cache, 6, True, True, True, allowed_beams=_bt, seed=600 + gpu_id)
                    va_ds = make_fast_patch_dataset(val_408, img_cache, track_cache, 6, False, True, True, seed=42)
                    res = train_and_eval_semisup_single_gpu(
                        _k,
                        lambda: MultiTaskDINOv3(in_channels=8, base_ch=48, pretrained=False),
                        os.path.join(_sd, "best_model.pt"),
                        tr_ds,
                        va_ds,
                        _md,
                        gpu_id=gpu_id,
                        epochs=10,
                        batch_size=12,
                    )
                    beam_res[_k] = res
                    save_json(all_res, out_json)
                return _run
            semi_jobs.append(_make_beam_semi_job())

    # 6: Synoptic-Block Split (Semi-Supervised U-Net, SAM, DINOv3)
    for arch_label, arch_code in [("U-Net", "unet"), ("SAM", "sam"), ("DINOv3", "dinov3")]:
        m_key = f"Semi-Sup {arch_label} (Synoptic-Split 3f IR+VIS)"
        s_dir = f"outputs/cvpr_experiments/synoptic_sup_{arch_code}"
        m_dir = f"outputs/cvpr_experiments/synoptic_semisup_{arch_code}"
        if m_key not in syn_res:
            def _make_syn_semi_job(_k=m_key, _ac=arch_code, _sd=s_dir, _md=m_dir) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_fast_patch_dataset(syn_train, img_cache, track_cache, 6, True, True, True, seed=700 + gpu_id)
                    va_ds = make_fast_patch_dataset(syn_val, img_cache, track_cache, 6, False, True, True, seed=42)
                    res = train_and_eval_semisup_single_gpu(
                        _k,
                        lambda: build_model(arch=_ac, in_channels=8, base_ch=48, pretrained=False),
                        os.path.join(_sd, "best_model.pt"),
                        tr_ds,
                        va_ds,
                        _md,
                        gpu_id=gpu_id,
                        epochs=10,
                        batch_size=12,
                    )
                    syn_res[_k] = res
                    save_json(all_res, out_json)
                return _run
            semi_jobs.append(_make_syn_semi_job())

    print(f"\n=== WAVE 2: Launching {len(semi_jobs)} Semi-Supervised Ablation Jobs across 4 GPUs in parallel ===")
    run_jobs_across_4_gpus(semi_jobs)

    # =========================================================================
    # STEP 7: Diurnal (Day vs Night on 1f IR) & Seasonal Breakdown (DJF, MAM, JJA, SON)
    # =========================================================================
    if "diurnal_and_seasonal" not in all_res:
        print("\n=== Step 7: Computing Diurnal (Day vs Night) and Seasonal Breakdown ===")
        device = torch.device("cuda:0")
        day_490 = [
            g for g in val_490
            if 6 <= datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour <= 15
        ]
        night_490 = [
            g for g in val_490
            if datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour < 6
            or datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour > 15
        ]
        ds_day_1f = DataLoader(make_fast_patch_dataset(day_490, img_cache, track_cache, 10, False, False, False), batch_size=64, shuffle=False)
        ds_night_1f = DataLoader(make_fast_patch_dataset(night_490, img_cache, track_cache, 10, False, False, False), batch_size=64, shuffle=False)

        diurnal_seasonal: dict[str, Any] = {"day_count": len(day_490), "night_count": len(night_490), "models": {}}
        for label, arch, ckpt_p in [
            ("Semi-Sup U-Net (1f IR)", "unet", "outputs/semisup_unet/best_semisup.pt"),
            ("Semi-Sup SAM (1f IR)", "sam", "outputs/semisup_sam_1f/best_semisup.pt"),
            ("Semi-Sup DINOv3 (1f IR)", "dinov3", "outputs/semisup_dinov3_1f/best_semisup.pt"),
        ]:
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            base_ch = ckpt.get("args", {}).get("base_ch", 48)
            mod = build_model(arch=arch, in_channels=3, base_ch=base_ch, pretrained=False).to(device)
            mod.load_state_dict(ckpt.get("model_state_dict", ckpt.get("teacher_state", ckpt.get("model_state"))))
            diurnal_seasonal["models"][label] = {
                "Daytime (06-15 UTC)": evaluate_sparse_predictions(mod, ds_day_1f, device),
                "Night/Twilight (16-05 UTC)": evaluate_sparse_predictions(mod, ds_night_1f, device),
            }
            del mod
            torch.cuda.empty_cache()

        seasons = {
            "Winter (DJF)": (12, 1, 2),
            "Spring (MAM)": (3, 4, 5),
            "Summer (JJA)": (6, 7, 8),
            "Autumn (SON)": (9, 10, 11),
        }
        diurnal_seasonal["seasonal_3f"] = {}
        for s_name, months in seasons.items():
            s_grans = [
                g for g in val_408
                if datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).month in months
            ]
            if not s_grans:
                continue
            s_loader = DataLoader(make_fast_patch_dataset(s_grans, img_cache, track_cache, 6, False, True, True), batch_size=32, shuffle=False)
            diurnal_seasonal["seasonal_3f"][s_name] = {"n_granules": len(s_grans)}
            for label, arch, ckpt_p in [
                ("Semi-Sup SAM (3f IR+VIS)", "sam", "outputs/semisup_sam_3f_vis/best_semisup.pt"),
                ("Semi-Sup DINOv3 (3f IR+VIS)", "dinov3", "outputs/semisup_dinov3_3f_vis/best_semisup.pt"),
            ]:
                ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
                base_ch = ckpt.get("args", {}).get("base_ch", 48)
                mod = build_model(arch=arch, in_channels=8, base_ch=base_ch, pretrained=False).to(device)
                mod.load_state_dict(ckpt.get("model_state_dict", ckpt.get("teacher_state", ckpt.get("model_state"))))
                diurnal_seasonal["seasonal_3f"][s_name][label] = evaluate_sparse_predictions(mod, s_loader, device)
                del mod
                torch.cuda.empty_cache()

        all_res["diurnal_and_seasonal"] = diurnal_seasonal
        save_json(all_res, out_json)

    print("\n=== Generating All Paper Figures ===")
    import generate_paper_figures
    generate_paper_figures.main()
    print("\n=== ALL CVPR 2027 EXPERIMENTS & FIGURES COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()
