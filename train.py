"""
train.py
========
End-to-end training script for DVHnet. Assumes slice-level examples have
already been cached to disk as per-patient .npz shards via
preprocessing.build_slice_dataset + dataset.save_patient_shard.

Usage:
    python train.py --shard_dir /path/to/shards --oar Parotid_L --epochs 100
"""

from __future__ import annotations

import argparse
import os
import json
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel

from dataset import DVHSliceDataset, collate_fn, load_examples_from_shards
from model import DVHNet, enforce_monotonic
from losses import DVHLoss
from preprocessing import split_patients
from aggregate import aggregate_from_batch_outputs
from metrics import evaluate_patient, summarize_results, check_acceptance_benchmarks


def get_patient_ids(shard_dir: str):
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(shard_dir) if f.endswith(".npz")
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "loss_fidelity": 0.0, "loss_clinical": 0.0, "loss_mono": 0.0}
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)

            pred = model(x)
            losses = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    """Recompute BatchNorm running statistics for a weight-averaged (SWA) model.
    torch.optim.swa_utils.update_bn assumes the loader yields tensors or
    tuples; ours yields dicts, so this does the same job by hand."""
    bn_mods = [m for m in model.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_mods:
        return
    saved_momentum = {}
    for m in bn_mods:
        m.reset_running_stats()
        saved_momentum[m] = m.momentum
        m.momentum = None  # cumulative moving average over the pass
    was_training = model.training
    model.train()
    for batch in loader:
        model(batch["input"].to(device))
    for m in bn_mods:
        m.momentum = saved_momentum[m]
    model.train(was_training)


@torch.no_grad()
def evaluate(model, loader, device, dose_max_gy: float):
    model.eval()
    all_preds, all_labels, all_pids, all_slices, all_voxels = [], [], [], [], []

    for batch in loader:
        x = batch["input"].to(device)
        pred = enforce_monotonic(model(x)).cpu().numpy()

        all_preds.append(pred)
        all_labels.append(batch["label"].numpy())
        all_pids.extend(batch["patient_id"])
        all_slices.extend(batch["slice_index"])
        all_voxels.append(batch["voxel_count"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    voxels = np.concatenate(all_voxels, axis=0)

    pred_dvhs = aggregate_from_batch_outputs(all_pids, all_slices, preds, voxels)
    true_dvhs = aggregate_from_batch_outputs(all_pids, all_slices, labels, voxels)

    results = [
        evaluate_patient(pid, "OAR", pred_dvhs[pid], true_dvhs[pid], dose_max_gy)
        for pid in pred_dvhs
    ]
    summary = summarize_results(results)
    summary["acceptance"] = check_acceptance_benchmarks(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True, help="Directory of per-patient .npz shards")
    parser.add_argument("--oar", required=True, help="OAR name (used only for logging/output naming)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=12,
                        help="early stop if the moving-average val loss doesn't "
                             "improve for this many epochs (0 disables)")
    parser.add_argument("--val_ma_window", type=int, default=5,
                        help="window for the moving-average val loss used for "
                             "checkpoint selection and early stopping")
    parser.add_argument("--swa_start_frac", type=float, default=0.5,
                        help="begin SWA weight averaging after this fraction of "
                             "epochs (1.0 disables SWA)")
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--dose_max_gy", type=float, default=80.0)
    parser.add_argument("--lambda_clinical", type=float, default=0.5)
    parser.add_argument("--lambda_mono", type=float, default=0.1)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="./runs/dvhnet")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # Patient-level split first -- never split by slice, to avoid leakage
    # from spatially-adjacent slices ending up on both sides of the split.
    patient_ids = get_patient_ids(args.shard_dir)
    splits = split_patients(patient_ids, args.train_frac, args.val_frac, seed=args.seed)

    train_examples = load_examples_from_shards(args.shard_dir, splits["train"])
    val_examples = load_examples_from_shards(args.shard_dir, splits["val"])
    test_examples = load_examples_from_shards(args.shard_dir, splits["test"])

    train_ds = DVHSliceDataset(train_examples, augment=True)
    val_ds = DVHSliceDataset(val_examples, augment=False)
    test_ds = DVHSliceDataset(test_examples, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    model = DVHNet(in_channels=2, num_bins=args.num_bins, dropout=args.dropout).to(device)
    criterion = DVHLoss(num_bins=args.num_bins,
                         lambda_clinical=args.lambda_clinical,
                         lambda_mono=args.lambda_mono).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_path = os.path.join(args.out_dir, f"dvhnet_{args.oar}_best.pt")
    swa_path = os.path.join(args.out_dir, f"dvhnet_{args.oar}_swa.pt")

    # Checkpoint selection and early stopping run off a moving average of the
    # val loss, not the single-epoch value: on a 50-patient val set the raw
    # curve is noisy enough (bounces ~0.10-0.15) that the single best epoch is
    # often just a lucky dip.
    best_val_ma = float("inf")
    val_ma = deque(maxlen=args.val_ma_window)
    epochs_since_improve = 0

    swa_start = int(args.swa_start_frac * args.epochs)
    swa_model = AveragedModel(model)
    swa_n = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        if args.swa_start_frac < 1.0 and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_n += 1

        val_ma.append(val_stats["loss"])
        val_ma_now = float(np.mean(val_ma))

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats,
                        "val_loss_ma": val_ma_now})
        print(f"[epoch {epoch:03d}] "
              f"train_loss={train_stats['loss']:.4f} "
              f"(fid={train_stats['loss_fidelity']:.4f} "
              f"clin={train_stats['loss_clinical']:.4f} "
              f"mono={train_stats['loss_mono']:.4f}) "
              f"val_loss={val_stats['loss']:.4f} "
              f"(fid={val_stats['loss_fidelity']:.4f} "
              f"clin={val_stats['loss_clinical']:.4f} "
              f"mono={val_stats['loss_mono']:.4f}) "
              f"val_ma={val_ma_now:.4f}")

        if val_ma_now < best_val_ma - 1e-5:
            best_val_ma = val_ma_now
            epochs_since_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_since_improve += 1
            if args.patience > 0 and epochs_since_improve >= args.patience:
                print(f"[early stop] no val_ma improvement for {args.patience} "
                      f"epochs (best val_ma={best_val_ma:.4f})")
                break

    with open(os.path.join(args.out_dir, f"history_{args.oar}.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Two candidate models: the moving-average-selected checkpoint, and the SWA
    # weight average over the back half of training. Pick whichever has the
    # lower raw val loss, then run the patient-level test evaluation once.
    candidates = []

    model.load_state_dict(torch.load(best_path, map_location=device))
    best_val = run_epoch(model, val_loader, criterion, device, optimizer=None)["loss"]
    candidates.append(("best_ckpt", model, best_val))

    if swa_n > 0:
        recalibrate_bn(swa_model, train_loader, device)
        swa_val = run_epoch(swa_model, val_loader, criterion, device, optimizer=None)["loss"]
        torch.save(swa_model.module.state_dict(), swa_path)
        candidates.append(("swa", swa_model, swa_val))

    selected_name, selected_model, selected_val = min(candidates, key=lambda c: c[2])
    print(f"model selection: " +
          ", ".join(f"{n}={v:.4f}" for n, _, v in candidates) +
          f"  ->  {selected_name}")

    test_summary = evaluate(selected_model, test_loader, device, args.dose_max_gy)
    test_summary["selected_model"] = selected_name
    test_summary["val_loss_by_candidate"] = {n: v for n, _, v in candidates}
    print("Test summary:", json.dumps(test_summary, indent=2))

    with open(os.path.join(args.out_dir, f"test_summary_{args.oar}.json"), "w") as f:
        json.dump(test_summary, f, indent=2)


if __name__ == "__main__":
    main()
