"""Train and evaluate the monotonic Cumulative Logit ordinal model on any run folder.

Rubric-agnostic: reads `features_hard.csv` and `features_soft.csv` produced by `agentrisk.run`
(alongside `data/split_manifest.jsonl` for grouped cross-validation and `data/train.jsonl` for
Subtask 2a/2b gold labels).

Usage:
  python -m agentrisk.model results/v1_binary_factors__qwen3-32b__train
  python -m agentrisk.model results/RUN_DIR --folds 5 --epochs 120
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from . import data as data_mod
from .rubric import REPO
from .taxonomy import DOMAINS, DRIVERS, NO_DOMAIN, NO_DRIVER


# ── Model ─────────────────────────────────────────────────────────────────────

class CumulativeLogitOrdinal(nn.Module):
    """Monotonic proportional-odds (cumulative logit) ordinal regression model.

    Because every rubric feature is phrased so YES = more risk (except UNK flags),
    non-negative weights w = softplus(w_raw) >= 0 enforce monotonicity: firing a
    hazard indicator can never lower the predicted severity score.
    """

    def __init__(self, num_features: int, num_classes: int = 5):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.w_raw = nn.Parameter(torch.zeros(num_features))
        self.theta_0 = nn.Parameter(torch.tensor(0.5))
        self.delta_thetas = nn.Parameter(torch.ones(num_classes - 2) * 0.5)

    def weights(self) -> torch.Tensor:
        return F.softplus(self.w_raw)

    def thresholds(self) -> torch.Tensor:
        thetas = [self.theta_0]
        for delta in self.delta_thetas:
            thetas.append(thetas[-1] + F.softplus(delta) + 1e-4)
        return torch.stack(thetas)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.weights()
        score = torch.matmul(x, w)
        thetas = self.thresholds()

        # P(y <= k | x) = sigmoid(theta_k - score)
        cum_probs = torch.sigmoid(thetas.unsqueeze(1) - score.unsqueeze(0)).T
        p1 = cum_probs[:, 0:1]
        p_mid = cum_probs[:, 1:] - cum_probs[:, :-1]
        p_last = 1.0 - cum_probs[:, -1:]
        probs = torch.cat([p1, p_mid, p_last], dim=1).clamp(min=1e-7, max=1.0)
        return probs, w, thetas

    @staticmethod
    def predict_median(probs: torch.Tensor) -> torch.Tensor:
        """Median of the discrete distribution over {1..K}: mathematically minimizes MAE."""
        cum_sum = torch.cumsum(probs, dim=1)
        return torch.argmax((cum_sum >= 0.5).int(), dim=1) + 1


def ordinal_nll_loss(probs: torch.Tensor, y_true: torch.Tensor, l2_reg: float = 0.0,
                     weights: torch.Tensor | None = None) -> torch.Tensor:
    idx = torch.arange(len(y_true))
    loss = -torch.log(probs[idx, y_true - 1] + 1e-7).mean()
    if l2_reg > 0.0 and weights is not None:
        loss = loss + l2_reg * (weights ** 2).mean()
    return loss


class MultiLabelHead(nn.Module):
    """Lightweight linear multi-label classifier on f(T) for Subtasks 2a (domains) & 2b (drivers)."""

    def __init__(self, num_features: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(num_features, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_feature_csv(path: Path) -> tuple[list[str], list[int], list[str], np.ndarray]:
    ids, gold, rows = [], [], []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        feat_names = header[2:]
        for r in reader:
            ids.append(r[0])
            gold.append(int(float(r[1])))
            rows.append([float(v) for v in r[2:]])
    return ids, gold, feat_names, np.asarray(rows, dtype=np.float32)


def load_scenario_groups(ids: list[str]) -> np.ndarray:
    manifest_path = data_mod.DATA / "split_manifest.jsonl"
    mapping = {}
    if manifest_path.exists():
        for line in manifest_path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                mapping[rec["example_id"]] = rec.get("scenario_group", rec["example_id"])
    return np.array([mapping.get(i, i) for i in ids])


def find_gate_column(feat_names: list[str]) -> int | None:
    """Identify the gate=YES column (always first if the rubric has a gate block)."""
    if feat_names and feat_names[0].startswith("G") and feat_names[0].endswith("=YES"):
        return 0
    return None


def mean_jaccard(y_true_sets: list[set[str]], y_pred_sets: list[set[str]]) -> float:
    scores = []
    for yt, yp in zip(y_true_sets, y_pred_sets):
        union = yt | yp
        scores.append(1.0 if not union else len(yt & yp) / len(union))
    return float(np.mean(scores)) if scores else 0.0


def train_ordinal_model(X_tr: np.ndarray, y_tr: np.ndarray, epochs: int = 120,
                        lr: float = 0.08, l2_reg: float = 1e-3, seed: int = 42) -> CumulativeLogitOrdinal:
    torch.manual_seed(seed)
    model = CumulativeLogitOrdinal(num_features=X_tr.shape[1], num_classes=5)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)

    for _ in range(epochs):
        model.train()
        probs, w, _ = model(xt)
        loss = ordinal_nll_loss(probs, yt, l2_reg=l2_reg, weights=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def predict_ordinal(model: CumulativeLogitOrdinal, X_va: np.ndarray,
                    gate_col: int | None = None, hard_gate: bool = True) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probs, _, _ = model(torch.tensor(X_va, dtype=torch.float32))
        preds = model.predict_median(probs).numpy()
    if hard_gate and gate_col is not None:
        closed = X_va[:, gate_col] < 0.5
        preds = np.where(closed, 1, preds)
    return preds


def evaluate_st1(X: np.ndarray, y: np.ndarray, groups: np.ndarray, feat_names: list[str],
                 n_splits: int = 5, hard_gate: bool = True, epochs: int = 120,
                 lr: float = 0.08) -> dict:
    """5-fold GroupKFold CV for Subtask 1 (MAE + per-level MAE + top learned weights)."""
    unique_groups = len(set(groups))
    k = min(n_splits, unique_groups)
    if k < 2:
        raise ValueError(f"Need at least 2 distinct scenario groups for CV (got {unique_groups}).")

    gkf = GroupKFold(n_splits=k)
    gate_col = find_gate_column(feat_names)

    b0_maes, fold_maes = [], []
    oof_preds = np.zeros_like(y)
    weights_accum = np.zeros(X.shape[1], dtype=np.float64)
    thetas_accum = np.zeros(4, dtype=np.float64)

    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # Baseline 0: class median
        med = int(np.median(y_tr))
        b0_maes.append(float(np.mean(np.abs(y_va - med))))

        # Ordinal model (for hard_gate, train only on open-gate rows if enough remain)
        if hard_gate and gate_col is not None:
            open_mask = X_tr[:, gate_col] >= 0.5
            if open_mask.sum() >= 10:
                model = train_ordinal_model(X_tr[open_mask], y_tr[open_mask], epochs=epochs, lr=lr)
            else:
                model = train_ordinal_model(X_tr, y_tr, epochs=epochs, lr=lr)
        else:
            model = train_ordinal_model(X_tr, y_tr, epochs=epochs, lr=lr)

        preds = predict_ordinal(model, X_va, gate_col=gate_col, hard_gate=hard_gate)
        oof_preds[va_idx] = preds
        fold_maes.append(float(np.mean(np.abs(y_va - preds))))

        with torch.no_grad():
            weights_accum += model.weights().numpy()
            thetas_accum += model.thresholds().numpy()

    mean_w = weights_accum / k
    mean_thetas = (thetas_accum / k).round(3).tolist()

    per_level = {}
    for lvl in range(1, 6):
        mask = y == lvl
        if mask.any():
            per_level[f"level_{lvl}"] = {
                "n": int(mask.sum()),
                "mae": round(float(np.mean(np.abs(oof_preds[mask] - y[mask]))), 3),
            }

    top_weights = sorted(
        [(feat_names[i], round(float(mean_w[i]), 3)) for i in range(len(feat_names))],
        key=lambda kv: kv[1],
        reverse=True,
    )[:10]

    return {
        "folds": k,
        "hard_gate": hard_gate,
        "baseline_median_mae": round(float(np.mean(b0_maes)), 4),
        "cv_mae": round(float(np.mean(fold_maes)), 4),
        "cv_mae_std": round(float(np.std(fold_maes)), 4),
        "oof_mae": round(float(np.mean(np.abs(oof_preds - y))), 4),
        "per_level_mae": per_level,
        "mean_thresholds": mean_thetas,
        "top_weights": top_weights,
    }


def evaluate_st2(X: np.ndarray, ids: list[str], groups: np.ndarray, feat_names: list[str],
                 split: str = "train", n_splits: int = 5, epochs: int = 120) -> dict | None:
    """5-fold GroupKFold CV for Subtask 2a (Harm Domains) & 2b (Risk Drivers) using Mean Jaccard."""
    split_path = data_mod.DATA / f"{split}.jsonl"
    if not split_path.exists():
        return None
    by_id = {r["example_id"]: r for r in data_mod.load(split)}
    if not all(i in by_id for i in ids):
        return None

    gate_col = find_gate_column(feat_names)
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    out = {}

    for key, vocab, empty_label in [
        ("subtask_2a_domains", DOMAINS, NO_DOMAIN),
        ("subtask_2b_drivers", DRIVERS, NO_DRIVER),
    ]:
        field = "risk_domains" if "domains" in key else "risk_drivers"
        Y = np.zeros((len(ids), len(vocab)), dtype=np.float32)
        gold_sets = []
        for idx, eid in enumerate(ids):
            labs = set(by_id[eid].get(field) or [])
            gold_sets.append(labs)
            for j, name in enumerate(vocab):
                if name in labs:
                    Y[idx, j] = 1.0

        oof_sets: list[set[str]] = [set() for _ in ids]
        for tr_idx, va_idx in gkf.split(X, Y[:, 0], groups):
            torch.manual_seed(42)
            head = MultiLabelHead(X.shape[1], len(vocab))
            opt = torch.optim.Adam(head.parameters(), lr=0.05, weight_decay=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            xt = torch.tensor(X[tr_idx], dtype=torch.float32)
            yt = torch.tensor(Y[tr_idx], dtype=torch.float32)
            for _ in range(epochs):
                opt.zero_grad()
                loss = loss_fn(head(xt), yt)
                loss.backward()
                opt.step()

            head.eval()
            with torch.no_grad():
                probs = torch.sigmoid(head(torch.tensor(X[va_idx], dtype=torch.float32))).numpy()
            for local_i, global_i in enumerate(va_idx):
                if gate_col is not None and X[global_i, gate_col] < 0.5:
                    oof_sets[global_i] = {empty_label}
                else:
                    picked = {vocab[j] for j in range(len(vocab)) if probs[local_i, j] >= 0.4}
                    if not picked:
                        picked = {vocab[int(np.argmax(probs[local_i]))]}
                    oof_sets[global_i] = picked

        out[key] = {"mean_jaccard": round(mean_jaccard(gold_sets, oof_sets), 4)}
    return out


def evaluate_run(run_dir: Path, n_splits: int = 5, epochs: int = 120, lr: float = 0.08) -> dict:
    meta_path = run_dir / "meta.json"
    split = json.loads(meta_path.read_text(encoding="utf-8")).get("split", "train") if meta_path.exists() else "train"

    report: dict = {"run": run_dir.name, "split": split}
    for mode, fname in [("hard_features", "features_hard.csv"), ("soft_features", "features_soft.csv")]:
        csv_path = run_dir / fname
        if not csv_path.exists():
            continue
        ids, gold, feat_names, X = load_feature_csv(csv_path)
        y = np.asarray(gold, dtype=np.int64)
        groups = load_scenario_groups(ids)
        report[mode] = {
            "n_samples": len(ids),
            "n_features": X.shape[1],
            "st1_hard_gate": evaluate_st1(X, y, groups, feat_names, n_splits=n_splits,
                                          hard_gate=True, epochs=epochs, lr=lr),
            "st1_soft_gate": evaluate_st1(X, y, groups, feat_names, n_splits=n_splits,
                                          hard_gate=False, epochs=epochs, lr=lr),
        }
        st2 = evaluate_st2(X, ids, groups, feat_names, split=split, n_splits=n_splits, epochs=epochs)
        if st2:
            report[mode]["st2"] = st2

    (run_dir / "model_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Run folder containing features_hard.csv / features_soft.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.08)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    run_dir = args.run_dir if args.run_dir.is_absolute() or args.run_dir.exists() else REPO / args.run_dir
    rep = evaluate_run(run_dir, n_splits=args.folds, epochs=args.epochs, lr=args.lr)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
