"""Round7 metrics per PREREG_R7: Mann-Whitney AUC, pooled/per-person,
S2 recovery, working-point ranges, H16-H19 checks. No literal thresholds
beyond the preregistered ones are baked into outputs; every number is
computed from runs/*/scores.csv.
"""
import numpy as np


def auc_mw(err_scores, cor_scores):
    """P(error score > correct score), ties 0.5."""
    e = np.asarray(err_scores, dtype=float)
    c = np.asarray(cor_scores, dtype=float)
    if len(e) == 0 or len(c) == 0:
        return float("nan")
    gt = (e[:, None] > c[None, :]).sum()
    eq = (e[:, None] == c[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(e) * len(c)))


def auc_of(rows):
    e = [r["score"] for r in rows if r["label"] == 1]
    c = [r["score"] for r in rows if r["label"] == 0]
    return auc_mw(e, c)


def rates(rows):
    """Return (tpr, fpr): fraction of error/correct rows with pred==1."""
    e = [r["pred"] for r in rows if r["label"] == 1]
    c = [r["pred"] for r in rows if r["label"] == 0]
    tpr = float(np.mean(e)) if e else float("nan")
    fpr = float(np.mean(c)) if c else float("nan")
    return tpr, fpr
