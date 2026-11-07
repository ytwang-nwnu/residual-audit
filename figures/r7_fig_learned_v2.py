# -*- coding: utf-8 -*-
"""正文图 v2：五个学习式检测器的留一人审计（round7）。

(a) 逐方法逐动作 ΔAUC 分组柱（含 D2 距离打分器参照与逐 seed 点）
(b) 深蹲逐人误报率矩阵：左格留一人、右格个人校准，行=学员、列=方法
(c) 失效格上 k 条本人正确重复的回收曲线

只从落盘文件读数。用法：
  EXP=... FIG_DIR=... uv run --with matplotlib --with numpy python r7_fig_learned_v2.py
输出：$FIG_DIR/fig_learned.pdf 与 .png
"""
import csv, json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get("EXP", os.path.join(HERE, "..", "exp"))
R7 = os.path.join(E, "prl_v2", "round7")
OUT_DIR = os.environ.get("FIG_DIR", os.path.join(HERE, "..", "figures", "out"))

ALL_METHODS = [("m1", "Zhao'22"), ("m2", "Deb'22"), ("m3p", "Zheng'23"),
               ("m4", "LITEMV"), ("m5", "ConvTran")]
ACTS = [("SQUAT", "Squat"), ("Lunges", "Lunge"), ("Plank", "Plank")]   # 按 ρ 从大到小
# 与正文图一致的柔和配色；D2 为灰
COL = {"d2": "0.62", "m1": "#4C78A8", "m2": "#E0823D", "m3p": "#54A24B",
       "m4": "#B55A60", "m5": "#8172B2"}
METHODS = [(m, lab) for m, lab in ALL_METHODS
           if os.path.exists(os.path.join(R7, m, "metrics.json"))]
THR_DELTA, THR_REC = 0.05, 0.8

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def load():
    met = {m: json.load(open(os.path.join(R7, m, "metrics.json"))) for m, _ in METHODS}
    per = {m: list(csv.DictReader(open(os.path.join(R7, m, "per_person.csv")))) for m, _ in METHODS}
    rec = {m: list(csv.DictReader(open(os.path.join(R7, m, "recovery.csv")))) for m, _ in METHODS}
    d2 = {}
    for r in csv.DictReader(open(os.path.join(E, "prl_v2", "round3", "figs", "deploy_table.csv"))):
        if r["dataset"] == "EC3D" and r["ruler"] == "positions":
            d2[r["act"]] = float(r["delta"])
    return met, per, rec, d2


def tag(ax, s):
    ax.text(-0.06, 1.07, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right")


def panel_a(ax, met, d2):
    x = np.arange(len(ACTS))
    series = [("d2", "D2 (dist.)")] + METHODS
    wbar = 0.82 / len(series)
    for i, (m, label) in enumerate(series):
        xs = x - 0.41 + wbar * (i + 0.5)
        if m == "d2":
            ax.bar(xs, [d2[a] for a, _ in ACTS], wbar * 0.92, color=COL["d2"], zorder=2)
            continue
        ys = [met[m]["acts"][a]["delta_auc_L"] for a, _ in ACTS]
        ax.bar(xs, ys, wbar * 0.92, color=COL[m], zorder=2)
        for xi, a in zip(xs, [a for a, _ in ACTS]):
            seeds = list(met[m]["acts"][a]["delta_auc_L_per_seed"].values())
            if len(seeds) > 1:
                ax.plot([xi] * len(seeds), seeds, "o", ms=2.2, mfc="white",
                        mec="0.35", mew=0.5, zorder=3)
    ax.axhline(THR_DELTA, ls="--", lw=0.7, color="0.3", zorder=1)
    ax.text(len(ACTS) - 0.45, THR_DELTA + 0.008, "0.05", fontsize=6.5, ha="right", va="bottom")
    ax.axhline(0, lw=0.6, color="k")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in ACTS])
    ax.set_ylabel(r"$\Delta$AUC, calibrated $-$ LOPO")
    ax.set_xlim(-0.55, len(ACTS) - 0.45)
    ax.spines[["top", "right"]].set_visible(False)
    tag(ax, "(a)")


def fmt(v):
    if v >= 0.995: return "1"
    if v < 0.005:  return "0"
    return ("%.2f" % v).lstrip("0")


def panel_b(ax, per, act="SQUAT"):
    # 逐人 × 逐方法的误报率，左格留一人（s1），右格校准（s2_all）
    persons = sorted({r["person"] for m, _ in METHODS for r in per[m]})
    fpr = {}
    for m, _ in METHODS:
        for p in persons:
            for sp in ("s1", "s2_all"):
                v = [float(r["fpr"]) for r in per[m]
                     if r["act"] == act and r["person"] == p and r["split"] == sp]
                fpr[(m, p, sp)] = np.mean(v) if v else np.nan
    persons = sorted(persons, key=lambda p: -np.nanmean([fpr[(m, p, "s1")] for m, _ in METHODS]))
    nm, np_, gap = len(METHODS), len(persons), 0.9
    HEAD = {"m1": "Zhao", "m2": "Deb", "m3p": "Zheng", "m4": "LITEMV", "m5": "ConvTran"}
    for blk, (sp, blab) in enumerate([("s1", "LOPO"), ("s2_all", "calibrated")]):
        x0 = blk * (nm + gap)
        ax.text(x0 + nm / 2 - 0.5, -0.8, blab, ha="center", va="top", fontsize=7.5)
        for j, (m, lab) in enumerate(METHODS):
            ax.text(x0 + j, np_ - 0.02, HEAD[m], rotation=90, ha="center", va="bottom", fontsize=5.8)
            for i, p in enumerate(persons):
                v = fpr[(m, p, sp)]
                y = np_ - 1 - i
                c = plt.cm.Reds(0.12 + 0.88 * v) if v == v else "white"
                ax.add_patch(Rectangle((x0 + j - 0.5, y - 0.5), 1, 1,
                                       facecolor=c, edgecolor="0.85", lw=0.4, zorder=2))
                if v == v:
                    ax.text(x0 + j, y, fmt(v), ha="center", va="center", fontsize=7.0,
                            color="white" if v > 0.55 else "0.25", zorder=3)
    for i, p in enumerate(persons):
        ax.text(-0.6, np_ - 1 - i, p, ha="right", va="center", fontsize=7)
    ax.set_xlim(-2.35, 2 * nm + gap - 0.5)
    ax.set_ylim(-1.7, np_ + 1.95)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 0.985, "(b)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="top")


def panel_c(ax, met, rec):
    ks = sorted({int(r["k"]) for m, _ in METHODS for r in rec[m]})
    styles = {"SQUAT": "-", "Lunges": "--", "Plank": ":"}
    for m, label in METHODS:
        for a, _ in ACTS:
            if met[m]["acts"][a]["delta_auc_L"] < THR_DELTA:
                continue
            ys = []
            for k in ks:
                v = [float(r["recovery"]) for r in rec[m]
                     if r["act"] == a and int(r["k"]) == k and float(r["recovery"]) == float(r["recovery"])]
                ys.append(np.mean(v))
            ax.plot(ks, ys, styles[a], color=COL[m], marker="o", ms=3, lw=1.1, mew=0.7)
    ax.axhline(THR_REC, ls="--", lw=0.7, color="0.3")
    ax.text(ks[-1], THR_REC + 0.03, "0.8", fontsize=6.5, ha="right", va="bottom", color="0.3")
    ax.axhline(0, lw=0.6, color="k")
    ax.set_xticks(ks); ax.set_xlabel("own correct repetitions $k$")
    ax.set_ylabel("recovery $R_k$")
    ax.set_ylim(-0.05, 1.4)
    hs = [Line2D([0], [0], color="k", ls=styles[a], lw=1.1) for a, _ in ACTS]
    ax.legend(hs, [lab for _, lab in ACTS], frameon=False, loc=(0.62, 0.02), handlelength=2.2)
    ax.spines[["top", "right"]].set_visible(False)
    tag(ax, "(c)")


def main():
    met, per, rec, d2 = load()
    fig = plt.figure(figsize=(7.0, 3.0))
    axa = fig.add_axes([0.055, 0.185, 0.235, 0.68])
    axb = fig.add_axes([0.305, 0.10, 0.42, 0.80])
    axc = fig.add_axes([0.795, 0.185, 0.195, 0.68])
    panel_a(axa, met, d2)
    panel_b(axb, per)
    panel_c(axc, met, rec)
    hs = [Patch(color=COL["d2"], label="D2 (distance scorer)")] + \
         [Patch(color=COL[m], label=lab) for m, lab in METHODS]
    fig.legend(handles=hs, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.04), handlelength=1.1, columnspacing=0.9)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, "fig_learned." + ext), dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(OUT_DIR, "fig_learned.pdf"))


if __name__ == "__main__":
    main()
