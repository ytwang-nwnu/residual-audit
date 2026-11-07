"""Figures and GAITEX person summaries from files already under exp/prl_v2. No new experiments."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

sys.path.insert(0, "/data1/users/yangrui/projects/idea67b/experiment/code/prl_v2")
import common as C

ROOT = "/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2"
OUT = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), mean=float(np.mean(x)), median=float(np.median(x)),
                minimum=float(np.min(x)), maximum=float(np.max(x)))


def gaitex_table():
    rows = []
    for ses, fn in (("rd", "round2/r4/per_person_rd.csv"), ("rgs", "round2/r4/per_person_rgs.csv")):
        df = pd.read_csv(os.path.join(ROOT, fn))
        for ref, g in df.groupby("ref"):
            if ref not in ("O", "D2"):
                continue
            for _, r in g.iterrows():
                rows.append(dict(session=ses, ref=ref, newcomer=r.newcomer, n_correct=int(r.n_correct),
                                 n_error=int(r.n_error), fpr=float(r.fpr), tpr=float(r.tpr)))
            for col in ("fpr", "tpr"):
                rec = dict(session=ses, ref=ref, newcomer="ALL", column=col)
                rec.update(stats(g[col]))
                rows.append(rec)
    return pd.DataFrame(rows)


def eight_points():
    rho_pos = pd.read_csv(os.path.join(ROOT, "e1/rho_realign.csv"))
    e1 = json.load(open(os.path.join(ROOT, "e1/prereg_check.json")))
    rho_ang = pd.read_csv(os.path.join(ROOT, "round2/r2/rho.csv"))
    r2 = json.load(open(os.path.join(ROOT, "round2/r2/prereg_check.json")))
    r4 = json.load(open(os.path.join(ROOT, "round2/r4/prereg_check.json")))
    r4s = json.load(open(os.path.join(ROOT, "round2/r4/selfcheck.json")))
    rows = []
    for r in e1["deltas"]:
        if r["view"] != "positions":
            continue
        rho = float(rho_pos[(rho_pos.view == "positions") & (rho_pos.act == r["act"])].rho.iloc[0])
        rows.append(dict(group="EC3D position", act=r["act"], rho=rho, delta_auc=float(r["delta"])))
    for r in r2["deltas"]:
        if r["view"] != "bone angles":
            continue
        hit = rho_ang[(rho_ang.act == r["act"]) & (rho_ang.view == "bone angles, all joints")]
        rows.append(dict(group="EC3D 27-pair angle", act=r["act"], rho=float(hit.ratio.iloc[0]),
                         delta_auc=float(r["delta"])))
    published = {d["session"]: float(d["ratio"]) for d in r4s["diffs"]}
    for ses, delta in r4["delta"].items():
        rows.append(dict(group="GAITEX", act=ses, rho=published[ses], delta_auc=float(delta)))
    return pd.DataFrame(rows)


def main_figure(points):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = {"EC3D position": "o", "EC3D 27-pair angle": "s", "GAITEX": "^"}
    for name, g in points.groupby("group", sort=False):
        ax.scatter(g.rho, g.delta_auc, marker=markers[name], s=46, label=name)
        for _, r in g.iterrows():
            ax.annotate(r.act, (r.rho, r.delta_auc), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ρ")
    ax.set_ylabel("ΔAUC")
    ax.legend(frameon=False)
    fig.text(0.01, -0.02,
             "各数据集内部的排序是预注册检验。8 点合在一起是事后描述。EC3D 的 6 个点来自同一批 4 人。",
             fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rho_delta.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "rho_delta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def gaitex_figure():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, ses in zip(axes, ("rd", "rgs")):
        df = pd.read_csv(os.path.join(ROOT, "round2/r4/per_person_%s.csv" % ses))
        for i, ref in enumerate(("O", "D2")):
            sub = df[df.ref == ref]
            x = i + rng.uniform(-0.08, 0.08, len(sub))
            ax.scatter(x, sub.fpr, s=16)
        ax.axhline(0.10, ls="--", color="0.4", lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["O", "D2"])
        ax.set_title(ses)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("实际误报率")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gaitex_fpr.svg"))
    fig.savefig(os.path.join(OUT, "gaitex_fpr.png"), dpi=150)
    plt.close(fig)


def main():
    t0 = C.now_iso()
    t_start = time.time()
    people = gaitex_table()
    people.to_csv(os.path.join(OUT, "gaitex_person.csv"), index=False)
    points = eight_points()
    points.to_csv(os.path.join(OUT, "rho_delta.csv"), index=False)
    main_figure(points)
    gaitex_figure()
    t1 = C.now_iso()
    outs = [os.path.join(OUT, f) for f in
            ["gaitex_person.csv", "rho_delta.csv", "rho_delta.svg", "rho_delta.png",
             "gaitex_fpr.svg", "gaitex_fpr.png"]]
    C.update_manifest(os.path.abspath(__file__), outs, t0, t1)
    print("FIGS", round(time.time() - t_start, 3), "points", len(points))


if __name__ == "__main__":
    main()
