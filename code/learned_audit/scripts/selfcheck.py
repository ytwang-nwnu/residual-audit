"""Round7 data self-check: re-derive the 371 EC3D segments via the shared
reading path and compare per-(act, person) correct/error counts against e1's
scores_long.csv / per_person.csv / pooled.csv. Writes selfcheck_data.json.
"""
import json, os, sys
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ec3d  # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "common.yaml")
E1 = "/data1/users/yangrui/results/idea67b/exp/prl_v2/e1"
OUT = "/data1/users/yangrui/results/idea67b/exp/prl_v2/round7/selfcheck_data.json"


def main():
    cfg = yaml.safe_load(open(CFG))
    segs = ec3d.load_segments(cfg["data_pickle"])
    got_total = len(segs)
    got_cor = sum(1 for s in segs if s["err"] == 0)
    got_err = sum(1 for s in segs if s["err"] == 1)

    mine = {}
    for s in segs:
        d = mine.setdefault((s["act"], s["sub"]), {"correct": 0, "error": 0})
        d["error" if s["err"] else "correct"] += 1

    pp = pd.read_csv(os.path.join(E1, "per_person.csv"))
    diffs = []
    for (act, sub), g in pp.groupby(["act", "newcomer"]):
        for _, r in g.iterrows():
            m = mine.get((act, sub))
            if m is None or m["correct"] != int(r["n_correct"]) or m["error"] != int(r["n_error"]):
                diffs.append(dict(act=act, sub=sub, ref=r["ref"], view=r["view"],
                                  e1_correct=int(r["n_correct"]), e1_error=int(r["n_error"]),
                                  mine=None if m is None else m["correct"],
                                  mine_err=None if m is None else m["error"]))
    # e1 covers every act x person exactly once per (view, ref); check coverage both ways
    missing = [dict(act=a, sub=s, **v) for (a, s), v in mine.items()
               if ((pp.act == a) & (pp.newcomer == s)).sum() == 0]

    pooled = pd.read_csv(os.path.join(E1, "pooled.csv"))
    # ref O and D2 count raw segments; ref D1 counts rep x demonstrator
    # pairs (3x segments), so only O and D2 rows are comparable here.
    pooled = pooled[pooled.ref.isin(["O", "D2"])]
    act_diffs = []
    for _, r in pooled.iterrows():
        e1_cor, e1_err = int(r["n_correct"]), int(r["n_error"])
        m_cor = sum(v["correct"] for (a, _), v in mine.items() if a == r["act"])
        m_err = sum(v["error"] for (a, _), v in mine.items() if a == r["act"])
        if m_cor != e1_cor or m_err != e1_err:
            act_diffs.append(dict(act=r["act"], view=r["view"], ref=r["ref"],
                                  e1_correct=e1_cor, e1_error=e1_err,
                                  mine_correct=m_cor, mine_error=m_err))

    passed = (got_total == 371 and got_cor == 120 and got_err == 251
              and not diffs and not missing and not act_diffs)
    out = dict(
        passed=passed,
        expected=dict(total=371, correct=120, error=251,
                      note="9 段标签未知的深蹲按错误计"),
        got=dict(total=got_total, correct=got_cor, error=got_err),
        per_person_diffs=diffs,
        missing_in_e1=missing,
        per_act_diffs=act_diffs,
        seg_table={"{a}|{s}".format(a=a, s=s): v for (a, s), v in sorted(mine.items())},
    )
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("passed:", passed)
    print("got:", out["got"])
    if diffs or missing or act_diffs:
        print("diffs:", diffs[:5], "missing:", missing, "act_diffs:", act_diffs)


if __name__ == "__main__":
    main()
