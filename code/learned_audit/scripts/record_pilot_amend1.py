"""Record the PREREG_R7_amend1 pilot measurement and budget rule in pilot.json.

Adds (never removes) two keys to exp/prl_v2/round7/pilot.json:
  measured_m4          : resource.json of the M4 S1 fold 0 seed 0 run unit (the
                         pilot unit named by the amendment) + per-model fit
                         seconds parsed from its log.txt, + the measurement
                         conditions (how many run_split processes shared the GPU).
  m4m5_budget_decision : the amendment's rule (M4+M5 projected > 24 GPU-hours
                         -> seeds {0}) re-evaluated from the measured unit times
                         (m4 from the pilot unit, m5 from the first finished m5
                         unit), the number of run units per seed count, and the
                         seeds actually run; records explicitly that the launch
                         decision (seeds {0}) was taken before the measured unit
                         finished, on a 50-epoch extrapolation (pilot.json key
                         'amend1'), and whether the measured numbers agree.
Every number is read from files; nothing is typed in by hand.
"""
import glob, json, os, re, sys

RES = "/data1/users/yangrui/results/idea67b/exp/prl_v2/round7"
BUDGET_HOURS = 24.0          # PREREG_R7 / amend1 budget rule
UNITS_PER_SEED = {"m4": 8, "m5": 8}   # (s1 + s2) x 4 folds
SEEDS_FULL = [0, 1, 2]
FIT_RE = re.compile(r"\] (m[45]) (\w+) init (\d)/(\d) seed (\d+): (\{.*\})$")


def unit_dirs(method, split, fold, seed):
    pat = os.path.join(RES, "runs", "*_{}_{}_fold{}_seed{}".format(method, split, fold, seed))
    return sorted(d for d in glob.glob(pat) if os.path.exists(os.path.join(d, "scores.csv")))


def fit_infos(run_dir):
    out = []
    for line in open(os.path.join(run_dir, "log.txt"), errors="replace"):
        m = FIT_RE.search(line.rstrip("\n"))
        if m:
            try:
                d = eval(m.group(6), {"__builtins__": {}}, {})  # dict literal written by drp_bench.log
            except Exception:
                continue
            out.append(dict(action=m.group(2), init=int(m.group(3)), sub_seed=int(m.group(5)), **d))
    return out


def unit_measure(run_dir):
    res = json.load(open(os.path.join(run_dir, "resource.json")))
    fits = fit_infos(run_dir)
    secs = [f["fit_sec"] for f in fits if "fit_sec" in f]
    return dict(run=os.path.basename(run_dir), **res,
                n_models=len(fits), fit_sec_sum=sum(secs),
                fit_sec_mean=(sum(secs) / len(secs) if secs else None),
                epochs_run=sorted({f.get("epochs_run") for f in fits}),
                best_epoch_min_max=([min(f["best_epoch"] for f in fits), max(f["best_epoch"] for f in fits)]
                                    if fits and all("best_epoch" in f for f in fits) else None))


def main():
    pj = os.path.join(RES, "pilot.json")
    pilot = json.load(open(pj))
    m4_dirs = unit_dirs("m4", "s1", 0, 0)
    if not m4_dirs:
        print("pilot unit m4 s1 fold0 seed0 not finished yet"); return 1
    m4 = unit_measure(m4_dirs[-1])
    # first finished m5 unit (any split/fold, seed 0) by resource mtime
    m5_all = []
    for split in ("s1", "s2"):
        for fold in range(4):
            m5_all += unit_dirs("m5", split, fold, 0)
    m5 = None
    if m5_all:
        m5_all.sort(key=lambda d: os.path.getmtime(os.path.join(d, "resource.json")))
        m5 = unit_measure(m5_all[0])
    conds = ("measured while several run_split processes shared the V100s (see log timestamps / "
             "SESSION_STATE_amend1.txt); contended, so per-unit time is an upper bound of the "
             "uncontended time")
    pilot["measured_m4"] = dict(unit="M4 S1 fold 0 seed 0 (3 actions x 5 initializations, 1500 epochs each)",
                                conditions=conds, **m4)
    proj = {}
    for name, meas in (("m4", m4), ("m5", m5)):
        if meas is None:
            proj[name] = None
            continue
        h = meas["elapsed_sec"] / 3600.0
        proj[name] = dict(unit_hours=h, units_per_seed=UNITS_PER_SEED[name],
                          hours_seeds3=h * UNITS_PER_SEED[name] * len(SEEDS_FULL),
                          hours_seeds1=h * UNITS_PER_SEED[name])
    tot3 = sum(p["hours_seeds3"] for p in proj.values() if p)
    tot1 = sum(p["hours_seeds1"] for p in proj.values() if p)
    seeds_run = sorted({int(re.search(r"_seed(\d+)$", d).group(1))
                        for d in glob.glob(os.path.join(RES, "runs", "*_m[45]_s[12]_fold[0-3]_seed[0-9]"))
                        if os.path.exists(os.path.join(d, "scores.csv"))})
    rule_seeds = [0] if tot3 > BUDGET_HOURS else SEEDS_FULL
    prior = pilot.get("amend1", {}).get("decision")
    pilot["m4m5_budget_decision"] = dict(
        rule="PREREG_R7_amend1: M4 + M5 projected > {:.0f} GPU-hours -> seeds reduced to {{0}}, else {{0,1,2}}".format(BUDGET_HOURS),
        measured_projection=dict(per_method=proj, m5_measured_from=(m5["run"] if m5 else None),
                                 total_hours_seeds3=tot3, total_hours_seeds1=tot1,
                                 m5_missing=(m5 is None)),
        rule_outcome_from_measured=dict(seeds=rule_seeds),
        launch_decision=dict(seeds=[0], basis=("taken before the measured pilot unit finished, from the "
                                                "50-epoch extrapolation recorded under pilot.json['amend1'] "
                                                "(smoke_amend1/timing.json)"), prior=prior),
        consistent=(rule_seeds == [0]),
        seeds_with_finished_runs=seeds_run,
        note=("GPU-hours = run units x measured unit wall time, as for the M1/M2 decisions; measured under "
              "GPU sharing. If 'consistent' were false the matrix would have to be extended to seeds 1, 2."))
    json.dump(pilot, open(pj, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"measured_m4": pilot["measured_m4"], "m4m5_budget_decision": pilot["m4m5_budget_decision"]},
                     indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
