"""Generate round7/MANIFEST.json: script md5s, third-party commits,
per-output md5s, run start/end times, env versions, decisions."""
import glob, hashlib, json, os, subprocess, sys, time
import yaml

CODE = "/data1/users/yangrui/projects/idea67b/experiment/code/prl_v2/round7"
RES = "/data1/users/yangrui/results/idea67b/exp/prl_v2/round7"
TP = "/data1/users/yangrui/projects/idea67b/third_party"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return p.replace("/data1/users/yangrui/", "")


def main():
    m = {}
    m["generated_unix"] = time.time()
    m["prereg"] = {rel(p): md5(p) for p in
                   [os.path.join(RES, "PREREG_R7.md"), os.path.join(RES, "R0_stamp.json"),
                    os.path.join(RES, "selfcheck_data.json"), os.path.join(RES, "pilot.json"),
                    # PREREG_R7_amend1 (M4/M5) + its pre-run stamp
                    os.path.join(RES, "PREREG_R7_amend1.md"),
                    os.path.join(RES, "R0_stamp_amend1.json")]
                   if os.path.exists(p)}
    for stamp in ("R0_stamp.json", "R0_stamp_amend1.json"):
        p = os.path.join(RES, stamp)
        if os.path.exists(p):
            m.setdefault("stamps", {})[stamp] = json.load(open(p))
    m["code"] = dict(
        root=CODE,
        commit=subprocess.run(["git", "-C", CODE, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        scripts={rel(p): md5(p) for p in sorted(glob.glob(CODE + "/**/*.py", recursive=True))
                 if "__pycache__" not in p},
        configs={rel(p): md5(p) for p in sorted(glob.glob(CODE + "/configs/*.yaml"))})
    m["third_party"] = {}
    for name, role in [("zhao22", "M1 code identical to vendored experiment/code/zhao22"),
                       ("stgcn_rehab", "M2"),
                       ("exechecker", "M3 (not run: see m3_fallback)"),
                       ("rehabexer", "M3' fallback"),
                       ("deeprehabpile", "M4 LITEMV / M5 ConvTran (PREREG_R7_amend1); "
                                         "github.com/MSD-IRIMAS/DeepRehabPile, pip -e into env deeprehabpile")]:
        if not os.path.isdir(os.path.join(TP, name)):
            continue
        head = subprocess.run(["git", "-C", os.path.join(TP, name), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        m["third_party"][name] = dict(commit=head, role=role)
    m["amend1"] = dict(
        spec="PREREG_R7_amend1.md (md5 in prereg, pre-run stamp R0_stamp_amend1.json in stamps)",
        methods={"m4": "LITEMV", "m5": "ConvTran"},
        adapters=["src/drp_bench.py", "src/m4_litemv.py", "src/m5_convtran.py"],
        model_selection=("benchmark default: ModelCheckpoint(monitor='loss', save_best_only) -> "
                         "prediction with the lowest TRAINING-loss checkpoint; differs in wording "
                         "from PREREG_R7's 'last epoch' for M1-M3' but involves no test data "
                         "(recorded here as the amendment requires)"),
        ensemble="5 initializations per run unit, probability averaging; "
                 "initialization i of run seed s uses tf.keras.utils.set_random_seed(5*s + i)",
        input_view=("(100,19,3) -> (19,3,100) -> deep_rehab_pile.utils._normalize_skeletons "
                    "(per-dimension min-max fit on the training persons of that action) -> "
                    "(57,100) channels-first, as load_classification_data"),
        env_note=("env deeprehabpile pinned per the repo README (tensorflow 2.16.1, keras 3.6.0, "
                  "aeon 1.2.0, numpy 1.26.4, scikit-learn 1.4.2, hydra-core 1.3.2, omegaconf 2.3.0, "
                  "pandas 2.0.3, matplotlib 3.9.0); pyproject.toml at the recorded commit declares "
                  "tensorflow>=2.18 / aeon==1.4.0 / numpy>=2 instead, so the package was installed "
                  "with `pip install -e . --no-deps`. The estimator code is identical at both "
                  "requirement sets (only pyproject/dockerfile/tests changed after v0.1.2)."))
    m["m3_fallback"] = dict(
        decision="M3' Kelly510/RehabExerAssess used instead of M3 EvenGu/ExeChecker",
        reason=("ExeChecker @1e6df14 cannot run as shipped: trainMulti_perExe.py imports "
                "train_val_test.parser_args, loss.hierarchical_loss, joints_dtw — none in repo; "
                "and the shipped model returns (features, embeddings) only — no code path "
                "produces a per-rep scalar error score (p(error)), which PREREG_R7 requires. "
                "Both activation conditions in the prereg are met."))
    pilot = yaml.safe_load(open(os.path.join(RES, "pilot.json")))
    m["seeds"] = pilot["decision"]
    m["m2_budget_decision"] = pilot.get("m2_budget_decision")
    # PREREG_R7_amend1 pilot + budget decision (m4/m5)
    m["measured_m4"] = pilot.get("measured_m4")
    m["measured_m5_pilot"] = pilot.get("measured_m5_pilot")
    m["m4m5_budget_decision"] = pilot.get("m4m5_budget_decision")
    selp = os.path.join(RES, "summary", "run_selection.json")
    if os.path.exists(selp):
        sel = json.load(open(selp))
        m["run_selection"] = {k: sel.get(k) for k in ("void_rules", "voided", "superseded", "ignored")}
    selp = os.path.join(RES, "summary", "run_selection_amend1.json")
    if os.path.exists(selp):
        sel = json.load(open(selp))
        m["run_selection_amend1"] = {k: sel.get(k) for k in ("void_rules", "voided", "superseded", "ignored")}
    m["envs"] = {}
    probe = ("import sys,json,importlib.util;"
             "v={'python':sys.version.split()[0]};"
             "[v.__setitem__(x, __import__(x).__version__) for x in "
             "('torch','tensorflow','keras','numpy','pandas','sklearn','aeon','omegaconf','hydra') "
             "if importlib.util.find_spec(x)];"
             "print(json.dumps(v))")
    for name, py in [("zhao-ec3d", "/data1/users/yangrui/envs/zhao-ec3d/bin/python"),
                     ("stgcn-rehab", "/data1/users/yangrui/envs/stgcn-rehab/bin/python"),
                     ("idea67b", "/data1/users/yangrui/envs/idea67b/bin/python"),
                     ("deeprehabpile", "/data1/users/yangrui/envs/deeprehabpile/bin/python")]:
        try:
            out = subprocess.run([py, "-c", probe], capture_output=True,
                                 text=True, timeout=120).stdout.strip()
            m["envs"][name] = dict(interpreter=py, **json.loads(out))
        except Exception as e:
            m["envs"][name] = dict(interpreter=py, error=str(e))

    m["experiments"] = []
    for rd in sorted(glob.glob(os.path.join(RES, "runs", "*"))):
        cfgp = os.path.join(rd, "config.yaml")
        if not os.path.exists(cfgp):
            continue
        c = yaml.safe_load(open(cfgp))
        outs = {}
        for p in sorted(glob.glob(rd + "/*")):
            if os.path.isfile(p):
                outs[os.path.basename(p)] = md5(p)
        res = {}
        rp = os.path.join(rd, "resource.json")
        if os.path.exists(rp):
            res = json.load(open(rp))
        mt = os.path.getmtime(cfgp)
        m["experiments"].append(dict(
            run=rel(rd), method=c["method"], split=c["split"], fold=c["fold"],
            seed=c["seed"], commit=c.get("commit"), outputs=outs,
            elapsed_sec=res.get("elapsed_sec"), gpu_mem_max_mib=res.get("gpu_mem_max_mib")))
    amend1_extras = sorted(os.path.relpath(p, RES) for p in
                           glob.glob(os.path.join(RES, "matrix_m4*.log"))
                           + glob.glob(os.path.join(RES, "matrix_m5*.log"))
                           + glob.glob(os.path.join(RES, "pilot_m4*.log"))
                           + glob.glob(os.path.join(RES, "pilot_m5*.log"))
                           + glob.glob(os.path.join(RES, "watch_m4m5*.log"))
                           + glob.glob(os.path.join(RES, "summary_m4m5*.log"))
                           + glob.glob(os.path.join(RES, "watch_m4m5_then_aggregate.sh"))
                           + glob.glob(os.path.join(RES, "baseline_md5_pre_amend1.txt"))
                           # files of the parallel agent session that launched the first chains
                           + glob.glob(os.path.join(RES, "watch_amend1*"))
                           + glob.glob(os.path.join(RES, "summary_amend1*.log"))
                           + glob.glob(os.path.join(RES, "SESSION_STATE_amend1.txt"))
                           + glob.glob(os.path.join(RES, "smoke_amend1", "timing.json"))
                           + glob.glob(os.path.join(RES, "aborted_logs", "amend1_*.log")))
    for extra in ["selfcheck_data.json", "pilot.json", "matrix_m1.log", "matrix_m1_v2.log",
                  "matrix_m1_v3.log", "matrix_m1_v4.log", "matrix_m3p.log", "summary.log",
                  "watch_m2.log", "watch_m2_then_aggregate.sh"] +             sorted(os.path.relpath(p, RES) for p in glob.glob(os.path.join(RES, "aborted_logs", "*.log"))) + amend1_extras:
        p = os.path.join(RES, extra)
        if os.path.exists(p):
            m["prereg"][extra] = md5(p)
    for d in sorted(glob.glob(os.path.join(RES, "m?")) + glob.glob(os.path.join(RES, "m3p"))
                    + [os.path.join(RES, "summary")]):
        if not os.path.isdir(d):
            continue
        m.setdefault("aggregates", {})[os.path.basename(d)] = {
            os.path.basename(p): md5(p) for p in sorted(glob.glob(d + "/*")) if os.path.isfile(p)}
    json.dump(m, open(os.path.join(RES, "MANIFEST.json"), "w"), indent=2, ensure_ascii=False)
    print("manifest:", len(m["experiments"]), "runs")


if __name__ == "__main__":
    main()
