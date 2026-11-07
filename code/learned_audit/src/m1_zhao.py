"""M1: Zhao et al. 2022 (ACCV) GCN classification branch.

Adapter around third_party/zhao22 (commit recorded in MANIFEST). Recipe per
repo defaults (opt.py / main.py / utils.py): GCN_class(hidden=256, dropout=0.5,
EC3D 12-class), input = dct_2d((57,T))[:, :25], Adam lr=0.01 decay x0.9 every
5 epochs, 50 epochs, batch 32, drop_last, NLLLoss level=1, grad clip 1.0.
Score: within-action log-odds of error, logsumexp(logp[error classes]) -
logp[correct class] (= logit of the pre-registered 1 - p(correct), p
renormalized within the action's classes; same AUC, no float saturation).
Decision: argmax over 12 classes -> error iff argmax is not the rep's action
correct class. Last-epoch weights. Inference in one batch per condition (the
repo's test path; its forward skips BatchNorm for a batch of 1, see score_items).
"""
import numpy as np
import torch
import torch.nn as nn

FINE_LABEL = {"SQUAT": [1, 2, 3, 4, 5, 10], "Lunges": [1, 4, 6], "Plank": [1, 7, 8]}
OFFSET = {"SQUAT": 0, "Lunges": 6, "Plank": 9}
N_CLASS = 12
DCT_N = 25
HIDDEN = 256
DROPOUT = 0.5
EPOCHS = 50
BATCH = 32
LR = 0.01
LR_DECAY = 5
LR_GAMMA = 0.9
CLIP = 1.0

_ZHAO_DIR = None


def init(zhao_dir):
    """Put the vendored zhao22 repo on sys.path (models/utils/softdtw)."""
    global _ZHAO_DIR
    import sys
    if zhao_dir not in sys.path:
        sys.path.insert(0, zhao_dir)
    _ZHAO_DIR = zhao_dir


def class_of(act, lab):
    return OFFSET[act] + FINE_LABEL[act].index(lab)


def correct_class_of(act):
    return OFFSET[act] + FINE_LABEL[act].index(1)


def classes_of_act(act):
    return [OFFSET[act] + i for i in range(len(FINE_LABEL[act]))]


def to_input(x):
    """(100,19,3) -> zhao (57,T) layout -> dct_2d -> (57,25) float32."""
    from utils import dct_2d  # vendored zhao22 utils; init() ran first
    x57 = x.transpose(2, 1, 0).reshape(57, x.shape[0])
    x57 = np.nan_to_num(x57, nan=0.0)
    d = dct_2d(torch.from_numpy(np.asarray(x57, dtype=np.float64)))[:, :DCT_N]
    return d.float().numpy()


def train_model(train_items, zhao_dir, seed, device, log=print):
    import random
    init(zhao_dir)
    from models import GCN_class  # noqa: E402

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    X = np.stack([to_input(x) for _, x in train_items])
    y = np.array([class_of(s["act"], s["lab"]) for s, _ in train_items], dtype=np.int64)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)

    model = GCN_class(hidden_feature=HIDDEN, p_dropout=DROPOUT, dataset_name="EC3D").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.NLLLoss()
    lr_now = LR
    model.train()
    for epoch in range(EPOCHS):
        if (epoch + 1) % LR_DECAY == 0:
            lr_now *= LR_GAMMA
            for g in opt.param_groups:
                g["lr"] = lr_now
        tot, cor, ls = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.float().to(device), yb.to(device)
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)
            opt.step()
            ls += loss.item() * len(yb)
            tot += len(yb)
            cor += (out.argmax(1) == yb).sum().item()
        if epoch % 10 == 9 or epoch == 0:
            log("m1 epoch {}/{} loss {:.4f} acc {:.3f}".format(epoch + 1, EPOCHS, ls / max(tot, 1), cor / max(tot, 1)))
    return model


def score_items(model, items, device):
    """Return list of (seg, score, pred_err, extra) with extra = {pred_class}.

    All items go through the network in ONE batch, as the repo's own test path
    does (zhao22/main.py: DataLoader(data_test, batch_size=len(data_test))).
    GCN_class.forward skips both BatchNorm1d layers when the batch holds a
    single sample (`if b > 1:` guards in zhao22/models.py), so scoring one rep
    at a time ran inference on un-normalized activations (|log p| ~ 1e4, S0
    12-class accuracy 0.25 instead of ~0.9). In eval mode BN uses its running
    statistics, so a rep's output does not depend on the other reps in the
    batch; only b > 1 matters. Score / pred formulas are unchanged.
    """
    model.eval()
    out_rows = []
    if not items:
        return out_rows
    X = torch.from_numpy(np.stack([to_input(x) for _, x in items])).float().to(device)
    if len(items) == 1:   # keep b > 1 so BN is applied; eval-mode outputs are per-sample
        X = torch.cat([X, X], dim=0)
    with torch.no_grad():
        LP = model(X).double()   # (n, 12) log-softmax over 12 classes
    for i, (s, _) in enumerate(items):
        logp = LP[i]
        c = correct_class_of(s["act"])
        err = torch.tensor([k for k in classes_of_act(s["act"]) if k != c],
                           device=logp.device)
        # Log-odds of error-vs-correct within the action, computed in log space:
        #   logsumexp(logp[error classes]) - logp[correct]
        #   = log((1 - p_c) / p_c) with p renormalized within the action.
        # Strictly monotone transform of the pre-registered 1 - p(correct),
        # so AUC is identical in exact arithmetic; it only removes the
        # float saturation of 1 - exp(.) at exactly 0 / 1.
        score = float(torch.logsumexp(logp[err], dim=0) - logp[c])
        pred_class = int(logp.argmax().item())   # 12-class argmax (for S0 repro)
        pred = int(pred_class != c)              # repo decision: argmax != correct class
        out_rows.append((s, score, pred, {"pred_class": pred_class}))
    return out_rows
