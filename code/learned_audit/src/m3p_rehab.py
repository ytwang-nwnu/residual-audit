"""M3' (fallback for M3): Kelly510/RehabExerAssess (TNSRE 2023).

Adapter around third_party/rehabexer (commit in MANIFEST). Recipe per repo
defaults (utils/arguments.py): model ri-gcn, strategy uniform, Adam lr=1e-4,
100 epochs, batch 64, CrossEntropyLoss on binary labels (correct=1,
incorrect=0), no checkpoint selection -> last-epoch weights (their train.py
reloads best-accuracy; PREREG_R7 mandates last epoch instead).
Input per repo: sample resampled to dataset_fps (we feed T=100 already),
then per-frame centralize over joints (their _centralize), no orientation
normalization (--Pnorm off by default), no augmentation.
Score: p(error) = softmax prob of class 0 (incorrect). Decision: argmax==0.
Connectivity: ec3d BONES (self links added by their Graph).
"""
import numpy as np


def init(repo_dir):
    import sys
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def centralize(sample):
    """Their SingleMoveDataset._centralize: per-frame mean over joints."""
    center = np.repeat(np.mean(sample, axis=1, keepdims=True),
                       sample.shape[1], axis=1)
    return sample - center


def train_model(train_items, repo_dir, bones, seed, device, epochs=100,
                batch_size=64, lr=1e-4, log=print):
    import random
    import torch
    import torch.nn as nn
    init(repo_dir)
    from mvn.classifier import RotationInvariantGCNClassifier

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    X = np.stack([centralize(x.astype(np.float32)) for _, x in train_items])
    y = np.array([0 if s["err"] else 1 for s, _ in train_items], dtype=np.int64)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    num_joints = X.shape[2]
    model = RotationInvariantGCNClassifier(num_joints=num_joints,
                                         in_channels=num_joints,
                                         connectivity=list(bones),
                                         strategy="uniform",
                                         device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        tot, cor, ls = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.float().to(device), yb.to(device)
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls += loss.item() * len(yb)
            tot += len(yb)
            cor += (out.argmax(1) == yb).sum().item()
        if epoch % 20 == 19 or epoch == 0:
            log("m3p epoch {}/{} loss {:.4f} acc {:.3f}".format(epoch + 1, epochs, ls / max(tot, 1), cor / max(tot, 1)))
    return model


def score_items(model, items, device, batch_size=64):
    import torch
    model.eval()
    rows = []
    X = np.stack([centralize(x.astype(np.float32)) for _, x in items])
    with torch.no_grad():
        for i0 in range(0, len(items), batch_size):
            xb = torch.from_numpy(X[i0:i0 + batch_size]).float().to(device)
            p = torch.softmax(model(xb), dim=1)
            for j, (s, _) in enumerate(items[i0:i0 + batch_size]):
                rows.append((s, float(p[j, 0]), int(int(p[j].argmax().item()) == 0)))
    return rows
