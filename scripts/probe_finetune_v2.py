"""Stronger fine-tune smoke test: 100 steps at lr=2e-5, all four miners,
with rolling-window loss to see real movement.

The point is: a real LM, when given a few hundred new sequences and a
reasonable AdamW step, MUST decrease loss on the data it sees. If loss
stays flat or rises, the model is either (a) frozen / hand-edited so the
optimizer cannot find a descent direction, or (b) already at a local
minimum on this exact distribution (which would be suspicious if it
suggests memorization).
"""
from __future__ import annotations
import glob, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

HF_TOKEN = os.environ.get("HF_TOKEN")
SHARD_DIR = "/mnt/teutonic-dataset"

MINERS = [
    "ChrisJackieChan/Teutonic-I-7",
    "ClarenceDan/Teutonic-I-Clarence-A5504",
    "dasLOL/Teutonic-I-king8k",
]


def shard_token_batch(shard_path: str, n_seq: int, seq_len: int,
                      seed: int) -> torch.Tensor:
    arr = np.load(shard_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    if arr.ndim == 1:
        idx = rng.integers(0, arr.shape[0] - seq_len - 1, size=n_seq)
        out = np.stack([arr[i:i + seq_len] for i in idx])
    else:
        rows = rng.choice(arr.shape[0], size=n_seq, replace=False)
        out = arr[rows, :seq_len]
    return torch.from_numpy(out.astype(np.int64))


def smoke(repo: str, ids: torch.Tensor, lr: float, steps: int,
          batch: int) -> list[float]:
    print(f"\n--- {repo}  lr={lr}  steps={steps}  batch={batch} ---")
    model = AutoModelForCausalLM.from_pretrained(
        repo, token=HF_TOKEN, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    losses: list[float] = []
    for s in range(steps):
        x = ids[s * batch:(s + 1) * batch].cuda()
        out = model(x).logits.float()
        loss = F.cross_entropy(
            out[:, :-1, :].contiguous().view(-1, out.size(-1)),
            x[:, 1:].contiguous().view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.detach().item())
        if s % 10 == 0 or s == steps - 1:
            roll = float(np.mean(losses[max(0, s - 9):s + 1]))
            print(f"  step {s:>3d}  loss={losses[-1]:.4f}  roll10={roll:.4f}")
    del model
    torch.cuda.empty_cache()
    return losses


def main() -> None:
    if not HF_TOKEN:
        sys.exit("HF_TOKEN required")
    shards = sorted(glob.glob(os.path.join(SHARD_DIR,
                                           "dataset_v2_shards_shard_*.npy")))
    train_shard = shards[5]  # arbitrary fresh shard
    print(f"train shard: {train_shard}")

    steps, batch, seq_len = 100, 4, 512
    lr = 2e-5

    # Same data for all models so the comparison is fair
    ids = shard_token_batch(train_shard, n_seq=batch * steps,
                            seq_len=seq_len, seed=20260420)

    summaries = {}
    for repo in MINERS:
        losses = smoke(repo, ids, lr=lr, steps=steps, batch=batch)
        first = float(np.mean(losses[:10]))
        last = float(np.mean(losses[-10:]))
        best = float(np.min(np.convolve(losses,
                                         np.ones(10) / 10, mode="valid")))
        summaries[repo] = (first, last, best, last - first)

    print()
    print("=" * 80)
    print("SUMMARY (all models trained on the SAME 400 sequences)")
    print("=" * 80)
    print("{:<45} {:>9} {:>9} {:>9} {:>10}".format(
        "miner", "first10", "last10", "best10", "delta"))
    for repo, (f, l, b, d) in summaries.items():
        verdict = "trains" if d < -0.05 else "FLAT"
        print("{:<45} {:>9.4f} {:>9.4f} {:>9.4f} {:>+10.4f}  {}".format(
            repo, f, l, b, d, verdict))


if __name__ == "__main__":
    main()
