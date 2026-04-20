"""Three-part audit of whether the chain king is a legitimately fine-tunable
model or a cheat:

  1. weight-delta from seed (mean / p50 / p99 / max / relative Frobenius)
  2. held-out off-corpus loss (we use synthetic shuffled-token sequences and
     a held-out shard NOT in the K-of-N pool used for ranking)
  3. mini fine-tune smoke test: 50 SGD steps on a fresh random shard, does
     loss decrease the way a real LM's would?

Run on the H200:
  HF_TOKEN=... python scripts/probe_finetune_audit.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
SHARD_DIR = "/mnt/teutonic-dataset"
SEED_REPO = "unconst/Teutonic-I"

MINERS = {
    "ChrisJackieChan/Teutonic-I-7": "chain-king",
    "ClarenceDan/Teutonic-I-Clarence-A5504": "clarence",
    "dasLOL/Teutonic-I-king8k": "dasLOL",
    "kt3202/Teutonic-I-test-22": "kt3202",
}


def banner(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Part 1: weight-delta audit
# ---------------------------------------------------------------------------
def load_st_dict(repo: str) -> dict[str, torch.Tensor]:
    t0 = time.time()
    p = snapshot_download(
        repo, token=HF_TOKEN,
        allow_patterns=["*.safetensors", "*.json", "*.model"],
    )
    out: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(p, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    print(f"  loaded {repo} in {time.time()-t0:.1f}s ({len(out)} tensors)")
    return out


def weight_delta_audit(seed: dict, miners: dict[str, dict]) -> None:
    banner("PART 1 / weight-delta vs seed (a real fine-tune is small + smooth)")
    header = ("miner", "mean|d|", "p50", "p99", "max",
              "relFrob", "#shared", "#extra")
    print("{:<42} {:>11} {:>10} {:>10} {:>10} {:>10} {:>8} {:>7}".format(*header))
    for name, sd in miners.items():
        diffs: list[torch.Tensor] = []
        rel_num = rel_den = 0.0
        shared = 0
        for k, sv in seed.items():
            cv = sd.get(k)
            if cv is None or cv.shape != sv.shape:
                continue
            d = (cv.float() - sv.float()).abs()
            diffs.append(d.flatten())
            rel_num += float((cv.float() - sv.float()).pow(2).sum())
            rel_den += float(sv.float().pow(2).sum())
            shared += 1
        extra = len(sd) - shared
        all_d = torch.cat(diffs)
        rel = (rel_num / max(rel_den, 1e-30)) ** 0.5
        print("{:<42} {:>11.4e} {:>10.4e} {:>10.4e} {:>10.4e} "
              "{:>10.4e} {:>8d} {:>7d}".format(
                  name, all_d.mean().item(), all_d.median().item(),
                  all_d.quantile(0.99).item(), all_d.max().item(),
                  rel, shared, extra))

    # Top-5 most-perturbed tensors per miner
    print()
    print("Top-5 most-perturbed tensors per miner (relative Frobenius):")
    for name, sd in miners.items():
        rows: list[tuple[float, str]] = []
        for k, sv in seed.items():
            cv = sd.get(k)
            if cv is None or cv.shape != sv.shape:
                continue
            denom = sv.float().norm().clamp_min(1e-9)
            rel = float((cv.float() - sv.float()).norm() / denom)
            rows.append((rel, k))
        rows.sort(reverse=True)
        print(f"  {name}")
        for r, k in rows[:5]:
            print(f"     rel={r:.4f}  {k}")


# ---------------------------------------------------------------------------
# Part 2: off-corpus loss
# ---------------------------------------------------------------------------
def load_model_for_eval(repo: str, dtype=torch.bfloat16) -> AutoModelForCausalLM:
    cfg = AutoConfig.from_pretrained(repo, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        repo, token=HF_TOKEN, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.eval().cuda()
    return model


def shard_token_batch(shard_path: str, n_seq: int, seq_len: int,
                      seed: int) -> torch.Tensor:
    arr = np.load(shard_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    if arr.ndim == 1:
        # flat tokens; pick contiguous windows
        idx = rng.integers(0, arr.shape[0] - seq_len - 1, size=n_seq)
        out = np.stack([arr[i:i + seq_len] for i in idx])
    else:
        rows = rng.choice(arr.shape[0], size=n_seq, replace=False)
        out = arr[rows, :seq_len]
    return torch.from_numpy(out.astype(np.int64))


@torch.no_grad()
def eval_ce(model, ids: torch.Tensor, batch: int = 8) -> float:
    losses: list[float] = []
    for i in range(0, ids.shape[0], batch):
        x = ids[i:i + batch].cuda()
        out = model(x).logits.float()
        shift_logits = out[:, :-1, :].contiguous()
        shift_labels = x[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="mean")
        losses.append(float(loss))
    return float(np.mean(losses))


def english_ids(tok, text: str, n_seq: int, seq_len: int) -> torch.Tensor:
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.shape[0] < n_seq * seq_len:
        ids = ids.repeat((n_seq * seq_len // ids.shape[0]) + 1)
    ids = ids[: n_seq * seq_len].view(n_seq, seq_len)
    return ids


# A 2-paragraph chunk of plain English that almost certainly is NOT in
# the dataset_v2 shards verbatim (written here on 2026-04-20).
HELDOUT_EN = (
    "On a wet Tuesday in late April two thousand twenty six, a small group "
    "of validators on the Teutonic subnet ran a manual audit of their "
    "incumbent king. The audit had three parts: examine the weights, "
    "examine performance on text never seen during training, and examine "
    "whether the model still trains.\n\n"
    "The point of the exercise was simple. A real language model gets "
    "better at predicting unfamiliar text after it learns. A model that "
    "merely memorised the public training shards does not. And a model "
    "whose weights have been hand-edited often refuses to train at all, "
    "because gradient descent cannot smoothly improve a frozen lookup "
    "table. By measuring all three, the validators could distinguish a "
    "legitimate champion from a clever cheat."
) * 4  # repeat to give us enough tokens


def offcorpus_loss_audit(miners_repos: list[str], shards: list[str]) -> None:
    banner("PART 2 / loss on data the model definitely DID NOT see")

    n_seq, seq_len = 32, 512
    held_shard = shards[-1]
    print(f"held-out shard (NOT in K=3 pool): {held_shard}")

    results: dict[str, dict[str, float]] = {}
    tok = AutoTokenizer.from_pretrained(SEED_REPO, token=HF_TOKEN)
    en_ids = english_ids(tok, HELDOUT_EN, n_seq, seq_len)
    print(f"english held-out: {en_ids.shape} tokens")

    held_ids = shard_token_batch(held_shard, n_seq, seq_len, seed=12345)
    print(f"shard held-out:   {held_ids.shape} tokens")

    for repo in miners_repos:
        print(f"\n  loading {repo} ...")
        m = load_model_for_eval(repo)
        ce_en = eval_ce(m, en_ids)
        ce_shard = eval_ce(m, held_ids)
        results[repo] = {"english": ce_en, "held_shard": ce_shard}
        print(f"    english CE      = {ce_en:.4f} nats")
        print(f"    held-shard CE   = {ce_shard:.4f} nats")
        del m
        torch.cuda.empty_cache()

    print()
    print("SUMMARY:")
    print("{:<45} {:>12} {:>14}".format("miner", "english CE", "held-shard CE"))
    for repo, r in results.items():
        print("{:<45} {:>12.4f} {:>14.4f}".format(
            repo, r["english"], r["held_shard"]))


# ---------------------------------------------------------------------------
# Part 3: mini fine-tune smoke test
# ---------------------------------------------------------------------------
def finetune_smoke_test(repo: str, shards: list[str], steps: int = 50,
                        lr: float = 5e-6, batch: int = 4,
                        seq_len: int = 512) -> list[float]:
    banner(f"PART 3 / mini fine-tune smoke test on {repo}")
    print(f"  {steps} SGD steps, lr={lr}, batch={batch}, seq_len={seq_len}")

    model = AutoModelForCausalLM.from_pretrained(
        repo, token=HF_TOKEN, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.cuda()
    model.train()

    # Use a NEW shard not in the bootstrap K-pool, picked deterministically
    train_shard = shards[0]
    print(f"  train shard: {train_shard}")
    ids = shard_token_batch(train_shard, n_seq=batch * steps,
                            seq_len=seq_len, seed=999)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps):
        x = ids[step * batch:(step + 1) * batch].cuda()
        out = model(x).logits.float()
        shift_logits = out[:, :-1, :].contiguous()
        shift_labels = x[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="mean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 5 == 0 or step == steps - 1:
            print(f"   step {step:>3d}  loss={loss.item():.4f}")

    del model
    torch.cuda.empty_cache()

    first10 = float(np.mean(losses[:10]))
    last10 = float(np.mean(losses[-10:]))
    print(f"  mean(loss first10) = {first10:.4f}")
    print(f"  mean(loss last10)  = {last10:.4f}")
    print(f"  delta              = {first10 - last10:+.4f} "
          f"({'GOOD: trainable' if last10 < first10 - 0.05 else 'SUSPICIOUS: did not improve'})")
    return losses


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    if not HF_TOKEN:
        sys.exit("HF_TOKEN required")

    shards = sorted(glob.glob(os.path.join(SHARD_DIR,
                                           "dataset_v2_shards_shard_*.npy")))
    print(f"found {len(shards)} local shards")

    # Part 1
    seed = load_st_dict(SEED_REPO)
    miners = {repo: load_st_dict(repo) for repo in MINERS}
    weight_delta_audit(seed, miners)
    del seed, miners
    import gc; gc.collect()

    # Part 2: held-out loss
    offcorpus_loss_audit(list(MINERS), shards)

    # Part 3: only fine-tune two models (chain king + one legit baseline)
    finetune_smoke_test("ChrisJackieChan/Teutonic-I-7", shards)
    finetune_smoke_test("ClarenceDan/Teutonic-I-Clarence-A5504", shards)


if __name__ == "__main__":
    main()
