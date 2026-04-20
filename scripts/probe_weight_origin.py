"""Deep weight-delta diagnostics. Goal: explain WHY some tensors show
relative Frobenius ratios of 20-30x. Two hypotheses:

  H1 (innocent): seed is near-zero on those tensors (small denominator),
                 so any normal training step produces large rel ratios.
                 Signature: ALL miners blow up on the SAME tensors,
                 absolute |delta| is small.
  H2 (cheat):    weights surgically edited only on top1.
                 Signature: only top1 blows up on those tensors,
                 absolute |delta| is large compared to other miners.
"""
from __future__ import annotations
import glob, os, sys, time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

HF_TOKEN = os.environ.get("HF_TOKEN")

REPOS = {
    "seed": "unconst/Teutonic-I",
    "top1": "ChrisJackieChan/Teutonic-I-7",
    "clarence": "ClarenceDan/Teutonic-I-Clarence-A5504",
    "dasLOL": "dasLOL/Teutonic-I-king8k",
    "kt3202": "kt3202/Teutonic-I-test-22",
}


def load_st(repo: str) -> dict[str, torch.Tensor]:
    t0 = time.time()
    p = snapshot_download(repo, token=HF_TOKEN,
                           allow_patterns=["*.safetensors"])
    out = {}
    for f in sorted(glob.glob(os.path.join(p, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k).float()
    print(f"  {repo}: {len(out)} tensors in {time.time()-t0:.1f}s")
    return out


def fnorm(t: torch.Tensor) -> float:
    return float(t.norm())


def main() -> None:
    weights = {label: load_st(repo) for label, repo in REPOS.items()}
    seed = weights["seed"]
    miners = {k: v for k, v in weights.items() if k != "seed"}

    # ------------------------------------------------------------------
    # Diagnostic 1: For tensors where ANY miner has large rel, show
    # the seed's own Frobenius norm + per-miner absolute |delta| norm.
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("Diagnostic 1: per-tensor seed-norm + per-miner |delta| Frobenius")
    print("=" * 100)

    # Collect all tensors and their per-miner rel ratios.
    rows = []
    for k, sv in seed.items():
        rels = {}
        deltas = {}
        for name, sd in miners.items():
            cv = sd.get(k)
            if cv is None or cv.shape != sv.shape:
                continue
            d = cv - sv
            deltas[name] = fnorm(d)
            rels[name] = deltas[name] / max(fnorm(sv), 1e-9)
        if rels:
            rows.append((max(rels.values()), k, fnorm(sv), rels, deltas))
    rows.sort(reverse=True)

    print(f"\n{'tensor':<50} {'||seed||':>10} | "
          f"{'rel(top1)':>9} {'rel(clar)':>9} {'rel(dasL)':>9} {'rel(kt32)':>9} | "
          f"{'|d|top1':>10} {'|d|clar':>10} {'|d|dasL':>10} {'|d|kt32':>10}")
    print("-" * 165)
    for _, k, sn, rels, deltas in rows[:30]:
        print(f"{k:<50} {sn:>10.4f} | "
              f"{rels.get('top1',float('nan')):>9.3f} "
              f"{rels.get('clarence',float('nan')):>9.3f} "
              f"{rels.get('dasLOL',float('nan')):>9.3f} "
              f"{rels.get('kt3202',float('nan')):>9.3f} | "
              f"{deltas.get('top1',float('nan')):>10.3f} "
              f"{deltas.get('clarence',float('nan')):>10.3f} "
              f"{deltas.get('dasLOL',float('nan')):>10.3f} "
              f"{deltas.get('kt3202',float('nan')):>10.3f}")

    # ------------------------------------------------------------------
    # Diagnostic 2: aggregate by layer family
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("Diagnostic 2: which TENSOR FAMILY does each miner perturb most?")
    print("=" * 100)

    def family(k: str) -> str:
        if "self_attn" in k:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                       "q_norm", "k_norm"):
                if p in k:
                    return f"attn.{p}"
        if "mlp" in k:
            for p in ("gate_proj", "up_proj", "down_proj"):
                if p in k:
                    return f"mlp.{p}"
        if "layernorm" in k or "norm" in k:
            return "norm"
        if "embed" in k:
            return "embed"
        if "lm_head" in k:
            return "lm_head"
        return "other"

    fam_data: dict[str, dict[str, list[float]]] = {}
    for k, sv in seed.items():
        fam = family(k)
        for name, sd in miners.items():
            cv = sd.get(k)
            if cv is None or cv.shape != sv.shape:
                continue
            rel = fnorm(cv - sv) / max(fnorm(sv), 1e-9)
            fam_data.setdefault(fam, {}).setdefault(name, []).append(rel)

    print(f"\n{'family':<20} {'top1 mean':>10} {'top1 max':>10} | "
          f"{'clar mean':>10} {'clar max':>10} | "
          f"{'dasL mean':>10} {'dasL max':>10} | "
          f"{'kt32 mean':>10} {'kt32 max':>10}")
    for fam in sorted(fam_data.keys()):
        per = fam_data[fam]
        cells = []
        for name in ["top1", "clarence", "dasLOL", "kt3202"]:
            vals = per.get(name, [])
            cells.append(float(np.mean(vals)) if vals else float("nan"))
            cells.append(float(np.max(vals)) if vals else float("nan"))
        print(f"{fam:<20} {cells[0]:>10.3f} {cells[1]:>10.3f} | "
              f"{cells[2]:>10.3f} {cells[3]:>10.3f} | "
              f"{cells[4]:>10.3f} {cells[5]:>10.3f} | "
              f"{cells[6]:>10.3f} {cells[7]:>10.3f}")

    # ------------------------------------------------------------------
    # Diagnostic 3: are the top-1 high-rel tensors INITIALIZED near zero
    # in the seed? Compare ||seed|| of "high-rel" tensors vs typical.
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("Diagnostic 3: is the seed near-zero on the tensors where top1 "
          "shows huge rel?")
    print("=" * 100)
    all_seed_norms = sorted([fnorm(v) for v in seed.values()])
    median_norm = float(np.median(all_seed_norms))
    print(f"  median ||seed_tensor||_F across all 266 tensors = {median_norm:.4f}")
    print()
    print("  Top-10 tensors by top1's rel ratio:")
    print(f"  {'tensor':<50} {'||seed||':>10} {'  vs median':>14} {'rel(top1)':>10}")
    top1_rows = sorted(rows, key=lambda r: r[3].get("top1", 0), reverse=True)
    for _, k, sn, rels, deltas in top1_rows[:10]:
        ratio = sn / median_norm
        print(f"  {k:<50} {sn:>10.4f} {ratio:>13.3f}x {rels.get('top1',0):>10.3f}")


if __name__ == "__main__":
    if not HF_TOKEN:
        sys.exit("HF_TOKEN required")
    main()
