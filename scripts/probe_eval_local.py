"""Real GPU end-to-end test for L2 (coherence_probe) + L3 (K-shard pooled
bootstrap), using local pre-tokenized .npy shards instead of R2.

Stands up the new behavioral pipeline against real Gemma3-class checkpoints,
confirming that:

  * a legit miner's checkpoint passes the coherence probe (gap ~ king's),
  * the K-shard pooled bootstrap produces a verdict on real data,
  * a same-model self-eval short-circuits the probe (sanity).

Usage (on H200):

    python scripts/probe_eval_local.py \
        --king unconst/Teutonic-I \
        --challenger ClarenceDan/Teutonic-I-Clarence-A5504 \
        --shard-dir /mnt/teutonic-dataset \
        --num-shards 3 --eval-n 96 --batch-size 16 --seed test:0 \
        --gpus 0
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time

# Make the parent (repo root) importable.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

# Force-disable R2 env so eval_torch.R2 won't even try to construct.
for v in ("TEUTONIC_R2_ENDPOINT", "TEUTONIC_R2_ACCESS_KEY",
          "TEUTONIC_R2_SECRET_KEY", "TEUTONIC_DS_ENDPOINT",
          "TEUTONIC_DS_ACCESS_KEY", "TEUTONIC_DS_SECRET_KEY"):
    os.environ.pop(v, None)

from huggingface_hub import HfApi  # noqa: E402

import eval_torch as ET  # noqa: E402

log = logging.getLogger("probe-eval")


class _LocalDsClient:
    """Tiny stand-in for boto3 S3 client.head/get_object on a local file."""

    def __init__(self, root: str):
        self.root = root

    def _path(self, key: str) -> str:
        return key if os.path.isabs(key) else os.path.join(self.root, key)

    def head_object(self, Bucket=None, Key=None):
        return {"ContentLength": os.path.getsize(self._path(Key))}

    def get_object(self, Bucket=None, Key=None, Range=None):
        path = self._path(Key)
        # Range = "bytes=START-END"
        rng = Range[len("bytes="):]
        start_s, end_s = rng.split("-")
        start, end = int(start_s), int(end_s)
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)

        class _Body:
            def __init__(self, b):
                self.b = b

            def read(self):
                return self.b

        return {"Body": _Body(data)}


class LocalR2:
    """Drop-in replacement for ``eval_torch.R2`` that reads local .npy files."""

    def __init__(self, root: str):
        self.root = root
        self.bucket = "local"
        self.ds_bucket = "local"
        self.ds_client = _LocalDsClient(root)

    def ds_range_get(self, key: str, start: int, end: int) -> bytes:
        path = key if os.path.isabs(key) else os.path.join(self.root, key)
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)


def _resolve(api: HfApi, spec: str) -> tuple[str, str]:
    repo, _, rev = spec.partition("@")
    info = api.model_info(repo, revision=rev or None)
    return repo, info.sha


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--shard-dir", default="/mnt/teutonic-dataset")
    ap.add_argument("--shard-glob", default="dataset_v2_shards_shard_*.npy")
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--seed", default="test:0")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import glob
    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, args.shard_glob)))
    if not shard_paths:
        log.error("no shards matched %s/%s", args.shard_dir, args.shard_glob)
        return 2
    if len(shard_paths) < args.num_shards:
        log.error("only %d shards on disk, need %d", len(shard_paths), args.num_shards)
        return 2

    # Pick K shards deterministically from args.seed, mirroring how the
    # validator picks via blake2b. We just use the absolute paths as keys.
    seed_mat = args.seed.encode()
    chosen: list[str] = []
    for i in range(args.num_shards):
        h = int.from_bytes(
            hashlib.blake2b(seed_mat, digest_size=8,
                            person=f"shard{i:02d}".encode()).digest(),
            "big",
        )
        chosen.append(shard_paths[h % len(shard_paths)])
    log.info("picked %d local shards: %s", len(chosen),
             [os.path.basename(p) for p in chosen])

    r2 = LocalR2(args.shard_dir)

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    king_repo, king_rev = _resolve(api, args.king)
    chall_repo, chall_rev = _resolve(api, args.challenger)
    log.info("king:       %s @ %s", king_repo, king_rev[:12])
    log.info("challenger: %s @ %s", chall_repo, chall_rev[:12])

    gpu_ids = ET.parse_gpu_ids(args.gpus)
    log.info("loading king on GPUs %s ...", gpu_ids)
    king_eval = ET.MultiGPUEvaluator(
        king_repo, gpu_ids, label="king", revision=king_rev,
    )
    same = (king_repo == chall_repo and king_rev == chall_rev)
    if same:
        challenger_eval = king_eval
        log.info("self-eval: reusing king as challenger")
    else:
        log.info("loading challenger on GPUs %s ...", gpu_ids)
        challenger_eval = ET.MultiGPUEvaluator(
            chall_repo, gpu_ids, label="challenger", revision=chall_rev,
        )

    t0 = time.time()
    verdict = ET.run_bootstrap_test(
        king_eval, challenger_eval, r2, chosen,
        eval_n=args.eval_n, alpha=args.alpha, delta=args.delta,
        seq_len=args.seq_len, batch_size=args.batch_size,
        seed_str=args.seed, n_bootstrap=args.n_bootstrap,
    )
    log.info("eval finished in %.1fs", time.time() - t0)

    import json
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict.get("accepted") is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
