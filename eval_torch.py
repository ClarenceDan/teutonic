#!/usr/bin/env python3
"""Standalone multi-GPU PyTorch eval — king-vs-challenger paired bootstrap test.

Loads model replicas across all available GPUs, fetches sequences from R2
with prefetch overlap, and computes cross-entropy loss via chunked lm_head
forward passes to minimize VRAM. Accepts the challenger only when the
bootstrapped lower confidence bound on the per-token log-loss advantage
exceeds a configurable delta threshold.

Usage:
    python eval_torch.py \
        --king unconst/Teutonic-I \
        --challenger unconst/Teutonic-I \
        --n 100 --delta 0.01 --batch-size 64 --seq-len 2048 --gpus 0,1,2,3,4,5,6,7

Env vars:
    HF_TOKEN              HuggingFace token for gated repos
    TEUTONIC_R2_ENDPOINT  R2 endpoint URL
    TEUTONIC_R2_BUCKET    R2 bucket name (default: constantinople)
    TEUTONIC_R2_ACCESS_KEY R2 access key
    TEUTONIC_R2_SECRET_KEY R2 secret key
    TEUTONIC_DS_ENDPOINT   Dataset store endpoint (default: R2 endpoint)
    TEUTONIC_DS_BUCKET     Dataset store bucket (default: R2 bucket)
    TEUTONIC_DS_ACCESS_KEY Dataset store access key (default: R2 key)
    TEUTONIC_DS_SECRET_KEY Dataset store secret key (default: R2 key)
"""
import argparse
import hashlib
import io
import json
import logging
import os
import pathlib
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3
import numpy as np
import torch
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
from transformers import AutoModelForCausalLM

log = logging.getLogger("eval_torch")


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------

class R2:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
            aws_access_key_id=os.environ["TEUTONIC_R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["TEUTONIC_R2_SECRET_KEY"],
            region_name="auto",
            config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
        )
        self.bucket = os.environ.get("TEUTONIC_R2_BUCKET", "constantinople")

        ds_endpoint = os.environ.get("TEUTONIC_DS_ENDPOINT")
        ds_access = os.environ.get("TEUTONIC_DS_ACCESS_KEY")
        ds_secret = os.environ.get("TEUTONIC_DS_SECRET_KEY")
        if ds_endpoint and ds_access and ds_secret:
            self.ds_client = boto3.client(
                "s3",
                endpoint_url=ds_endpoint,
                aws_access_key_id=ds_access,
                aws_secret_access_key=ds_secret,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    s3={"addressing_style": "path"},
                ),
            )
            self.ds_bucket = os.environ.get("TEUTONIC_DS_BUCKET", self.bucket)
            log.info("dataset store: %s bucket=%s", ds_endpoint, self.ds_bucket)
        else:
            self.ds_client = self.client
            self.ds_bucket = self.bucket

    def get(self, key):
        try:
            return json.loads(
                self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def range_get(self, key, start, end):
        return self.client.get_object(
            Bucket=self.bucket, Key=key, Range=f"bytes={start}-{end}"
        )["Body"].read()

    def ds_get(self, key):
        try:
            return json.loads(
                self.ds_client.get_object(Bucket=self.ds_bucket, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def ds_range_get(self, key, start, end):
        return self.ds_client.get_object(
            Bucket=self.ds_bucket, Key=key, Range=f"bytes={start}-{end}"
        )["Body"].read()


# ---------------------------------------------------------------------------
# Dataset (identical to validator.py)
# ---------------------------------------------------------------------------

def get_shard_info(r2, shard_key):
    # Try local cache first to avoid network round-trip
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name
    if cache_path.exists():
        header = cache_path.read_bytes()[:1024]
    else:
        header = r2.ds_range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    hdr = eval(buf.read(hl).decode("latin1").strip())
    n = 1
    for s in hdr["shape"]:
        n *= s
    return n


R2_FETCH_WORKERS = 32

def _parse_shard_header(r2, shard_key):
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name
    if cache_path.exists():
        header = cache_path.read_bytes()[:1024]
    else:
        header = r2.ds_range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    buf.read(hl)
    return buf.tell()


def fetch_sequences(r2, shard_key, indices, seq_len):
    data_offset = _parse_shard_header(r2, shard_key)
    bps = seq_len * 4
    sorted_idx = sorted(set(indices))
    idx_set = set(indices)

    groups, gs, ge = [], sorted_idx[0], sorted_idx[0]
    for i in sorted_idx[1:]:
        if i - ge <= 64:
            ge = i
        else:
            groups.append((gs, ge))
            gs = ge = i
    groups.append((gs, ge))

    def _fetch_group(gs_ge):
        gs, ge = gs_ge
        chunk = r2.ds_range_get(shard_key, data_offset + gs * bps, data_offset + (ge + 1) * bps - 1)
        partial = {}
        for idx in range(gs, ge + 1):
            if idx in idx_set:
                off = (idx - gs) * bps
                partial[idx] = np.frombuffer(chunk[off : off + bps], dtype="<u4").tolist()
        return partial

    result = {}
    with ThreadPoolExecutor(max_workers=R2_FETCH_WORKERS) as pool:
        for partial in pool.map(_fetch_group, groups):
            result.update(partial)
    return result


SHARD_CACHE_DIR = os.environ.get("TEUTONIC_SHARD_CACHE", "/tmp/shard_cache")
SHARD_CACHE_MAX = int(os.environ.get("TEUTONIC_SHARD_CACHE_MAX", "10"))
SHARD_DL_WORKERS = int(os.environ.get("TEUTONIC_SHARD_DL_WORKERS", "16"))


def _parse_npy_header(raw: bytes) -> int:
    """Return the byte offset where data begins in a .npy file."""
    buf = io.BytesIO(raw)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    buf.read(hl)
    return buf.tell()


def _evict_shard_cache():
    """Keep only the most recent SHARD_CACHE_MAX files in the cache dir."""
    cache = pathlib.Path(SHARD_CACHE_DIR)
    if not cache.exists():
        return
    files = sorted(cache.glob("*.npy"), key=lambda f: f.stat().st_mtime)
    while len(files) > SHARD_CACHE_MAX:
        victim = files.pop(0)
        victim.unlink(missing_ok=True)
        log.info("evicted cached shard %s", victim.name)


def download_shard(r2, shard_key):
    """Download shard with parallel streams and local disk cache."""
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name

    if cache_path.exists():
        t0 = time.time()
        raw = cache_path.read_bytes()
        data_offset = _parse_npy_header(raw)
        elapsed = time.time() - t0
        log.info("shard cache HIT %s: %.1f MB read in %.2fs",
                 shard_key, len(raw) / 1e6, elapsed)
        return data_offset, raw

    t0 = time.time()
    head = r2.ds_client.head_object(Bucket=r2.ds_bucket, Key=shard_key)
    total_size = head["ContentLength"]

    n_workers = min(SHARD_DL_WORKERS, max(1, total_size // (64 * 1024 * 1024)))
    chunk_size = total_size // n_workers
    chunks = [None] * n_workers

    def _dl_chunk(i):
        start = i * chunk_size
        end_byte = total_size - 1 if i == n_workers - 1 else (i + 1) * chunk_size - 1
        chunks[i] = r2.ds_client.get_object(
            Bucket=r2.ds_bucket, Key=shard_key,
            Range=f"bytes={start}-{end_byte}",
        )["Body"].read()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_dl_chunk, range(n_workers)))

    raw = b"".join(chunks)
    del chunks

    data_offset = _parse_npy_header(raw)
    elapsed = time.time() - t0
    log.info("downloaded shard %s: %.1f MB in %.1fs (%.0f MB/s, %d streams)",
             shard_key, len(raw) / 1e6, elapsed,
             len(raw) / 1e6 / elapsed if elapsed > 0 else 0, n_workers)

    try:
        pathlib.Path(SHARD_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        _evict_shard_cache()
        log.info("cached shard to %s", cache_path)
    except Exception:
        log.warning("failed to cache shard to disk", exc_info=True)

    return data_offset, raw


def extract_sequences(shard_data, data_offset, indices, seq_len):
    """Extract sequences from a locally-cached shard."""
    bps = seq_len * 4
    result = {}
    for idx in indices:
        off = data_offset + idx * bps
        result[idx] = np.frombuffer(shard_data[off : off + bps], dtype="<u4").tolist()
    return result


# ---------------------------------------------------------------------------
# Chunked loss computation — avoids materializing full [batch, seq, vocab]
# ---------------------------------------------------------------------------

LM_HEAD_CHUNK = 256

@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    """Forward pass with chunked lm_head to avoid OOM on large vocabs.

    Instead of model(input_ids).logits which allocates [batch, seq, vocab],
    we get hidden states first then apply lm_head in small chunks along the
    sequence dimension. Peak VRAM drops ~7x for vocab_size=262144.
    """
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    hidden = model.model(input_ids).last_hidden_state
    lm_head = model.lm_head

    n_positions = input_ids.size(1) - 1
    total_loss = torch.zeros(len(token_batches), device=device)

    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        chunk_logits = lm_head(hidden[:, i:end_pos, :])
        chunk_labels = input_ids[:, i + 1 : end_pos + 1]
        loss = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        )
        total_loss += loss.reshape(len(token_batches), -1).sum(dim=1)
        del chunk_logits, loss

    return (total_loss / n_positions).cpu().tolist()


# ---------------------------------------------------------------------------
# Paired losses — runs both models' lm_heads per chunk
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_paired_losses(king_model, chall_model, token_batches,
                          king_device, chall_device,
                          chunk_size=LM_HEAD_CHUNK):
    """Compute per-sequence mean cross-entropy for both models on the same tokens.

    Returns (king_losses, chall_losses) as lists of floats (nats/token).
    """
    B = len(token_batches)
    input_ids_k = torch.tensor(token_batches, dtype=torch.long, device=king_device)
    input_ids_c = torch.tensor(token_batches, dtype=torch.long, device=chall_device)

    hidden_k = king_model.model(input_ids_k).last_hidden_state
    hidden_c = chall_model.model(input_ids_c).last_hidden_state

    n_pos = input_ids_k.size(1) - 1
    king_loss = torch.zeros(B, device=king_device)
    chall_loss = torch.zeros(B, device=chall_device)

    for i in range(0, n_pos, chunk_size):
        end = min(i + chunk_size, n_pos)

        logits_k = king_model.lm_head(hidden_k[:, i:end, :])
        logits_c = chall_model.lm_head(hidden_c[:, i:end, :])

        labels_k = input_ids_k[:, i + 1 : end + 1]
        labels_c = input_ids_c[:, i + 1 : end + 1]
        king_loss += F.cross_entropy(
            logits_k.reshape(-1, logits_k.size(-1)), labels_k.reshape(-1),
            reduction="none",
        ).reshape(B, -1).sum(1)
        chall_loss += F.cross_entropy(
            logits_c.reshape(-1, logits_c.size(-1)), labels_c.reshape(-1),
            reduction="none",
        ).reshape(B, -1).sum(1)

        del logits_k, logits_c

    return (
        (king_loss / n_pos).cpu().tolist(),
        (chall_loss / n_pos).cpu().tolist(),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def verify_commit_hash(repo: str, revision: str) -> str:
    """Verify that a HF repo revision resolves to the expected commit SHA.

    Calls the HF API to resolve the revision and returns the actual commit
    hash. Raises ValueError if the resolved commit does not match the
    requested 40-hex SHA. This is a defence-in-depth measure: even though
    ``from_pretrained(revision=<sha>)`` already pins the download, a
    post-download check ensures no TOCTOU or cache-poisoning attack can
    silently substitute a different model.
    """
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN") or None
    info = HfApi(token=token).model_info(repo, revision=revision)
    actual = info.sha
    if revision and len(revision) == 40 and actual != revision:
        raise ValueError(
            f"commit hash mismatch for {repo}: requested {revision} "
            f"but HF resolved to {actual}"
        )
    return actual


def load_model(repo, device, label="model", force_download=False, revision=None):
    log.info("loading %s from %s onto %s (force_download=%s, revision=%s)",
             label, repo, device, force_download, revision[:12] if revision else None)

    # Pre-download commit verification: confirm the revision resolves
    # to the expected SHA before we spend time pulling weights.
    if revision and len(revision) == 40:
        verify_commit_hash(repo, revision)
        log.info("commit hash verified for %s@%s", repo, revision[:12])

    t0 = time.time()
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                attn_implementation=attn_impl,
                token=os.environ.get("HF_TOKEN") or None,
                force_download=force_download,
                revision=revision or None,
                use_safetensors=True,
            )
            log.info("using attn_implementation=%s", attn_impl)
            break
        except Exception as e:
            log.warning("attn %s failed (%s), trying next", attn_impl, e)
    else:
        raise RuntimeError("could not load model with any attention implementation")

    # Post-download commit verification: re-check that the cached model's
    # commit still matches. Guards against cache-level tampering.
    if revision and len(revision) == 40:
        verify_commit_hash(repo, revision)

    model.eval()
    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("%s loaded: %.1fB params in %.1fs (rev=%s verified)", label, params, elapsed,
             revision[:12] if revision else "none")
    return model


# ---------------------------------------------------------------------------
# Weight norm guard — reject models with inflated L2 norms
# ---------------------------------------------------------------------------

NORM_EPSILON = 1e-8

@torch.no_grad()
def check_weight_norms(king_model, challenger_model, max_ratio=5.0):
    """Compare per-parameter L2 norms between king and challenger.

    Returns (ok, violations) where violations is a list of dicts describing
    each parameter that failed the check (non-finite values or norm ratio
    exceeding max_ratio).
    """
    king_params = dict(king_model.named_parameters())
    violations = []

    for c_name, c_param in challenger_model.named_parameters():
        if not torch.isfinite(c_param.data).all():
            violations.append({"param": c_name, "reason": "non-finite values"})
            continue

        k_param = king_params.get(c_name)
        if k_param is None:
            continue

        k_norm = torch.linalg.vector_norm(k_param.data.float()).item()
        c_norm = torch.linalg.vector_norm(c_param.data.float()).item()

        if k_norm < NORM_EPSILON:
            continue

        ratio = c_norm / k_norm
        if ratio > max_ratio:
            violations.append({
                "param": c_name,
                "reason": "norm_ratio",
                "king_norm": round(k_norm, 6),
                "challenger_norm": round(c_norm, 6),
                "ratio": round(ratio, 4),
            })

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Multi-GPU evaluator
# ---------------------------------------------------------------------------

class MultiGPUEvaluator:
    """Manages model replicas across GPUs and dispatches batches in parallel."""

    def __init__(self, repo, gpu_ids, label="model", force_download=False, revision=None):
        self.gpu_ids = gpu_ids
        self.models = {}
        self.devices = {}

        if len(gpu_ids) == 0:
            raise ValueError("need at least one GPU")

        first_model = load_model(repo, f"cuda:{gpu_ids[0]}", f"{label}-gpu{gpu_ids[0]}",
                                 force_download=force_download, revision=revision)
        self.models[gpu_ids[0]] = first_model
        self.devices[gpu_ids[0]] = f"cuda:{gpu_ids[0]}"

        for gid in gpu_ids[1:]:
            self.models[gid] = load_model(repo, f"cuda:{gid}", f"{label}-gpu{gid}",
                                          force_download=force_download, revision=revision)
            self.devices[gid] = f"cuda:{gid}"

        self.pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
        log.info("%s evaluator ready: %d GPUs %s", label, len(gpu_ids), gpu_ids)

    def compute_losses(self, token_batches):
        """Split token_batches across GPUs, compute in parallel, reassemble."""
        n_gpus = len(self.gpu_ids)
        if not token_batches:
            return []

        per_gpu = [[] for _ in range(n_gpus)]
        idx_map = [[] for _ in range(n_gpus)]
        for i, batch in enumerate(token_batches):
            g = i % n_gpus
            per_gpu[g].append(batch)
            idx_map[g].append(i)

        futures = {}
        for g_idx, gid in enumerate(self.gpu_ids):
            if per_gpu[g_idx]:
                fut = self.pool.submit(
                    compute_batch_losses,
                    self.models[gid], per_gpu[g_idx], self.devices[gid],
                )
                futures[fut] = g_idx

        results = [None] * len(token_batches)
        for fut in as_completed(futures):
            g_idx = futures[fut]
            losses = fut.result()
            for local_i, global_i in enumerate(idx_map[g_idx]):
                results[global_i] = losses[local_i]

        return results

    def shutdown(self):
        self.pool.shutdown(wait=False)


def compute_paired_multi_gpu(king_eval, chall_eval, token_batches):
    """Pair king GPUs with challenger GPUs to compute losses in parallel."""
    if not token_batches:
        return [], []

    n_pairs = min(len(king_eval.gpu_ids), len(chall_eval.gpu_ids))
    per_pair = [[] for _ in range(n_pairs)]
    idx_map = [[] for _ in range(n_pairs)]
    for i, batch in enumerate(token_batches):
        p = i % n_pairs
        per_pair[p].append(batch)
        idx_map[p].append(i)

    futures = {}
    pool = ThreadPoolExecutor(max_workers=n_pairs)
    for p_idx in range(n_pairs):
        if not per_pair[p_idx]:
            continue
        k_gid = king_eval.gpu_ids[p_idx]
        c_gid = chall_eval.gpu_ids[p_idx]
        fut = pool.submit(
            compute_paired_losses,
            king_eval.models[k_gid], chall_eval.models[c_gid],
            per_pair[p_idx],
            king_eval.devices[k_gid], chall_eval.devices[c_gid],
        )
        futures[fut] = p_idx

    king_results = [None] * len(token_batches)
    chall_results = [None] * len(token_batches)
    for fut in as_completed(futures):
        p_idx = futures[fut]
        k_losses, c_losses = fut.result()
        for local_i, global_i in enumerate(idx_map[p_idx]):
            king_results[global_i] = k_losses[local_i]
            chall_results[global_i] = c_losses[local_i]

    pool.shutdown(wait=False)
    return king_results, chall_results


# ---------------------------------------------------------------------------
# Coherence probe — cheap sanity check before expensive full eval
# ---------------------------------------------------------------------------

COHERENCE_PROBE_N = 16          # sequences for the probe
COHERENCE_RATIO_FLOOR = 0.5     # chall gap must be >= king gap * ratio
COHERENCE_FLOOR_NATS = 0.1      # absolute minimum gap (nats/token)
COHERENCE_MAX_LOSS = 20.0       # reject if mean loss exceeds this


def coherence_probe(king_eval, challenger_eval, sequences, rng):
    """Quick sanity check: both models should produce sensible losses.

    Picks a small random subset, computes mean losses for each model, and
    verifies they are finite and not degenerate. Returns
    (king_gap, chall_gap, ok, reason) where gap = mean_loss - ln(vocab_size)
    (positive means the model learned *something*).
    """
    n = min(COHERENCE_PROBE_N, len(sequences))
    indices = rng.choice(len(sequences), size=n, replace=False).tolist()
    probe_seqs = [sequences[i] for i in indices]

    king_losses, chall_losses = compute_paired_multi_gpu(
        king_eval, challenger_eval, probe_seqs,
    )

    k_mean = sum(king_losses) / len(king_losses)
    c_mean = sum(chall_losses) / len(chall_losses)

    # ln(vocab_size) is the loss of a uniform random model.
    # For typical LLMs vocab ~256k => ln(256000) ≈ 12.45
    ln_vocab = 12.45
    king_gap = ln_vocab - k_mean
    chall_gap = ln_vocab - c_mean

    if not (np.isfinite(k_mean) and np.isfinite(c_mean)):
        return king_gap, chall_gap, False, "non-finite loss detected"

    if k_mean > COHERENCE_MAX_LOSS:
        return king_gap, chall_gap, False, f"king loss {k_mean:.2f} exceeds max {COHERENCE_MAX_LOSS}"

    if c_mean > COHERENCE_MAX_LOSS:
        return king_gap, chall_gap, False, f"challenger loss {c_mean:.2f} exceeds max {COHERENCE_MAX_LOSS}"

    floor = min(king_gap * COHERENCE_RATIO_FLOOR, COHERENCE_FLOOR_NATS)
    if chall_gap < floor:
        return king_gap, chall_gap, False, (
            f"challenger coherence gap {chall_gap:.3f} below floor {floor:.3f}"
        )

    return king_gap, chall_gap, True, ""


# ---------------------------------------------------------------------------
# Bootstrap test
# ---------------------------------------------------------------------------

def run_bootstrap_test(king_eval, challenger_eval, r2, shard_keys, eval_n,
                       alpha, delta, seq_len, batch_size, seed_str,
                       n_bootstrap=10000, on_progress=None):
    """Paired bootstrap LCB across K independent shards with cross-shard shuffling.

    For K = len(shard_keys), draws ~eval_n / K sequences from each shard
    (deterministic from seed_str + shard index), downloads all shards first,
    pools sequences, shuffles them across shard boundaries, then evaluates
    in shuffled order. This prevents a miner from overfitting a single shard
    and dilutes any shard-specific gaming across all K shards.

    Backwards-compat: ``shard_keys`` may also be a single string for
    one-off CLI invocations.
    """
    if isinstance(shard_keys, str):
        shard_keys = [shard_keys]
    if not shard_keys:
        raise ValueError("at least one shard_key required")

    k = len(shard_keys)
    per_shard_n = max(1, eval_n // k)
    log.info("bootstrap test: K=%d shards, per_shard_n=%d, eval_n=%d alpha=%s delta=%.6f B=%d",
             k, per_shard_n, eval_n, alpha, delta, n_bootstrap)

    base_seed = int.from_bytes(
        hashlib.blake2b(seed_str.encode(), digest_size=8).digest(), "little"
    )

    same_evaluator = king_eval is challenger_eval

    # --- Phase 1: Download all shards and sample sequences ---
    # Each sequence is tagged with (shard_index, local_index) for traceability.
    all_sequences: list[list[int]] = []
    t0 = time.time()

    for shard_idx, shard_key in enumerate(shard_keys):
        n_tokens = get_shard_info(r2, shard_key)
        n_sequences = n_tokens // seq_len
        actual_N = min(per_shard_n, n_sequences)

        shard_seed = base_seed ^ (0xA5A5A5A5 * (shard_idx + 1))
        rng = np.random.Generator(np.random.PCG64(shard_seed))
        eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

        log.info("shard %d/%d %s: downloading & extracting %d sequences",
                 shard_idx + 1, k, shard_key, actual_N)
        data_offset, shard_data = download_shard(r2, shard_key)
        seq_cache = extract_sequences(shard_data, data_offset, eval_indices, seq_len)
        del shard_data  # free memory early

        # Collect sequences in order; we will shuffle below.
        for idx in eval_indices:
            all_sequences.append(seq_cache[idx])
        del seq_cache

    total_N = len(all_sequences)
    log.info("total sequences from %d shards: %d (download phase %.1fs)",
             k, total_N, time.time() - t0)

    # --- Phase 2: Cross-shard shuffle ---
    # Deterministic shuffle so the eval is reproducible but the order is
    # unpredictable to the miner.
    shuffle_rng = np.random.Generator(np.random.PCG64(base_seed ^ 0x5F1E5F1E))
    shuffle_order = shuffle_rng.permutation(total_N).tolist()
    all_sequences = [all_sequences[i] for i in shuffle_order]
    log.info("sequences shuffled across %d shards", k)

    # --- Phase 2b: Coherence probe (before expensive full eval) ---
    coherence_info: dict | None = None
    if not same_evaluator:
        probe_rng = np.random.Generator(np.random.PCG64(base_seed ^ 0xC0DE))
        king_gap, chall_gap, ok, reason = coherence_probe(
            king_eval, challenger_eval, all_sequences, probe_rng,
        )
        coherence_info = {
            "king_coherence_gap_nats": round(king_gap, 4),
            "challenger_coherence_gap_nats": round(chall_gap, 4),
            "floor_nats": round(min(king_gap * COHERENCE_RATIO_FLOOR,
                                    COHERENCE_FLOOR_NATS), 4),
        }
        log.info("coherence probe: king_gap=%.3f chall_gap=%.3f ok=%s",
                 king_gap, chall_gap, ok)
        if not ok:
            elapsed = time.time() - t0
            return {
                "accepted": False,
                "verdict": "king",
                "rejection_reason": "coherence_probe_failed",
                "rejection_detail": reason,
                **coherence_info,
                "K_shards": k,
                "shards": list(shard_keys),
                "wall_time_s": round(elapsed, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # --- Phase 3: Batched eval on shuffled sequences ---
    batches = [
        list(range(i, min(i + batch_size, total_N)))
        for i in range(0, total_N, batch_size)
    ]

    all_diffs: list[float] = []
    king_sum = 0.0
    chall_sum = 0.0
    total_done = 0

    for bi, batch_indices in enumerate(batches):
        token_batches = [all_sequences[idx] for idx in batch_indices]

        if same_evaluator:
            king_losses = king_eval.compute_losses(token_batches)
            chall_losses = king_losses
        else:
            king_losses, chall_losses = compute_paired_multi_gpu(
                king_eval, challenger_eval, token_batches,
            )

        for k_loss, c_loss in zip(king_losses, chall_losses):
            total_done += 1
            king_sum += k_loss
            chall_sum += c_loss
            all_diffs.append(k_loss - c_loss)

        elapsed = time.time() - t0
        seqs_per_sec = total_done / elapsed if elapsed > 0 else 0
        mu_hat = float(np.mean(all_diffs)) if all_diffs else 0.0
        log.info(
            "batch %d/%d | done=%d/%d | mu_hat=%.6f | %.1f seq/s",
            bi + 1, len(batches), total_done, total_N, mu_hat, seqs_per_sec,
        )

        if on_progress:
            on_progress({
                "done": total_done, "total": total_N,
                "mu_hat": round(float(mu_hat), 6),
                "avg_king_loss": round(king_sum / total_done, 6),
                "avg_challenger_loss": round(chall_sum / total_done, 6),
                "seqs_per_sec": round(seqs_per_sec, 1),
                "shard_count": k,
            })

    del all_sequences

    elapsed = time.time() - t0
    d = np.array(all_diffs)
    mu_hat = float(d.mean())

    boot_rng = np.random.Generator(np.random.PCG64(base_seed ^ 0xB007))
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, len(d), size=len(d))
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, alpha))

    accepted = lcb > delta
    log.info("bootstrap result: K=%d N=%d mu_hat=%.6f lcb=%.6f delta=%.6f accepted=%s",
             k, total_done, mu_hat, lcb, delta, accepted)

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": delta,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "N": total_done,
        "K_shards": k,
        "shards": list(shard_keys),
        "avg_king_loss": round(king_sum / total_done, 6) if total_done else 0,
        "avg_challenger_loss": round(chall_sum / total_done, 6) if total_done else 0,
        "wall_time_s": round(elapsed, 1),
        "seqs_per_sec": round(total_done / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if coherence_info:
        verdict.update(coherence_info)
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_gpu_ids(gpu_str):
    if gpu_str == "auto":
        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in gpu_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU PyTorch model eval")
    parser.add_argument("--king", required=True, help="HF repo for king model")
    parser.add_argument("--challenger", required=True, help="HF repo for challenger model")
    parser.add_argument("--n", type=int, default=100, help="Number of sequences to evaluate")
    parser.add_argument("--alpha", type=float, default=0.001, help="Bootstrap confidence level (one-sided)")
    parser.add_argument("--delta", type=float, default=0.01, help="Minimum effect threshold in nats/token")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--batch-size", type=int, default=64, help="Sequences per batch (split across GPUs)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Tokens per sequence")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU IDs or 'auto' (default: auto)")
    parser.add_argument("--seed", default="test:eval", help="Seed string for deterministic sequence selection")
    parser.add_argument("--shard", default=None, help="Specific shard key (default: first shard from manifest)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for var in ["TEUTONIC_R2_ENDPOINT", "TEUTONIC_R2_ACCESS_KEY", "TEUTONIC_R2_SECRET_KEY"]:
        if var not in os.environ:
            log.error("missing env var: %s", var)
            sys.exit(1)

    gpu_ids = parse_gpu_ids(args.gpus)
    log.info("using GPUs: %s", gpu_ids)

    r2 = R2()

    if args.shard:
        shard_keys = [args.shard]
    else:
        manifest = r2.ds_get("dataset/v2/manifest.json")
        if not manifest:
            manifest = r2.get("dataset/v1/manifest.json")
        if not manifest:
            log.error("could not fetch dataset manifest")
            sys.exit(1)
        # Default: use first shard only for CLI; pass --shard multiple times
        # or use the eval_server for multi-shard.
        shard_keys = [manifest["shards"][0]["key"]]
        log.info("using shard: %s (%d shards available, version=%s)",
                 shard_keys[0], len(manifest["shards"]), manifest.get("version", "v1"))

    same_model = args.king == args.challenger

    if same_model:
        log.info("king == challenger, using all %d GPUs for shared evaluator", len(gpu_ids))
        king_eval = MultiGPUEvaluator(args.king, gpu_ids, label="king")
        challenger_eval = king_eval
    else:
        mid = len(gpu_ids) // 2
        king_gpus = gpu_ids[:mid] or gpu_ids[:1]
        chall_gpus = gpu_ids[mid:] or gpu_ids[:1]
        log.info("king GPUs: %s  challenger GPUs: %s", king_gpus, chall_gpus)
        king_eval = MultiGPUEvaluator(args.king, king_gpus, label="king")
        challenger_eval = MultiGPUEvaluator(args.challenger, chall_gpus, label="challenger")

    log.info("=" * 60)
    log.info("EVAL CONFIG")
    log.info("  king:       %s", args.king)
    log.info("  challenger: %s", args.challenger)
    log.info("  GPUs:       %s (%s)", gpu_ids, "shared" if same_model else "split")
    log.info("  N=%d  alpha=%s  delta=%.6f  bootstrap=%d  batch=%d  seq_len=%d",
             args.n, args.alpha, args.delta, args.n_bootstrap, args.batch_size, args.seq_len)
    log.info("  shards: %s", shard_keys)
    log.info("  seed:  %s", args.seed)
    log.info("=" * 60)

    verdict = run_bootstrap_test(
        king_eval, challenger_eval,
        r2, shard_keys, args.n, args.alpha, args.delta,
        args.seq_len, args.batch_size, args.seed,
        n_bootstrap=args.n_bootstrap,
    )

    king_eval.shutdown()
    if not same_model:
        challenger_eval.shutdown()

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(json.dumps(verdict, indent=2))
    print("=" * 60)

    return 0 if not verdict["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
