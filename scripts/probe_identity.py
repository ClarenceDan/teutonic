"""Local L1 probe: does compute_identity work and does the seed king pass?

Self-contained dry run for the L1 cryptographic identity layer of the new
validator. Only uses HuggingFace API (header-only metadata + tokenizer
files) — no GPU, no R2, no chain, no bittensor.

Usage:
    HF_TOKEN=... python scripts/probe_identity.py [repo[@revision] ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

from huggingface_hub import HfApi
from safetensors import safe_open

SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", "unconst/Teutonic-I")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

IDENTITY_CONFIG_KEYS = (
    "model_type", "architectures", "vocab_size", "hidden_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "intermediate_size", "max_position_embeddings",
    "rope_theta", "rms_norm_eps",
    "hidden_activation", "attention_bias", "initializer_range",
    "attn_logit_softcapping", "final_logit_softcapping",
    "use_bidirectional_attention", "query_pre_attn_scalar",
    "sliding_window", "sliding_window_pattern",
    "_sliding_window_pattern", "layer_types",
)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safetensors_shapes(api: HfApi, repo: str, revision: str,
                        files: list[str]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    try:
        meta = api.get_safetensors_metadata(repo, revision=revision,
                                            token=HF_TOKEN or None)
        for fname, fmeta in meta.files_metadata.items():
            for tname, tinfo in fmeta.tensors.items():
                shapes[tname] = list(tinfo.shape)
        if shapes:
            return shapes
    except Exception as exc:
        print(f"   (header API fell back: {exc})")

    st_files = sorted(p for p in files if p.endswith(".safetensors"))
    if not st_files:
        raise ValueError("no safetensors files in repo")
    with tempfile.TemporaryDirectory(prefix="teutonic-id-") as tmpdir:
        for rel in st_files:
            local = api.hf_hub_download(repo, rel, token=HF_TOKEN or None,
                                        revision=revision, local_dir=tmpdir)
            with safe_open(local, framework="np") as f:
                for name in f.keys():
                    shapes[name] = list(f.get_slice(name).get_shape())
    return shapes


def compute_identity(repo: str, revision: str) -> dict:
    if not revision:
        raise ValueError("revision required")
    api = HfApi(token=HF_TOKEN or None)
    files = list(api.list_repo_files(repo, token=HF_TOKEN or None,
                                     revision=revision))
    py_files = sorted(p for p in files if p.endswith(".py"))
    cfg_path = api.hf_hub_download(repo, "config.json",
                                   token=HF_TOKEN or None, revision=revision)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "auto_map" in cfg:
        raise ValueError("config.json declares auto_map (custom code)")

    tok_files = sorted(
        p for p in files
        if p.startswith("tokenizer") or p == "special_tokens_map.json"
    )
    tok_hashes: dict[str, str] = {}
    for tf in tok_files:
        local = api.hf_hub_download(repo, tf, token=HF_TOKEN or None,
                                    revision=revision)
        tok_hashes[tf] = _file_sha256(local)

    shapes = _safetensors_shapes(api, repo, revision, files)
    return {
        "config": {k: cfg.get(k) for k in IDENTITY_CONFIG_KEYS if k in cfg},
        "config_full": cfg,
        "tokenizer": tok_hashes,
        "shapes": shapes,
        "py_files": py_files,
    }


def validate_identity(repo: str, revision: str, seed_id: dict) -> str | None:
    if repo == SEED_REPO:
        return None
    try:
        ident = compute_identity(repo, revision)
    except Exception as e:
        return f"cannot compute identity: {e}"
    if ident["py_files"]:
        return f"repo ships executable Python: {ident['py_files'][:3]}"
    seed_full = seed_id.get("config_full", {})
    chall_full = ident.get("config_full", {})
    for k in IDENTITY_CONFIG_KEYS:
        if k in seed_full and k in chall_full and seed_full[k] != chall_full[k]:
            return (f"config.{k} differs: seed={seed_full[k]!r} "
                    f"challenger={chall_full[k]!r}")
    seed_tok = seed_id["tokenizer"]
    chall_tok = ident["tokenizer"]
    if set(seed_tok) != set(chall_tok):
        diff = sorted(set(seed_tok) ^ set(chall_tok))[:3]
        return f"tokenizer file set differs from seed: {diff}"
    for fname, h in seed_tok.items():
        if chall_tok.get(fname) != h:
            return f"tokenizer/{fname} sha256 differs from seed"
    seed_shapes = seed_id["shapes"]
    chall_shapes = ident["shapes"]
    missing = sorted(set(seed_shapes) - set(chall_shapes))
    if missing:
        return f"missing tensors from seed: {missing[:3]}"
    for name, shape in seed_shapes.items():
        if chall_shapes[name] != shape:
            return (f"{name} shape differs: seed={shape} "
                    f"challenger={chall_shapes[name]}")
    return None


def _resolve(spec: str) -> tuple[str, str]:
    repo, _, rev = spec.partition("@")
    api = HfApi(token=HF_TOKEN or None)
    info = api.model_info(repo, revision=rev or None)
    return repo, info.sha


def main() -> int:
    print(f"seed repo: {SEED_REPO}")
    seed_repo, seed_rev = _resolve(SEED_REPO)
    print(f"seed revision: {seed_rev}")

    print("\n[1/3] computing seed identity...")
    seed_id = compute_identity(seed_repo, seed_rev)
    print(f"   config keys: {sorted(seed_id['config'])}")
    print(f"   config values: {json.dumps(seed_id['config'], sort_keys=True)}")
    print(f"   tokenizer files: {sorted(seed_id['tokenizer'])}")
    print(f"   tensor count: {len(seed_id['shapes'])}")
    print(f"   .py files: {seed_id['py_files']}")

    print("\n[2/3] re-computing for determinism...")
    seed_id2 = compute_identity(seed_repo, seed_rev)
    print(f"   deterministic: {seed_id == seed_id2}")
    if seed_id != seed_id2:
        return 2

    print("\n[3/3] validating seed against itself...")
    reason = validate_identity(seed_repo, seed_rev, seed_id)
    print(f"   pass={reason is None} reason={reason!r}")

    extras = sys.argv[1:]
    for spec in extras:
        try:
            repo, rev = _resolve(spec)
        except Exception as exc:
            print(f"\n[{spec}] resolve failed: {exc}")
            continue
        print(f"\n[{repo}@{rev[:12]}]")
        try:
            reason = validate_identity(repo, rev, seed_id)
        except Exception as exc:
            print(f"   identity check raised: {exc}")
            continue
        print(f"   pass={reason is None} reason={reason!r}")

    print("\nL1 probe complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
