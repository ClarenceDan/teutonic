#!/usr/bin/env python3
"""Teutonic validator — single-file king-of-the-hill evaluator.

Polls Bittensor chain for challenger submissions, dispatches evaluations
to a remote eval server (eval_server.py on a GPU box), manages king
lifecycle on HuggingFace, persists all state to R2.
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone

import bittensor as bt
import boto3
import httpx
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_N = 20_000
EVAL_ALPHA = 0.001
EVAL_DELTA = float(os.environ.get("TEUTONIC_EVAL_DELTA", "0.01"))
SEQ_LEN = 2048
POLL_INTERVAL = 30
WEIGHT_INTERVAL = 300
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", "unconst/Teutonic-I")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EVAL_SERVER_URL = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")

R2_ENDPOINT = os.environ.get("TEUTONIC_R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("TEUTONIC_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("TEUTONIC_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("TEUTONIC_R2_SECRET_KEY", "")

HIPPIUS_ENDPOINT = os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com")
HIPPIUS_BUCKET = os.environ.get("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3")
HIPPIUS_ACCESS_KEY = os.environ.get("TEUTONIC_HIPPIUS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY = os.environ.get("TEUTONIC_HIPPIUS_SECRET_KEY", "")

DS_ENDPOINT = os.environ.get("TEUTONIC_DS_ENDPOINT", "")
DS_BUCKET = os.environ.get("TEUTONIC_DS_BUCKET", "")
DS_ACCESS_KEY = os.environ.get("TEUTONIC_DS_ACCESS_KEY", "")
DS_SECRET_KEY = os.environ.get("TEUTONIC_DS_SECRET_KEY", "")

TMC_API_KEY = os.environ.get("TMC_API_KEY", "")

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")

# Reveal payload format (post-V2): "<king_revision>:<challenger_repo>:<challenger_revision>"
# - king_revision and challenger_revision MUST be 40-char lowercase hex git SHAs.
# - challenger_repo MUST match the Teutonic-I namespace pattern below.
# Anything else is silently dropped at scan time. There is no backwards compat.
import re as _re
_REPO_FRAGMENT = r"[A-Za-z0-9_.\-]+"
REVEAL_RE = _re.compile(
    rf"^(?P<king_rev>[0-9a-f]{{40}}):"
    rf"(?P<repo>{_REPO_FRAGMENT}/Teutonic-I-{_REPO_FRAGMENT}):"
    rf"(?P<chal_rev>[0-9a-f]{{40}})$"
)

# Architectural identity (post-V3): a challenger is rejected at scan time
# unless it presents the SAME architecture, tokenizer, and tensor shape
# map as the seed model. These checks are deterministic and cryptographic;
# behavioral safety is enforced separately at eval time via the
# coherence probe in eval_torch. We removed all per-tensor norm/zero
# statistics on purpose — they amounted to playing "guess my threshold"
# with attackers and never reliably caught poisoned weights. Behavior is
# what we actually care about, and behavior is what we measure.
KING_HEALTH_INTERVAL_S = float(os.environ.get("TEUTONIC_KING_HEALTH_INTERVAL_S", "3600"))

TMC_BASE = "https://api.taomarketcap.com/public/v1"

log = logging.getLogger("teutonic")


# ---------------------------------------------------------------------------
# TaoMarketCap
# ---------------------------------------------------------------------------

async def fetch_tmc_data() -> dict | None:
    """Fetch TAO price, SN3 alpha price, and registration burn from TMC API."""
    if not TMC_API_KEY:
        return None
    headers = {"Authorization": TMC_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            market_resp, subnet_resp, burn_resp = await asyncio.gather(
                client.get(f"{TMC_BASE}/market/market-data/", headers=headers),
                client.get(f"{TMC_BASE}/subnets/{NETUID}/", headers=headers),
                client.get(f"{TMC_BASE}/subnets/burn/{NETUID}/", headers=headers),
            )
        m = market_resp.json()
        s = subnet_resp.json()
        b = burn_resp.json()
        asp = float(s["latest_snapshot"]["alpha_sqrt_price"])
        tao_price = m["current_price"]
        alpha_tao = asp ** 2
        return {
            "tao_price_usd": tao_price,
            "tao_change_24h": m["usd_quote"]["percent_change_24h"],
            "sn3_alpha_price_tao": alpha_tao,
            "sn3_alpha_price_usd": alpha_tao * tao_price,
            "sn3_reg_burn_tao": b[0]["burn"] / 1e9,
        }
    except Exception:
        log.warning("TMC fetch failed", exc_info=True)
        return None

# ---------------------------------------------------------------------------
# Discord notifications
# ---------------------------------------------------------------------------

async def notify_new_king(king_info: dict, verdict: dict | None = None):
    """Post a message to Discord when a new king is crowned."""
    if not DISCORD_BOT_TOKEN or not DISCORD_CHANNEL_ID:
        return
    repo = king_info.get("hf_repo", "?")
    hotkey = king_info.get("hotkey", "?")
    reign = king_info.get("reign_number", 0)
    revision = king_info.get("king_revision", "")[:12]

    lines = [
        f"**New King of Subnet 3!**",
        f"**Repo:** `{repo}`" + (f" (`{revision}`)" if revision else ""),
        f"**Hotkey:** `{hotkey[:16]}...`",
        f"**Reign:** #{reign}",
    ]
    if verdict:
        mu = verdict.get("mu_hat", 0)
        king_loss = verdict.get("avg_king_loss", 0)
        chall_loss = verdict.get("avg_challenger_loss", 0)
        wall = verdict.get("wall_time_s", 0)
        lines.append(f"**Eval:** challenger loss {chall_loss:.4f} vs king loss {king_loss:.4f} (μ̂={mu:.6f}, {wall:.0f}s)")
    prev = king_info.get("previous_king")
    if prev and prev.get("hf_repo"):
        lines.append(f"**Dethroned:** `{prev['hf_repo']}`")

    embed = {
        "title": "👑 New King Crowned",
        "description": "\n".join(lines),
        "color": 0xFFD700,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(
                f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
                headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}",
                         "Content-Type": "application/json"},
                json={"embeds": [embed]},
            )
            if resp.status_code < 300:
                log.info("discord notification sent for reign #%d", reign)
            else:
                log.warning("discord notification failed: %d %s", resp.status_code, resp.text[:200])
    except Exception:
        log.warning("discord notification error", exc_info=True)


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------

class R2:
    def __init__(self):
        self.client = boto3.client(
            "s3", endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
            config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
        )
        if HIPPIUS_ACCESS_KEY and HIPPIUS_SECRET_KEY:
            self._hippius = boto3.client(
                "s3", endpoint_url=HIPPIUS_ENDPOINT,
                aws_access_key_id=HIPPIUS_ACCESS_KEY,
                aws_secret_access_key=HIPPIUS_SECRET_KEY,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    s3={"addressing_style": "path"},
                ),
            )
        else:
            self._hippius = None

        if DS_ACCESS_KEY and DS_SECRET_KEY and DS_ENDPOINT:
            self._ds_client = boto3.client(
                "s3", endpoint_url=DS_ENDPOINT,
                aws_access_key_id=DS_ACCESS_KEY,
                aws_secret_access_key=DS_SECRET_KEY,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    s3={"addressing_style": "path"},
                ),
            )
            self._ds_bucket = DS_BUCKET
            log.info("dataset store: %s bucket=%s", DS_ENDPOINT, DS_BUCKET)
        else:
            self._ds_client = None
            self._ds_bucket = None

    def put_dashboard(self, key, data):
        body = json.dumps(data, default=str).encode()
        ct = "application/json"
        if self._hippius:
            self._hippius.put_object(
                Bucket=HIPPIUS_BUCKET, Key=key, Body=body, ContentType=ct,
            )
        else:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key, Body=body, ContentType=ct,
            )

    def put_dashboard_raw(self, key, body, content_type):
        if self._hippius:
            self._hippius.put_object(
                Bucket=HIPPIUS_BUCKET, Key=key, Body=body,
                ContentType=content_type,
            )
        else:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key, Body=body,
                ContentType=content_type,
            )

    def put(self, key, data):
        try:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key,
                Body=json.dumps(data, default=str).encode(),
                ContentType="application/json",
            )
        except Exception:
            log.warning("R2 put failed for %s (non-fatal)", key)

    def get(self, key):
        try:
            return json.loads(
                self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def ds_get(self, key):
        """Read JSON from the dataset store (Hippius), falling back to R2."""
        if self._ds_client:
            try:
                return json.loads(
                    self._ds_client.get_object(
                        Bucket=self._ds_bucket, Key=key
                    )["Body"].read()
                )
            except Exception:
                pass
        return self.get(key)

    def append_jsonl(self, key, record):
        try:
            line = json.dumps(record, default=str) + "\n"
            existing = b""
            try:
                existing = self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            except Exception:
                pass
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key,
                Body=existing + line.encode(),
                ContentType="application/x-ndjson",
            )
        except Exception:
            log.warning("R2 append_jsonl failed for %s (non-fatal)", key)

    def append_jsonl_batch(self, key, records):
        try:
            lines = "".join(json.dumps(r, default=str) + "\n" for r in records)
            existing = b""
            try:
                existing = self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            except Exception:
                pass
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key,
                Body=existing + lines.encode(),
                ContentType="application/x-ndjson",
            )
        except Exception:
            log.warning("R2 append_jsonl_batch failed for %s (non-fatal)", key)

    def put_raw(self, key, body, content_type):
        try:
            self.client.put_object(
                Bucket=R2_BUCKET, Key=key, Body=body, ContentType=content_type,
            )
        except Exception:
            log.warning("R2 put_raw failed for %s (non-fatal)", key)

    def range_get(self, key, start, end):
        return self.client.get_object(
            Bucket=R2_BUCKET, Key=key, Range=f"bytes={start}-{end}"
        )["Body"].read()


# ---------------------------------------------------------------------------
# Architectural identity
#
# A challenger that wants to be a Gemma3 model under the same evaluation
# protocol must commit to four things, all of which are independent of how
# the weights were trained:
#
#   1. config.json fields that define the architecture (shapes, vocab,
#      norm constants, rope, etc.) must equal the seed's.
#   2. The repo must contain NO executable Python and NO ``auto_map`` in
#      config.json. We always load with ``trust_remote_code=False``;
#      shipping custom code is always either pointless or hostile.
#   3. The tokenizer files must be byte-identical to the seed (sha256), so
#      that token IDs carry the same meaning across king/challenger/eval.
#   4. The set of safetensors tensor names AND each tensor's shape must
#      equal the seed's. Anything else is a different model.
#
# These four are deterministic, cryptographic where applicable, and free
# of magic numbers. They tell us whether a candidate is the right *kind*
# of object. Whether the weights inside that object actually behave like a
# language model is a separate question, answered behaviorally during
# eval (see eval_torch.coherence_probe, K-shard bootstrap).
# ---------------------------------------------------------------------------

IDENTITY_CONFIG_KEYS = (
    "model_type", "architectures", "vocab_size", "hidden_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "intermediate_size", "max_position_embeddings",
    "rope_theta", "rms_norm_eps", "tie_word_embeddings",
)

_seed_identity: dict | None = None
_identity_cache: dict[str, dict] = {}
_IDENTITY_CACHE_MAX = 32


def _identity_cache_put(key: str, value: dict) -> None:
    if len(_identity_cache) >= _IDENTITY_CACHE_MAX:
        _identity_cache.pop(next(iter(_identity_cache)))
    _identity_cache[key] = value


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safetensors_shapes(api: HfApi, repo: str, revision: str,
                        files: list[str]) -> dict[str, list[int]]:
    """Return ``{tensor_name: shape}`` for every tensor in the repo.

    Prefers the hub's ``get_safetensors_metadata`` (header-only, no weight
    download). Falls back to streaming the safetensors header ourselves
    on older hub versions, which still avoids reading weight bytes thanks
    to safetensors' lazy file format.
    """
    shapes: dict[str, list[int]] = {}
    try:
        meta = api.get_safetensors_metadata(repo, revision=revision,
                                            token=HF_TOKEN or None)
        for fname, fmeta in meta.files_metadata.items():
            for tname, tinfo in fmeta.tensors.items():
                shapes[tname] = list(tinfo.shape)
        if shapes:
            return shapes
    except Exception:
        pass

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
    """Cryptographic + structural identity card for a pinned revision.

    Always called with a 40-hex git SHA so HF cannot silently swap the
    underlying commit. The result is small (kilobytes) and process-cached;
    a second call for the same ``repo@revision`` is free.
    """
    if not revision:
        raise ValueError("revision required for identity")
    cache_key = f"{repo}@{revision}"
    cached = _identity_cache.get(cache_key)
    if cached is not None:
        return cached

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

    identity = {
        "config": {k: cfg.get(k) for k in IDENTITY_CONFIG_KEYS if k in cfg},
        "tokenizer": tok_hashes,
        "shapes": shapes,
        "py_files": py_files,
    }
    _identity_cache_put(cache_key, identity)
    return identity


def _load_seed_identity() -> dict:
    """Return the seed model's identity card, computing it once per process."""
    global _seed_identity
    if _seed_identity is not None:
        return _seed_identity
    api = HfApi(token=HF_TOKEN or None)
    seed_rev = api.model_info(SEED_REPO).sha
    log.info("computing seed identity for %s@%s", SEED_REPO, seed_rev[:12])
    _seed_identity = compute_identity(SEED_REPO, seed_rev)
    log.info("seed identity: %d tensors, %d tokenizer files, %d config keys",
             len(_seed_identity["shapes"]),
             len(_seed_identity["tokenizer"]),
             len(_seed_identity["config"]))
    return _seed_identity


def validate_identity(repo: str, revision: str, seed_id: dict) -> str | None:
    """Return None iff ``repo@revision`` is the same architecture+tokenizer
    as the seed. Any deviation is rejected verbatim, with no thresholds."""
    if repo == SEED_REPO:
        return None
    try:
        ident = compute_identity(repo, revision)
    except Exception as e:
        return f"cannot compute identity: {e}"

    if ident["py_files"]:
        return f"repo ships executable Python: {ident['py_files'][:3]}"

    for k, seed_val in seed_id["config"].items():
        chall_val = ident["config"].get(k)
        if seed_val != chall_val:
            return f"config.{k} differs: seed={seed_val!r} challenger={chall_val!r}"

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
    extra = sorted(set(chall_shapes) - set(seed_shapes))
    missing = sorted(set(seed_shapes) - set(chall_shapes))
    if extra or missing:
        return (f"tensor name set differs (extra={extra[:3]} "
                f"missing={missing[:3]})")
    for name, shape in seed_shapes.items():
        if chall_shapes[name] != shape:
            return (f"{name} shape differs: seed={shape} "
                    f"challenger={chall_shapes[name]}")

    return None


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def scan_reveals(subtensor, netuid, seen):
    """Pull revealed commitments and parse them under the strict v2 schema.

    Payload: ``<king_revision>:<challenger_repo>:<challenger_revision>``
    Both revisions MUST be lowercase 40-hex git SHAs; the repo MUST sit in the
    Teutonic-I namespace. Any reveal that fails to parse is silently dropped
    so a malformed legacy payload cannot keep being re-tried forever.
    """
    try:
        all_reveals = subtensor.get_all_revealed_commitments(netuid)
    except Exception:
        log.exception("failed to fetch reveals")
        return []
    if not all_reveals:
        return []

    new = []
    for hotkey, entries in all_reveals.items():
        if hotkey in seen or not entries:
            continue
        block, data = max(entries, key=lambda e: e[0])
        m = REVEAL_RE.match((data or "").strip())
        if not m:
            seen.add(hotkey)
            log.info("dropping malformed reveal from %s (block=%d)", hotkey[:16], block)
            continue
        seen.add(hotkey)
        new.append({
            "hotkey": hotkey,
            "block": block,
            "king_revision": m.group("king_rev"),
            "hf_repo": m.group("repo"),
            "challenger_revision": m.group("chal_rev"),
        })
    new.sort(key=lambda x: x["block"])
    return new


def set_weights(subtensor, wallet, netuid, king_hotkey) -> bool:
    """Set 100% weight to *king_hotkey*. Returns True on success."""
    try:
        meta = subtensor.metagraph(netuid)
        if king_hotkey in meta.hotkeys:
            uid = meta.hotkeys.index(king_hotkey)
            subtensor.set_weights(wallet=wallet, netuid=netuid, uids=[uid], weights=[1.0])
            log.info("weights set 100%% to uid=%d (%s)", uid, king_hotkey[:16])
            return True
        else:
            log.warning("king hotkey %s not in metagraph", king_hotkey[:16])
            return False
    except Exception:
        log.exception("failed to set weights")
        return False


def maybe_set_weights(subtensor, wallet, state, *, force: bool = False,
                      reason: str = "") -> bool:
    """Set weights to the current king if forced or WEIGHT_INTERVAL has elapsed.

    Always uses ``state.king["hotkey"]`` as the source of truth so the chain
    keeps tracking the current top miner even when no new dethrone fires.
    """
    king_hotkey = state.king.get("hotkey") if state.king else None
    if not king_hotkey:
        if force:
            log.info("skipping weight set (%s): no king yet", reason or "forced")
        return False
    try:
        current_block = subtensor.block
    except Exception:
        log.exception("failed to read current block for weight-set")
        return False
    if not force and current_block - state.last_weight_block < WEIGHT_INTERVAL:
        return False
    log.info("setting weights at block %d (last=%d, %s) to king %s",
             current_block, state.last_weight_block,
             reason or ("forced" if force else "interval"), king_hotkey[:16])
    if set_weights(subtensor, wallet, NETUID, king_hotkey):
        state.last_weight_block = current_block
        state.last_winner_hotkey = king_hotkey
        try:
            state.flush()
        except Exception:
            log.exception("failed to flush state after weight set")
        return True
    return False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


class State:
    def __init__(self, r2):
        self.r2 = r2
        self.king = {}
        self.queue = []
        self.seen = set()
        self.failed_repos: set[str] = set()
        self.evaluated_repos: set[str] = set()
        self.stats = {"queued": 0, "accepted": 0, "rejected": 0, "failed": 0}
        self.counter = 0
        self.current_eval = None
        self.history = []
        self.last_weight_block = 0
        self.last_winner_hotkey: str | None = None
        self.market: dict | None = None
        self.uid_map: dict[str, int] = {}

    def load(self):
        k = self.r2.get("king/current.json")
        if k:
            self.king = k
        q = self.r2.get("state/queue.json")
        if q:
            pending = q.get("pending", [])
            # Drop queue entries from older payload schemas. The new schema
            # requires both king_revision and challenger_revision as 40-hex.
            self.queue = [
                e for e in pending
                if isinstance(e.get("king_revision"), str)
                and isinstance(e.get("challenger_revision"), str)
                and len(e["king_revision"]) == 40
                and len(e["challenger_revision"]) == 40
            ]
            dropped = len(pending) - len(self.queue)
            if dropped:
                log.warning("dropped %d legacy queue entries during load", dropped)
        s = self.r2.get("state/seen_hotkeys.json")
        if s:
            self.seen = set(s.get("hotkeys", []))
        st = self.r2.get("state/validator_state.json")
        if st:
            loaded = st.get("stats", self.stats)
            if "challenges" in loaded and "queued" not in loaded:
                loaded["queued"] = loaded.pop("challenges")
            loaded.setdefault("failed", 0)
            loaded.setdefault("queued", 0)
            self.stats = loaded
            self.counter = st.get("counter", 0)
            self.last_weight_block = st.get("last_weight_block", 0)
            self.last_winner_hotkey = st.get("last_winner_hotkey")
        h = self.r2.get("state/dashboard_history.json")
        if h:
            self.history = h.get("history", [])
        log.info("loaded state: king=%s@%s queue=%d seen=%d",
                 self.king.get("hf_repo", "none"),
                 (self.king.get("king_revision") or "none")[:12],
                 len(self.queue), len(self.seen))

    def flush(self):
        self.r2.put("state/validator_state.json", {
            "king": self.king, "queue": self.queue,
            "stats": self.stats, "counter": self.counter,
            "last_weight_block": self.last_weight_block,
            "last_winner_hotkey": self.last_winner_hotkey,
            "updated_at": _now(),
        })
        self.r2.put("state/queue.json", {"pending": self.queue, "updated_at": _now()})
        self.r2.put("king/current.json", self.king)
        self.r2.put("state/seen_hotkeys.json", {
            "hotkeys": sorted(self.seen), "updated_at": _now(),
        })

    def event(self, data):
        data.setdefault("timestamp", _now())
        self.r2.append_jsonl("state/history.jsonl", data)

    def next_id(self):
        self.counter += 1
        return f"eval-{self.counter:04d}"

    def enqueue(self, reveal):
        repo = reveal.get("hf_repo", "")
        hotkey = reveal.get("hotkey", "")
        king_hotkey = self.king.get("hotkey", "")
        if king_hotkey and hotkey == king_hotkey:
            log.info("skipping enqueue: hotkey %s is the current king", hotkey[:16])
            return None
        for existing in self.queue:
            if existing.get("hf_repo") == repo:
                log.info("skipping duplicate repo: %s already queued", repo)
                return None
        if repo in self.evaluated_repos:
            log.info("skipping %s: already evaluated this cycle", repo)
            return None
        cid = self.next_id()
        entry = {"challenge_id": cid, **reveal, "queued_at": _now()}
        self.queue.append(entry)
        self.stats["queued"] += 1
        self.flush()
        self.flush_dashboard()
        self.event({"event": "queued", **entry})
        return cid

    def set_king(self, hotkey, hf_repo, king_revision, block, challenge_id="seed"):
        global _seed_config
        _seed_config = None
        self.failed_repos.clear()
        self.evaluated_repos.clear()
        reign = self.king.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
        prev = self.king.copy() if self.king else None
        if prev:
            prev.pop("previous_king", None)
        self.king = {
            "hotkey": hotkey, "hf_repo": hf_repo,
            "king_revision": king_revision,
            "reign_number": reign, "crowned_at": _now(),
            "crowned_block": block, "challenge_id": challenge_id,
            "previous_king": prev,
        }
        self.flush()
        self.flush_dashboard()
        self.event({"event": "king_changed", "hotkey": hotkey, "reign": reign,
                     "challenge_id": challenge_id, "hf_repo": hf_repo,
                     "king_revision": king_revision})

    def record_verdict(self, verdict, challenger_repo, hotkey):
        king_loss = verdict["avg_king_loss"]
        chall_loss = verdict["avg_challenger_loss"]
        self.history.insert(0, {
            "challenge_id": verdict["challenge_id"],
            "hotkey": hotkey,
            "uid": self.uid_map.get(hotkey, "?"),
            "challenger_repo": challenger_repo,
            "accepted": verdict["accepted"],
            "verdict": verdict["verdict"],
            "mu_hat": verdict.get("mu_hat", 0),
            "lcb": verdict.get("lcb", 0),
            "delta": verdict.get("delta", 0),
            "avg_king_loss": king_loss,
            "avg_challenger_loss": chall_loss,
            "best_loss": min(king_loss, chall_loss),
            "wall_time_s": verdict["wall_time_s"],
            "timestamp": verdict["timestamp"],
        })
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def record_failure(self, entry, error_code, error_detail=""):
        self.history.insert(0, {
            "challenge_id": entry.get("challenge_id", "?"),
            "hotkey": entry.get("hotkey", ""),
            "uid": self.uid_map.get(entry.get("hotkey", ""), "?"),
            "challenger_repo": entry.get("hf_repo", ""),
            "accepted": False,
            "verdict": "error",
            "error_code": error_code,
            "error_detail": str(error_detail),
            "mu_hat": 0,
            "lcb": 0,
            "delta": 0,
            "avg_king_loss": 0,
            "avg_challenger_loss": 0,
            "best_loss": 0,
            "wall_time_s": 0,
            "timestamp": _now(),
        })
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def refresh_uid_map(self, subtensor, netuid):
        try:
            meta = subtensor.metagraph(netuid)
            self.uid_map = {hk: uid for uid, hk in enumerate(meta.hotkeys)}
        except Exception:
            log.warning("failed to refresh uid_map", exc_info=True)

    def replenish_reeval(self, subtensor, netuid):
        """Fill queue with re-eval candidates so the dashboard never shows empty."""
        self.evaluated_repos.clear()
        throwaway_seen = set()
        reeval_reveals = scan_reveals(subtensor, netuid, throwaway_seen)
        king_hk = self.king.get("hotkey", "")
        reeval_reveals = [r for r in reeval_reveals
                          if r["hotkey"] != king_hk
                          and r["hf_repo"] not in self.failed_repos]
        count = 0
        for rev in reeval_reveals:
            rev["reeval"] = True
            cid = self.enqueue(rev)
            if cid:
                count += 1
                log.info("queued %s from %s (re-eval)", cid, rev["hotkey"][:16])
        if count:
            log.info("replenished queue with %d re-eval candidates", count)
        return count

    def flush_dashboard(self):
        payload = {
            "updated_at": _now(),
            "king": self.king,
            "stats": self.stats,
            "current_eval": self.current_eval,
            "queue": [{"challenge_id": e.get("challenge_id"), "hotkey": e.get("hotkey"),
                        "uid": self.uid_map.get(e.get("hotkey", ""), "?"),
                        "hf_repo": e.get("hf_repo"), "queued_at": e.get("queued_at"),
                        "block": e.get("block"), "reeval": e.get("reeval", False)}
                       for e in self.queue],
            "history": self.history,
        }
        if self.market:
            payload["market"] = self.market
        self.r2.put_dashboard("dashboard.json", payload)



# ---------------------------------------------------------------------------
# King liveness
# ---------------------------------------------------------------------------

def check_king_alive(state):
    """Verify king repo is still accessible at pinned revision. Auto-dethrone if not."""
    repo = state.king.get("hf_repo", "")
    rev = state.king.get("king_revision", "")
    if not repo or not rev:
        return True
    try:
        HfApi(token=HF_TOKEN or None).model_info(repo, revision=rev)
        return True
    except Exception:
        log.warning("KING REPO UNAVAILABLE: %s@%s — auto-dethroning", repo, rev[:12])
        prev = state.king.get("previous_king")
        if prev and prev.get("hf_repo"):
            log.info("reverting to previous king: %s@%s",
                     prev["hf_repo"], prev.get("king_revision", "?")[:12])
            state.king = prev
            state.flush()
            state.flush_dashboard()
            state.event({"event": "king_dethroned_absent",
                         "lost_repo": repo, "lost_revision": rev[:12],
                         "reverted_to": prev.get("hf_repo")})
        else:
            log.error("no previous king to revert to — king repo is gone")
        return False


def enforce_king_health(state, wallet, subtensor, seed_id) -> bool:
    """Verify the current king passes structural checks against the seed.

    If it does not, walk down the previous_king chain until we find one that
    does, or fall back to the SEED_REPO. This is the circuit breaker that
    prevents a poisoned king from reigning forever just because no honest
    miner can statistically beat its tampered weights.

    Returns True if the king was changed.
    """
    repo = state.king.get("hf_repo", "")
    rev = state.king.get("king_revision", "")
    if not repo or not rev:
        return False

    rejection = validate_identity(repo, rev, seed_id)
    if rejection is None:
        return False

    log.error("KING %s@%s FAILS STRUCTURAL CHECK: %s — auto-dethroning",
              repo, rev[:12], rejection)
    state.event({"event": "king_dethroned_unhealthy",
                 "hf_repo": repo, "king_revision": rev,
                 "reason": rejection})

    candidate = state.king.get("previous_king")
    while candidate:
        c_repo = candidate.get("hf_repo", "")
        c_rev = candidate.get("king_revision", "")
        if not c_repo or not c_rev:
            candidate = candidate.get("previous_king")
            continue
        try:
            HfApi(token=HF_TOKEN or None).model_info(c_repo, revision=c_rev)
        except Exception as e:
            log.warning("previous king %s@%s unreachable, walking back: %s",
                        c_repo, c_rev[:12], e)
            candidate = candidate.get("previous_king")
            continue
        if validate_identity(c_repo, c_rev, seed_id) is not None:
            log.warning("previous king %s@%s also fails structural check, walking back",
                        c_repo, c_rev[:12])
            candidate = candidate.get("previous_king")
            continue
        log.info("reverting to healthy previous king %s@%s",
                 c_repo, c_rev[:12])
        state.set_king(candidate.get("hotkey", ""), c_repo, c_rev,
                       candidate.get("crowned_block", 0),
                       challenge_id=candidate.get("challenge_id", "auto-revert"))
        maybe_set_weights(subtensor, wallet, state, force=True,
                          reason="king_dethroned_unhealthy revert")
        return True

    # No healthy ancestor — fall back to the seed.
    try:
        seed_rev = HfApi(token=HF_TOKEN or None).model_info(SEED_REPO).sha
    except Exception:
        log.exception("FATAL: could not resolve seed king during recovery")
        return False
    log.warning("falling all the way back to seed king %s@%s",
                SEED_REPO, seed_rev[:12])
    state.set_king(wallet.hotkey.ss58_address, SEED_REPO, seed_rev,
                   subtensor.block, challenge_id="seed-revert")
    maybe_set_weights(subtensor, wallet, state, force=True,
                      reason="king_dethroned_unhealthy seed-revert")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def process_challenge(state, r2, entry, subtensor, wallet, *, check_stale=True):
    cid = entry["challenge_id"]
    hotkey = entry["hotkey"]
    hf_repo = entry["hf_repo"]
    challenger_revision = entry["challenger_revision"]
    entry_king_revision = entry["king_revision"]
    log.info("processing %s from %s repo=%s@%s", cid, hotkey[:16], hf_repo,
             challenger_revision[:12])

    king_hotkey = state.king.get("hotkey", "")
    if king_hotkey and hotkey == king_hotkey:
        log.info("skipping %s: challenger hotkey %s is the current king", cid, hotkey[:16])
        return

    if hf_repo in state.failed_repos:
        log.info("skipping %s: repo %s previously failed", cid, hf_repo)
        return

    if hf_repo in state.evaluated_repos:
        log.info("skipping %s: repo %s already evaluated this cycle", cid, hf_repo)
        return

    # Stale check uses full SHA equality. The reveal commits a specific king
    # by its 40-hex git revision; if the current king has rotated since, we
    # drop the entry rather than evaluate against a king the miner did not
    # target.
    current_king_revision = state.king.get("king_revision", "")
    if check_stale and current_king_revision and entry_king_revision != current_king_revision:
        log.info("stale %s: targets king rev %s but current is %s",
                 cid, entry_king_revision[:12], current_king_revision[:12])
        state.event({"event": "stale", "challenge_id": cid, "hotkey": hotkey,
                     "entry_king_revision": entry_king_revision,
                     "current_king_revision": current_king_revision})
        return

    # Verify the pinned challenger revision actually exists on HF. We never
    # follow ``main`` — the only revision we ever read is the one the miner
    # committed in the reveal payload.
    try:
        HfApi(token=HF_TOKEN or None).model_info(hf_repo, revision=challenger_revision)
    except Exception as exc:
        log.warning("challenger revision unreachable %s@%s: %s",
                    hf_repo, challenger_revision[:12], exc)
        state.failed_repos.add(hf_repo)
        state.record_failure(entry, "hf_revision_unreachable", str(exc))
        return

    seed_id = _load_seed_identity()
    rejection = validate_identity(hf_repo, challenger_revision, seed_id)
    if rejection:
        log.warning("rejecting %s (%s@%s): %s", cid, hf_repo,
                    challenger_revision[:12], rejection)
        state.failed_repos.add(hf_repo)
        state.record_failure(entry, "config_rejected", rejection)
        state.event({"event": "config_rejected", "challenge_id": cid,
                     "hf_repo": hf_repo, "reason": rejection})
        return

    block_hash = "default"
    eval_block = 0
    try:
        eval_block = subtensor.block
        block_hash = subtensor.get_block_hash(eval_block) or "default"
    except Exception:
        pass

    manifest = None
    manifest_attempts = 4
    for attempt in range(manifest_attempts):
        manifest = r2.ds_get("dataset/v2/manifest.json")
        if not manifest:
            manifest = r2.get("dataset/v1/manifest.json")
        if manifest:
            break
        if attempt < manifest_attempts - 1:
            backoff = 2 ** attempt
            log.warning("manifest fetch failed (attempt %d/%d), retrying in %ds",
                        attempt + 1, manifest_attempts, backoff)
            await asyncio.sleep(backoff)
    if not manifest:
        log.error("no dataset manifest after %d attempts; re-queuing %s",
                  manifest_attempts, cid)
        state.queue.insert(0, entry)
        state.flush()
        return
    n_shards = manifest["total_shards"]
    seed_mat = f"{block_hash}:{hotkey}".encode()
    shard_idx = int.from_bytes(hashlib.blake2b(seed_mat, digest_size=8).digest(), "little") % n_shards
    shard_key = manifest["shards"][shard_idx]["key"]

    king_repo = state.king.get("hf_repo", SEED_REPO)
    king_revision = state.king.get("king_revision", "")

    r2.put(f"eval/{cid}/meta.json", {
        "challenge_id": cid, "king_repo": king_repo,
        "king_revision": king_revision,
        "challenger_repo": hf_repo, "challenger_revision": challenger_revision,
        "hotkey": hotkey,
        "N": EVAL_N, "alpha": EVAL_ALPHA, "delta": EVAL_DELTA, "shard": shard_key,
        "eval_block": eval_block, "block_hash": block_hash,
    })

    state.current_eval = {
        "challenge_id": cid, "challenger_repo": hf_repo, "hotkey": hotkey,
        "progress": 0, "total": EVAL_N, "mu_hat": 0,
        "avg_king_loss": 0, "avg_challenger_loss": 0,
        "started_at": _now(),
    }
    state.flush_dashboard()

    verdict = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        eval_payload = {
            "king_repo": king_repo,
            "challenger_repo": hf_repo,
            "block_hash": block_hash,
            "hotkey": hotkey,
            "shard_key": shard_key,
            "king_revision": king_revision,
            "challenger_revision": challenger_revision,
            "eval_n": EVAL_N,
            "alpha": EVAL_ALPHA,
            "delta": EVAL_DELTA,
            "seq_len": SEQ_LEN,
        }

        max_busy_retries = 20
        for attempt in range(max_busy_retries):
            resp = await client.post(f"{EVAL_SERVER_URL}/eval", json=eval_payload)
            if resp.status_code != 409:
                break
            log.warning("%s: eval server busy (attempt %d/%d), waiting 30s",
                        cid, attempt + 1, max_busy_retries)
            await asyncio.sleep(30)
        else:
            log.error("%s: eval server still busy after %d attempts, re-queuing",
                      cid, max_busy_retries)
            state.queue.insert(0, entry)
            state.current_eval = None
            state.flush()
            state.flush_dashboard()
            return

        resp.raise_for_status()
        eval_id = resp.json()["eval_id"]
        log.info("eval %s dispatched to eval server as %s", cid, eval_id)

        async with client.stream("GET", f"{EVAL_SERVER_URL}/eval/{eval_id}/stream",
                                  timeout=httpx.Timeout(1800.0)) as stream:
            async for line in stream.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])

                if event["type"] == "progress":
                    d = event["data"]
                    state.current_eval.update({
                        "progress": d.get("done", 0),
                        "total": d.get("total", EVAL_N),
                        "mu_hat": d.get("mu_hat", 0),
                        "avg_king_loss": d.get("avg_king_loss", 0),
                        "avg_challenger_loss": d.get("avg_challenger_loss", 0),
                    })
                    state.flush_dashboard()

                elif event["type"] == "verdict":
                    verdict = event["data"]
                    verdict["challenge_id"] = cid
                    break

                elif event["type"] == "error":
                    raise RuntimeError(f"eval server error: {event['data']}")

    if not verdict:
        raise RuntimeError("eval stream ended without verdict")

    r2.put(f"eval/{cid}/verdict.json", verdict)
    log.info("verdict: %s (mu_hat=%.6f lcb=%.6f delta=%.6f %.1fs)",
             verdict["verdict"], verdict.get("mu_hat", 0), verdict.get("lcb", 0),
             verdict.get("delta", 0), verdict["wall_time_s"])

    state.current_eval = None
    state.evaluated_repos.add(hf_repo)
    state.record_verdict(verdict, hf_repo, hotkey)

    accepted = verdict.get("accepted", False)
    if accepted:
        state.stats["accepted"] += 1
    else:
        state.stats["rejected"] += 1

    state.flush_dashboard()
    state.event({"event": "eval_completed", "challenge_id": cid,
                 "hotkey": hotkey, "accepted": accepted, **verdict})

    if accepted:
        log.info("DETHRONE! %s wins via %s (repo=%s rev=%s)",
                 hotkey[:16], cid, hf_repo, challenger_revision[:12])
        state.set_king(hotkey, hf_repo, challenger_revision,
                       entry.get("block", 0), cid)
        state.last_winner_hotkey = hotkey
        await notify_new_king(state.king, verdict)
        maybe_set_weights(subtensor, wallet, state, force=True,
                          reason=f"new king via {cid}")

    state.flush()


async def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not EVAL_SERVER_URL:
        log.error("set TEUTONIC_EVAL_SERVER")
        sys.exit(1)

    r2 = R2()
    state = State(r2)
    state.load()
    if args.seen:
        log.info("--seen: will re-evaluate old challengers when queue is empty")
    else:
        log.info("new-only mode: will idle when all hotkeys have been seen")

    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.subtensor(network=NETWORK)
    state.refresh_uid_map(subtensor, NETUID)
    state.flush_dashboard()

    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            html_bytes = f.read()
        r2.put_dashboard_raw("index.html", html_bytes, "text/html")
        log.info("uploaded dashboard to Hippius")

    if not state.king:
        try:
            seed_info = HfApi(token=HF_TOKEN or None).model_info(SEED_REPO)
            seed_revision = seed_info.sha
            log.info("seed king %s at revision %s", SEED_REPO, seed_revision[:12])
        except Exception:
            log.error("could not resolve seed king revision from %s — refusing to start",
                      SEED_REPO)
            sys.exit(1)
        state.set_king(wallet.hotkey.ss58_address, SEED_REPO, seed_revision,
                       subtensor.block)

    # Resolve the immutable seed identity (config + tokenizer hashes +
    # tensor shape map). This is the trust anchor for every identity check.
    try:
        seed_id = _load_seed_identity()
    except Exception:
        log.exception("FATAL: could not establish seed identity")
        sys.exit(1)

    # Heal a king that no longer matches the seed identity: walk back
    # through previous_king until we find one that does, or fall all the
    # way back to the seed itself.
    enforce_king_health(state, wallet, subtensor, seed_id)
    last_king_health_check = time.time()

    maybe_set_weights(subtensor, wallet, state, force=True, reason="startup")

    # Verify eval server is reachable
    try:
        r = httpx.get(f"{EVAL_SERVER_URL}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        log.info("eval server healthy: %s", health)
    except Exception:
        log.warning("eval server at %s not reachable at startup (will retry on eval)", EVAL_SERVER_URL)

    def _on_signal(sig, frame):
        log.info("received signal %d, shutting down", sig)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    log.info("validator running | king=%s@%s | eval_server=%s | poll=%ds",
             state.king.get("hf_repo", "?"),
             state.king.get("king_revision", "?")[:12],
             EVAL_SERVER_URL, POLL_INTERVAL)

    while True:
        try:
            if not check_king_alive(state):
                log.warning("king repo check failed, skipping this tick")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            state.refresh_uid_map(subtensor, NETUID)

            tmc = await fetch_tmc_data()
            if tmc:
                state.market = tmc

            reveals = scan_reveals(subtensor, NETUID, state.seen)
            if reveals:
                state.flush()
                for rev in reveals:
                    cid = state.enqueue(rev)
                    if cid:
                        log.info("queued %s from %s (new)", cid, rev["hotkey"][:16])

            while state.queue:
                entry = state.queue.pop(0)
                is_reeval = entry.get("reeval", False)
                state.current_eval = {
                    "challenge_id": entry.get("challenge_id", "?"),
                    "challenger_repo": entry.get("hf_repo", ""),
                    "hotkey": entry.get("hotkey", ""),
                    "progress": 0, "total": EVAL_N, "mu_hat": 0,
                    "avg_king_loss": 0, "avg_challenger_loss": 0,
                    "loading": True,
                    "started_at": _now(),
                }
                state.flush_dashboard()
                state.flush()
                try:
                    await process_challenge(state, r2, entry, subtensor, wallet,
                                            check_stale=not is_reeval and args.seen)
                except Exception as exc:
                    log.exception("eval failed: %s", entry.get("challenge_id"))
                    state.stats["failed"] += 1
                    state.record_failure(entry, "eval_error", str(exc))
                    state.current_eval = None
                    state.flush_dashboard()

                fresh = scan_reveals(subtensor, NETUID, state.seen)
                if fresh:
                    state.flush()
                    for rev in fresh:
                        cid = state.enqueue(rev)
                        if cid:
                            log.info("queued %s from %s (new, mid-cycle)", cid, rev["hotkey"][:16])
                    new_items = [e for e in state.queue if not e.get("reeval")]
                    reeval_items = [e for e in state.queue if e.get("reeval")]
                    state.queue = new_items + reeval_items

                try:
                    maybe_set_weights(subtensor, wallet, state,
                                      reason="in-queue interval")
                except Exception:
                    log.exception("in-queue weight-set failed")

            state.current_eval = None

            if args.seen and not state.queue:
                state.replenish_reeval(subtensor, NETUID)

            if not args.seen and not state.queue:
                log.info("idle: all hotkeys seen, waiting for new submissions")

            state.flush_dashboard()

            try:
                maybe_set_weights(subtensor, wallet, state,
                                  reason="periodic interval")
            except Exception:
                log.exception("periodic weight-set failed")

            # Periodic king health re-check. The structural fingerprint of a
            # pinned revision can never change, but the king itself can be
            # rotated mid-loop by an accept path; recompute and fall back if
            # the new king turned out to be poisoned.
            if time.time() - last_king_health_check >= KING_HEALTH_INTERVAL_S:
                try:
                    if enforce_king_health(state, wallet, subtensor, seed_id):
                        log.info("king health enforcement triggered a revert")
                except Exception:
                    log.exception("periodic king health check failed")
                last_king_health_check = time.time()

        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("tick error")

        await asyncio.sleep(POLL_INTERVAL)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seen", action=argparse.BooleanOptionalAction, default=True,
                   help="When idle, replenish queue with re-eval candidates (default: on). "
                        "Use --no-seen to only evaluate genuinely new hotkeys and idle "
                        "when the queue is empty.")
    return p.parse_args()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()