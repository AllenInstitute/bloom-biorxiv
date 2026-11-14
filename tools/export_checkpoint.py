"""Phase 0 step 1: export hyacinth weights (model.* keys, remapped) to safetensors + config.json,
then verify round-trip bitwise (TEST_PLAN M4)."""

import dataclasses
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, OrderedDict

import torch
import yaml
from _paths import env_str
from safetensors.torch import load_file, save_file

CKPT = env_str("BLOOM_TRAINING_CKPT")
CFG = env_str("BLOOM_TRAINING_CONFIG")
CACHE = env_str("BLOOM_EVO2_CACHE").rstrip("/") + "/"
OUT = f"{env_str('BLOOM_STAGING')}/models/hyacinth"
TRAINING_REPO = env_str("BLOOM_TRAINING_REPO")
ARCH_WORKTREE = env_str("BLOOM_ARCH_WORKTREE")
#: identifier of the training run as recorded in config.json provenance (the released run is `hyacinth`)
SOURCE_RUN = os.environ.get("BLOOM_SOURCE_RUN", "hyacinth")
REMAP = OrderedDict([
    ("model.bloom_first_layer.", "first_layer."),
    ("model.encoder.", "encoder."),
    ("model.decoder.", "decoder."),
    ("model.feature_emb.", "feature_emb."),
])
os.makedirs(OUT, exist_ok=True)
summary = {}
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(o):
    if dataclasses.is_dataclass(o):
        return {k: jsonable(v) for k, v in dataclasses.asdict(o).items()}
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if hasattr(o, "__dict__"):
        return {k: jsonable(v) for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


import bloom_arch

tick(f"bloom_arch from {bloom_arch.__file__}")
summary["bloom_arch_file"] = bloom_arch.__file__

# ---- 1. load checkpoint (mmap) ----
ckpt = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
tick(f"ckpt loaded; top-level keys: {list(ckpt.keys())}")
summary["ckpt_top_level_keys"] = list(ckpt.keys())
summary["global_step"] = int(ckpt["global_step"])
summary["epoch"] = int(ckpt["epoch"])
summary["pytorch_lightning_version"] = ckpt.get("pytorch-lightning_version")
sd = ckpt["state_dict"]
dropped = Counter(k.split(".")[0] for k in sd if not k.startswith("model."))
summary["dropped_key_groups"] = {
    k: {
        "n_tensors": n,
        "n_elements": int(sum(sd[kk].numel() for kk in sd if kk.split(".")[0] == k)),
    }
    for k, n in dropped.items()
}
tick(f"non-model key groups dropped: {dict(dropped)}")

out = OrderedDict()
other_prefixes = Counter()
dtypes = Counter()
for k, v in sd.items():
    if not k.startswith("model."):
        continue
    for src, dst in REMAP.items():
        if k.startswith(src):
            nk = dst + k[len(src) :]
            break
    else:
        other_prefixes[k.split(".")[1]] += 1
        nk = k[len("model.") :]
    if v.dtype != torch.float32:
        dtypes[str(v.dtype)] += 1
    out[nk] = v.detach().clone().contiguous()
summary["other_model_prefixes_kept_stripped"] = dict(other_prefixes)
summary["non_fp32_tensors"] = dict(dtypes)
summary["n_tensors_exported"] = len(out)
summary["n_elements_exported"] = int(sum(t.numel() for t in out.values()))
tick(
    f"{len(out)} tensors, {summary['n_elements_exported']:,} elements; other prefixes: {dict(other_prefixes)}; non-fp32: {dict(dtypes)}"
)
if dtypes:
    print("WARNING: non-fp32 tensors present — exporting as-is (not cast)", flush=True)

# ---- 2. write safetensors ----
st_path = os.path.join(OUT, "model.safetensors")
meta = {
    "format": "pt",
    "model_id": "hyacinth",
    "reference_id": "mm10-evo2-v1",
    "source_run": SOURCE_RUN,
    "global_step": str(summary["global_step"]),
    "epoch": str(summary["epoch"]),
}
save_file(out, st_path, metadata=meta)
summary["safetensors_bytes"] = os.path.getsize(st_path)
summary["safetensors_sha256"] = sha256(st_path)
tick(
    f"wrote {st_path} ({summary['safetensors_bytes'] / 1e6:.1f} MB) sha256={summary['safetensors_sha256']}"
)

# ---- 3. live module: scale + param counts ----
from bloom.model.model import LitBloom

lit = LitBloom.load_from_checkpoint(
    CKPT, map_location="cpu", weights_only=False, embedding_cache_path=CACHE
)
lit.eval()
tick("LitBloom.load_from_checkpoint done")
gard = lit.model
scale = float(gard.bloom_first_layer.attn_block.seq_heads._scale)
expected = 1.0 / math.sqrt(32)
summary["seq_head_scale"] = scale
summary["seq_head_scale_expected"] = expected
if not math.isclose(scale, expected, rel_tol=0, abs_tol=1e-12):
    print(f"ABORT: seq_head_scale {scale} != 1/sqrt(32) {expected}", flush=True)
    json.dump(summary, open("export_summary.json", "w"), indent=2)
    sys.exit(2)
tick(f"seq_head_scale = {scale} == 1/sqrt(32) OK")
total_params = sum(p.numel() for p in gard.parameters())
per_module = {name: sum(p.numel() for p in m.parameters()) for name, m in gard.named_children()}
persistent_buffers = [n for n, _ in gard.named_buffers() if n in gard.state_dict()]
nonpersistent_buffers = [n for n, _ in gard.named_buffers() if n not in gard.state_dict()]
summary["n_parameters_total"] = int(total_params)
summary["n_parameters_per_module"] = {k: int(v) for k, v in per_module.items()}
summary["persistent_buffers"] = persistent_buffers
summary["nonpersistent_buffers"] = nonpersistent_buffers
summary["n_gardenia_state_dict_tensors"] = len(gard.state_dict())
tick(
    f"params total={total_params:,} per-module={per_module}; persistent buffers={persistent_buffers}; non-persistent={nonpersistent_buffers}"
)

# ---- 4. verification A: safetensors vs original ckpt tensors, bitwise ----
loaded = load_file(st_path)
inv = OrderedDict((v, k) for k, v in REMAP.items())


def inverse(nk):
    for dst, src in inv.items():
        if nk.startswith(dst):
            return src + nk[len(dst) :]
    return "model." + nk


mism = []
orig_keys = {k for k in sd if k.startswith("model.")}
inv_keys = {inverse(k) for k in loaded}
assert inv_keys == orig_keys, f"key set mismatch: {len(inv_keys ^ orig_keys)}"
for nk, t in loaded.items():
    o = sd[inverse(nk)]
    if not (o.dtype == t.dtype and o.shape == t.shape and torch.equal(o, t)):
        mism.append(nk)
summary["verify_A_safetensors_vs_ckpt_bitwise"] = {
    "n_compared": len(loaded),
    "n_mismatch": len(mism),
    "mismatch_keys": mism[:20],
}
tick(f"verify A: {len(loaded)} tensors compared, {len(mism)} mismatches")

# ---- 5. verification B: fresh LitBloom from ckpt hparams; strict load of inverse-remapped tensors; bitwise vs live model ----
hp = dict(ckpt["hyper_parameters"])
hp["embedding_cache_path"] = None
hp["compile_model"] = False
summary["hparams_keys"] = sorted(hp.keys())
fresh = LitBloom(**hp)
fresh.eval()
inv_sd = {inverse(k)[len("model.") :]: v for k, v in loaded.items()}
res = fresh.model.load_state_dict(inv_sd, strict=True)
live_sd = gard.state_dict()
fresh_sd = fresh.model.state_dict()
assert set(live_sd) == set(fresh_sd)
mism_b = [k for k in live_sd if not torch.equal(live_sd[k], fresh_sd[k])]
summary["verify_B_fresh_strict_load"] = {
    "missing": list(res.missing_keys),
    "unexpected": list(res.unexpected_keys),
    "n_compared": len(live_sd),
    "n_mismatch_vs_live": len(mism_b),
}
tick(
    f"verify B: strict load ok (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}); {len(mism_b)} mismatches vs live model"
)
fresh_total = sum(p.numel() for p in fresh.model.parameters())
assert fresh_total == total_params

# ---- 6. config.json ----
cfg = yaml.safe_load(open(CFG))
m = cfg["model"]
ba = dict(m["bloom_attention"])
st = cfg["strategy"]
ba["seq_head_scale"] = scale
export_commit = subprocess.check_output(
    ["git", "-C", TRAINING_REPO, "rev-parse", "HEAD"], text=True
).strip()
arch_commit = subprocess.check_output(
    ["git", "-C", ARCH_WORKTREE, "rev-parse", "HEAD"],
    text=True,
).strip()
ckpt_sha = sha256(CKPT)
tick(f"ckpt sha256 {ckpt_sha}")
config = OrderedDict([
    ("model_id", "hyacinth"),
    ("reference_id", "mm10-evo2-v1"),
    ("config_version", "1.0"),
    (
        "architecture",
        OrderedDict([
            ("d_cseq", m["d_cseq"]),
            ("d_descriptor", m["d_descriptor"]),
            ("d_metadata", m["d_metadata"]),
            ("d_measurement", m["d_measurement"]),
            ("d_model", m["d_model"]),
            ("n_latents", m["n_latents"]),
            ("n_transformer_layers", m["n_transformer_layers"]),
            ("n_heads", m["n_heads"]),
            ("decoder_hidden_dim", m["decoder_hidden_dim"]),
            ("dropout", m["dropout"]),
            ("attention", ba),
        ]),
    ),
    ("embedding_config_hparams", jsonable(hp["config"])),
    (
        "training_context",
        OrderedDict([
            ("Lmax", st["Lmax"]),
            ("Lq", st["Lq"]),
            ("Lmin", st["Lmin"]),
            ("binarize_input_genes", st["binarize_input_genes"]),
            ("binarize_input_peaks", st["binarize_input_peaks"]),
            ("binarize_query_genes", st["binarize_query_genes"]),
            ("binarize_query_peaks", st["binarize_query_peaks"]),
            ("peak_modality", st["peak_modality"]),
            ("mask_metadata_keys", st["mask_metadata_keys"]),
            ("strategy_target", st["_target_"]),
            ("train_pool", os.path.basename(st["pool_path"])),
            ("precision", cfg["trainer"]["precision"]),
        ]),
    ),
    (
        "strategy_defaults",
        {
            "gene": {"strategy": "random", "lmax": 8192, "draws": 4},
            "peak": {"strategy": "random", "lmax": 8192, "draws": 16},
            "top": {"lmax": 4096},
        },
    ),
    (
        "parameters",
        OrderedDict([
            ("total", int(total_params)),
            ("per_module", {k: int(v) for k, v in per_module.items()}),
            ("n_state_dict_tensors", len(out)),
            ("n_state_dict_elements", summary["n_elements_exported"]),
            ("persistent_buffers", persistent_buffers),
            ("dtype", "float32"),
        ]),
    ),
    ("state_dict_remap", dict(REMAP)),
    (
        "dropped_from_checkpoint",
        {
            "state_dict_groups": sorted(dropped.keys()),
            "other": ["optimizer_states", "lr_schedulers", "loops", "callbacks"],
        },
    ),
    (
        "weights",
        {
            "file": "model.safetensors",
            "size_bytes": summary["safetensors_bytes"],
            "sha256": summary["safetensors_sha256"],
        },
    ),
    (
        "source",
        OrderedDict([
            ("repo", "alleninstitute/bloom"),
            ("training_commit", "671c8cb"),
            ("export_commit", export_commit),
            ("run", SOURCE_RUN),
            ("checkpoint", os.path.basename(CKPT)),
            ("checkpoint_alias", "3458-step=530000.ckpt"),
            ("checkpoint_sha256", ckpt_sha),
            ("global_step", summary["global_step"]),
            ("epoch", summary["epoch"]),
            ("pytorch_lightning_version", summary["pytorch_lightning_version"]),
            ("torch_version_at_export", torch.__version__),
            ("bloom_arch_repo", "AllenInstitute/bloom-sprint-attn"),
            ("bloom_arch_commit", arch_commit),
            (
                "export_date",
                datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            ("export_host", "maintainer GPU host (docs/maintainers.md)"),
            ("export_env", os.path.basename(sys.prefix)),
        ]),
    ),
])
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
json.dump(config, open("config.json", "w"), indent=2)
summary["elapsed_s"] = time.time() - T0
json.dump(summary, open("export_summary.json", "w"), indent=2)
tick("DONE")
