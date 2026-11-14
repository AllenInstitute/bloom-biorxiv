"""Run alleninstitute/bloom evaluate.py unchanged, but assert the seq-head scale on every model load
(before any prediction) so a parity failure can never be caused by the bloom-arch scale.

Environment overrides: ``BLOOM_MATMUL_PRECISION`` (``highest`` = TF32 off; evaluate.py's hard-coded ``medium`` is
then ignored) and ``BLOOM_MHA_FASTPATH`` (``0`` = unfused ``nn.TransformerEncoder`` math, issue #27)."""

import math
import os
import sys

import bloom_arch

print(f"[scale-check] bloom_arch imported from {bloom_arch.__file__}", flush=True)
print(f"[scale-check] PYTHONPATH={os.environ.get('PYTHONPATH')}", flush=True)
from bloom.model import model as M

_orig = M.LitBloom.load_from_checkpoint
EXPECTED = 1.0 / math.sqrt(32)


def _checked(cls, *a, **k):
    m = _orig(*a, **k)
    s = float(m.model.bloom_first_layer.attn_block.seq_heads._scale)
    print(
        f"[scale-check] model.model.bloom_first_layer.attn_block.seq_heads._scale = {s!r} (expected 1/sqrt(32) = {EXPECTED!r})",
        flush=True,
    )
    assert math.isclose(s, EXPECTED, rel_tol=0, abs_tol=1e-12), f"seq_head_scale {s} != 1/sqrt(32)"
    print(f"[scale-check] OK; params={sum(p.numel() for p in m.model.parameters()):,}", flush=True)
    return m


M.LitBloom.load_from_checkpoint = classmethod(_checked)
import torch

print(
    f"[env] torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}",
    flush=True,
)
mp = os.environ.get("BLOOM_MATMUL_PRECISION")
if mp:
    torch.set_float32_matmul_precision(mp)
    _orig_set = torch.set_float32_matmul_precision

    def _ignore(p):
        print(
            f"[tf32-check] evaluate.py requested float32_matmul_precision={p!r}; keeping {mp!r} (BLOOM_MATMUL_PRECISION)",
            flush=True,
        )

    torch.set_float32_matmul_precision = _ignore
    if (
        mp == "highest"
    ):  # TF32 fully off: matmul (set above) and cuDNN (no convolutions in Bloom; recorded anyway)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
print(
    f"[tf32-check] float32_matmul_precision at start = {torch.get_float32_matmul_precision()!r}; "
    f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}",
    flush=True,
)
# Phase 1 review (#27): BLOOM_MHA_FASTPATH=0 disables PyTorch's fused CUDA nn.TransformerEncoder fast path so the
# TF32-off references use the same unfused math as CPU; unset / 1 keeps the torch default (fast path on, as the
# paper's evaluations and the paper-faithful TF32 references).
fp = os.environ.get("BLOOM_MHA_FASTPATH")
if fp is not None:
    torch.backends.mha.set_fastpath_enabled(fp.strip().lower() not in ("0", "false", "off"))
print(
    f"[fastpath-check] torch.backends.mha.get_fastpath_enabled() = {torch.backends.mha.get_fastpath_enabled()} "
    f"(BLOOM_MHA_FASTPATH={fp!r})",
    flush=True,
)
import runpy

EVAL = os.path.expanduser("~/code/bloom/src/bloom/evaluate.py")
print(f"[env] running {EVAL} as __main__ (argv: {sys.argv[1:]})", flush=True)
sys.argv[0] = EVAL
runpy.run_path(EVAL, run_name="__main__")
