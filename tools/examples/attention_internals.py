"""Seq-head pair logits and cross-attention on two fixture cells (docs/internals.md section 6).

Usage (assets downloaded: ``bloom download --what model,genes``; ~20 s on CPU, 5 s on an L4)::

    <prefix>/bin/python tools/examples/attention_internals.py [--cells 2] [--device auto]

Prints the sequence-head logits for a few gene pairs of cell 0, the tokens latent 0 attends to and
the saliency norm of the five most influential tokens for cell 0.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import anndata as ad
import torch

import bloom
from bloom import pair_logits
from bloom.strategies import TopExpressed, cell_rng
from bloom.tokens import collate

REPO = Path(__file__).resolve().parents[2]
GENES_OF_INTEREST = ("Malat1", "Snhg11", "Meg3", "Gad1", "Gad2", "Slc17a7", "Pvalb", "Sst")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rna", default=str(REPO / "tests" / "data" / "rna_100.h5ad"))
    ap.add_argument("--cells", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=4096)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model = bloom.load_model("hyacinth", device=args.device)
    adata = ad.read_h5ad(args.rna)[: args.cells].copy()
    ds = bloom.prepare(rna=adata, species="mouse", reference=model.reference)
    strat = TopExpressed(lmax=args.lmax)
    views = [strat.views(ds.cell_features(i, "rna"), cell_rng(0, i))[0] for i in range(ds.n_obs)]
    batch = collate(views).to(model.device)

    out = model.encode(batch.emb_idx, batch.kind, batch.lengths, return_internals=True)
    print(
        f"device {model.device}; B x L = {out.batch_size} x {out.n_tokens}; lengths {out.lengths.tolist()}"
    )
    print(f"seq scale {out.scales['seq']:.17g} (1/sqrt 32 = {1 / math.sqrt(32):.17g})")
    for g, (q, k) in out.qk.items():
        print(f"  qk[{g!r}]: q {tuple(q.shape)} k {tuple(k.shape)} scale {out.scales[g]:.4f}")

    ids0 = out.token_ids[0]
    genes = [g for g in GENES_OF_INTEREST if g in ids0][:4] or ids0[:4]
    pos = [out.position_of(g, 0) for g in genes]
    pairs = [(i, j) for i in pos for j in pos if i != j]
    logits = pair_logits(out, pairs, group="seq", batch=0)  # (P, 8)
    print("\nsequence-head logits q_i . k_j / sqrt(32), cell 0 (query -> key):")
    for (i, j), row in zip(pairs, logits.tolist(), strict=True):
        print(f"  {ids0[i]:>8} -> {ids0[j]:<8} " + " ".join(f"{v:+6.2f}" for v in row))

    top = out.cross_attn[0, 0].topk(5).indices.tolist()
    print("\nlatent 0 of cell 0 attends most to:", [ids0[t] for t in top])
    print(
        "cross_attn row sums:",
        out.cross_attn.sum(-1).min().item(),
        out.cross_attn.sum(-1).max().item(),
    )

    g = model.encode(
        batch.emb_idx[:1],
        batch.kind[:1],
        batch.lengths[:1],
        return_internals=True,
        requires_grad=True,
    )
    g.h_cell.norm().backward()
    sal = g.token_h.grad.norm(dim=-1)[0, : int(g.lengths[0])]  # (L_real,)
    top = sal.topk(5).indices.tolist()
    print(
        "\nsaliency |d||h_cell||/d token_h| top-5 tokens, cell 0:",
        [(ids0[t], round(sal[t].item(), 4)) for t in top],
    )
    with torch.inference_mode():
        plain = model.encode(batch.emb_idx, batch.kind, batch.lengths)
    print("h_cell bitwise equal to encode():", torch.equal(plain, out.h_cell))


if __name__ == "__main__":
    main()
