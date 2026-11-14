"""Derive the shipped ortholog table ``orthologs_human_mouse_one2one.tsv`` (issues #9, #14).

Ensembl one-to-one human->mouse orthologs (protein-coding human genes, orthology confidence 1),
restricted to mouse Ensembl ``gene_id``s present in the ``mm10-evo2-v1`` gene table, with the
gene-table row (``emb_idx``) attached. The join is on ``gene_id``, never on the mouse symbol.

Source of the pairs (``--source``):

* ``house`` (default): the raw BioMart response archived by ``~/code/dnaemb`` on 2026-05-08
  (``biomart_raw_response.tsv``). Its Ensembl release was identified on 2026-11-13 by replaying
  the same query against the archive marts: release 115 (``sep2025.archive.ensembl.org``) is
  row-identical; 113, 114 and 116 are not.
* ``biomart``: re-query BioMart (``--release`` selects the archive host; the current release
  when omitted). Use this to regenerate the table for a newer release.

The one2one / confidence-1 filter and the removal of pairs with a duplicated human or mouse
symbol reproduce ``dnaemb/scripts/download_orthologs.py`` exactly (15,732 pairs for release 115).

Usage::

    python tools/make_orthologs.py --genes <assets>/references/mm10-evo2-v1/genes.parquet \
        --out <assets>/references/mm10-evo2-v1/orthologs_human_mouse_one2one.tsv \
        --legacy house=$BLOOM_LEGACY_ORTHOLOGS/orthologs_one2one.tsv \
        --legacy mouseuniverse=$BLOOM_LEGACY_ORTHOLOGS/orthologs_one2one_mouseuniverse.tsv \
        --report tools/phase1_outputs/orthologs_derivation.json

Prints the sha256 of the written TSV (docs/standards.md: tools print the asset hashes they produce).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HOUSE_RAW = Path(os.environ.get("BLOOM_LEGACY_ORTHOLOGS", "")) / "biomart_raw_response.tsv"
HOUSE_RELEASE = 115
HOUSE_RELEASE_EVIDENCE = (
    "biomart_raw_response.tsv (downloaded 2026-05-08) is row-identical to the "
    "sep2025.archive.ensembl.org (Ensembl 115) mart response on 2026-11-13; releases 113, 114 "
    "and 116 differ"
)
# Ensembl archive hosts (https://www.ensembl.org/help/articles/archives)
ARCHIVE_HOSTS = {113: "oct2024", 114: "may2025", 115: "sep2025", 116: "jun2026"}
BIOMART_QUERY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="biotype" value="protein_coding"/>
    <Filter name="with_mmusculus_homolog" excluded="0"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="mmusculus_homolog_ensembl_gene"/>
    <Attribute name="mmusculus_homolog_associated_gene_name"/>
    <Attribute name="mmusculus_homolog_orthology_type"/>
    <Attribute name="mmusculus_homolog_orthology_confidence"/>
  </Dataset>
</Query>"""
COLUMNS = [
    "human_gene_id",
    "human_gene_name",
    "mouse_gene_id",
    "mouse_gene_name",
    "orthology_type",
    "orthology_confidence",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_biomart(text: str) -> pd.DataFrame:
    """Raw BioMart TSV -> standard columns (same cleaning as dnaemb.utils.orthologs)."""
    df = pd.read_csv(StringIO(text), sep="\t", dtype={5: "Int64"})
    df.columns = COLUMNS
    df = df.dropna(subset=["human_gene_id", "mouse_gene_id"])
    df = df[
        df["human_gene_id"].str.startswith("ENSG") & df["mouse_gene_id"].str.startswith("ENSMUSG")
    ]
    return df.reset_index(drop=True)


def one2one_confident(df: pd.DataFrame) -> pd.DataFrame:
    """ortholog_one2one & confidence 1, then drop pairs whose human or mouse symbol is duplicated."""
    sel = df[(df["orthology_type"] == "ortholog_one2one") & (df["orthology_confidence"] == 1)]
    sel = sel.reset_index(drop=True)
    human_dups = sel["human_gene_name"].duplicated(keep=False)
    mouse_dups = sel["mouse_gene_name"].duplicated(keep=False)
    return sel[~human_dups & ~mouse_dups].reset_index(drop=True)


def fetch_biomart(release: int | None) -> tuple[str, str]:
    import requests

    host = (
        "https://www.ensembl.org"
        if release is None
        else f"https://{ARCHIVE_HOSTS[release]}.archive.ensembl.org"
    )
    r = requests.get(f"{host}/biomart/martservice", params={"query": BIOMART_QUERY}, timeout=900)
    r.raise_for_status()
    if not r.text.startswith("Gene stable ID"):
        raise RuntimeError(f"unexpected BioMart response from {host}: {r.text[:200]}")
    if release is None:
        release = requests.get(
            "https://rest.ensembl.org/info/data/",
            headers={"Content-Type": "application/json"},
            timeout=60,
        ).json()["releases"][0]
    return r.text, str(release)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--genes", required=True, type=Path, help="genes.parquet of the reference release"
    )
    ap.add_argument("--out", required=True, type=Path, help="output TSV")
    ap.add_argument("--source", choices=["house", "biomart"], default="house")
    ap.add_argument(
        "--raw", type=Path, default=HOUSE_RAW, help="raw BioMart TSV for --source house"
    )
    ap.add_argument(
        "--release", type=int, default=None, help="Ensembl release for --source biomart"
    )
    ap.add_argument(
        "--legacy",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="legacy table(s) to diff against",
    )
    ap.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    args = ap.parse_args(argv)

    if args.source == "house":
        raw_text = args.raw.read_text()
        release, evidence, source_file = str(HOUSE_RELEASE), HOUSE_RELEASE_EVIDENCE, str(args.raw)
    else:
        raw_text, release = fetch_biomart(args.release)
        evidence = "queried directly from the mart named in the source line"
        source_file = "live query"
    raw = parse_biomart(raw_text)
    pairs = one2one_confident(raw)

    genes = pq.read_table(args.genes).to_pandas()
    if "emb_idx" not in genes.columns:
        genes["emb_idx"] = range(len(genes))
    assert genes["gene_id"].is_unique and genes["gene_name"].is_unique
    emb_of_id = pd.Series(genes["emb_idx"].to_numpy(), index=genes["gene_id"].to_numpy())
    name_of_id = pd.Series(genes["gene_name"].to_numpy(), index=genes["gene_id"].to_numpy())

    in_universe = pairs["mouse_gene_id"].isin(emb_of_id.index)
    kept = pairs[in_universe].copy()
    dropped = pairs[~in_universe].copy()
    kept["emb_idx"] = emb_of_id.loc[kept["mouse_gene_id"]].to_numpy().astype("int64")
    kept = kept.sort_values("emb_idx").reset_index(drop=True)
    assert (
        kept["human_gene_id"].is_unique
        and kept["mouse_gene_id"].is_unique
        and kept["emb_idx"].is_unique
    )
    symbol_drift = int(
        (
            name_of_id.loc[kept["mouse_gene_id"]].to_numpy() != kept["mouse_gene_name"].to_numpy()
        ).sum()
    )
    out_cols = ["human_gene_id", "human_gene_name", "mouse_gene_id", "mouse_gene_name", "emb_idx"]

    today = datetime.date.today().isoformat()
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except OSError:
        git_sha = "unknown"
    header = [
        "# orthologs_human_mouse_one2one.tsv: Ensembl one-to-one human->mouse orthologs restricted to the "
        "mm10-evo2-v1 gene universe (bloom-biorxiv, issue #9)",
        "# source: Ensembl BioMart hsapiens_gene_ensembl; filters biotype=protein_coding, with_mmusculus_homolog; "
        "attributes ensembl_gene_id, external_gene_name, mmusculus_homolog_ensembl_gene, "
        "mmusculus_homolog_associated_gene_name, mmusculus_homolog_orthology_type, "
        "mmusculus_homolog_orthology_confidence",
        f"# ensembl_release: {release}",
        f"# release_evidence: {evidence}",
        f"# source_file: {source_file}",
        "# filter: mmusculus_homolog_orthology_type == ortholog_one2one AND orthology_confidence == 1; pairs with a "
        "duplicated human or mouse symbol removed; mouse_gene_id restricted to genes.parquet.gene_id",
        f"# universe: {args.genes.name} ({len(genes)} genes, sha256 {sha256(args.genes)})",
        "# join: mouse_gene_id == genes.parquet.gene_id (never on the mouse symbol); emb_idx = gene-table row; "
        f"mouse_gene_name is the Ensembl {release} symbol and differs from genes.parquet.gene_name for "
        f"{symbol_drift} pairs",
        f"# counts: biomart_rows={len(raw)} one2one_confidence1={len(pairs)} kept={len(kept)} "
        f"dropped_mouse_gene_id_not_in_universe={len(dropped)}",
        f"# generated: {today} by tools/make_orthologs.py (bloom-biorxiv {git_sha})",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n") as fh:
        fh.write("\n".join(header) + "\n")
        kept[out_cols].to_csv(fh, sep="\t", index=False, lineterminator="\n")
    digest = sha256(args.out)
    print(f"wrote {args.out}  rows={len(kept)}  sha256={digest}")
    print(
        f"biomart rows {len(raw)}; one2one/conf1 {len(pairs)}; kept {len(kept)}; dropped {len(dropped)}; "
        f"symbol drift {symbol_drift}"
    )

    report: dict = {
        "output": str(args.out),
        "sha256": digest,
        "ensembl_release": release,
        "release_evidence": evidence,
        "source": args.source,
        "source_file": source_file,
        "counts": {
            "biomart_rows": len(raw),
            "one2one_confidence1": len(pairs),
            "kept": len(kept),
            "dropped_mouse_gene_id_not_in_universe": len(dropped),
            "mouse_symbol_drift_vs_gene_table": symbol_drift,
        },
        "dropped_pairs": dropped[
            ["human_gene_id", "human_gene_name", "mouse_gene_id", "mouse_gene_name"]
        ].to_dict(orient="records"),
        "legacy": {},
    }
    new_pairs = set(zip(kept["human_gene_id"], kept["mouse_gene_id"], strict=True))
    for spec in args.legacy:
        name, path = spec.split("=", 1)
        leg = pd.read_csv(path, sep="\t", comment="#", dtype=str)
        leg_pairs = set(zip(leg["human_gene_id"], leg["mouse_gene_id"], strict=True))
        only_new = sorted(new_pairs - leg_pairs)
        only_legacy = sorted(leg_pairs - new_pairs)
        report["legacy"][name] = {
            "path": path,
            "rows": len(leg),
            "common": len(new_pairs & leg_pairs),
            "only_in_new": len(only_new),
            "only_in_legacy": len(only_legacy),
            "only_in_new_examples": only_new[:10],
            "only_in_legacy_examples": only_legacy[:10],
        }
        print(
            f"vs {name} ({len(leg)} rows): common {len(new_pairs & leg_pairs)}, only new {len(only_new)}, "
            f"only legacy {len(only_legacy)}"
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
