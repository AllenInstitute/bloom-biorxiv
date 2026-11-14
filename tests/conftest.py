"""Shared fixtures: synthetic reference tables (always), the real gene table and the 20-cell fixture (if present).

The real `mm10-evo2-v1` gene table (`genes.parquet` 1.7 MB + `genes.safetensors` 99 MB) and the 20-cell
multiome fixture `examples/multiome_20/*` (~8 MB) are not in git; they live on the public bucket
`s3://bloom-biorxiv` (docs/assets.md). The fixtures below look for them under `$BLOOM_HOME`, try one
anonymous `bloom.assets.download` from the default source when they are missing, and skip the test with a
message otherwise -- CI (`unit or regression`) stays green either way. `BLOOM_FETCH_FIXTURES=0` disables
the attempt (offline runs).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REFERENCE_ID = "mm10-evo2-v1"

#: Fixtures whose presence means a test needs the released weights (497 MB). Such tests get the `weights`
#: marker automatically (issue #51): CI runs `(unit or regression) and not weights` on pull requests (~3 min)
#: and `weights and not assets` on pushes to main / nightly, after `bloom download --what model,genes,orthologs`.
WEIGHTS_FIXTURES = ("hyacinth_model_dir",)


def pytest_collection_modifyitems(config, items):
    """Add the `weights` marker to every test whose fixture closure includes :data:`WEIGHTS_FIXTURES`."""
    for item in items:
        names = getattr(item, "fixturenames", ())
        if any(f in names for f in WEIGHTS_FIXTURES) and item.get_closest_marker("weights") is None:
            item.add_marker(pytest.mark.weights)


# Real rows of genes.parquet (mm10-evo2-v1) used by R1/R4/R5 so that the synthetic table and
# the real gene table agree on the expected emb_idx values.
REAL_GENES = [
    # emb_idx, gene_name, gene_id, chrom, start, end, strand
    (25359, "Pvalb", "ENSMUSG00000005716", "chr15", 78206150, 78206650, "-"),
    (2498, "Gad1", "ENSMUSG00000070880", "chr2", 70561792, 70562292, "+"),
]
# Real rows (emb_idx / gene_id from genes.parquet) plus one synthetic row (Atp5o) that models symbol drift between the ortholog table ("Atp5po") and the gene table ("Atp5o"),
# a gene whose only route is the symbol fallback, and filler rows.
SYNTH_GENES = [
    (100, "Atp5o", "ENSMUSG00000022956", "chr16", 91900000, 91900500, "+"),
    (1938, "Gad2", "ENSMUSG00000026787", "chr2", 22621955, 22622455, "+"),
    (26231, "Sst", "ENSMUSG00000004366", "chr16", 23890708, 23891208, "-"),
    (16951, "Vip", "ENSMUSG00000019772", "chr10", 5638968, 5639468, "+"),
    (0, "Gm1992", "ENSMUSG00000089699", "chr1", 3466337, 3466837, "+"),
    (2, "Xkr4", "ENSMUSG00000051951", "chr1", 3671248, 3671748, "-"),
]
SYNTH_PEAKS = [
    (0, "chr1", 3003635, 3004136),
    (1, "chr1", 3007195, 3007696),
    (7, "chr2", 100000, 100501),
    (12, "chrX", 5000, 5501),
]
# human_gene_id, human_gene_name, mouse_gene_id, mouse_gene_name
SYNTH_ORTHOLOGS = [
    ("ENSG00000100362", "PVALB", "ENSMUSG00000005716", "Pvalb"),
    ("ENSG00000128683", "GAD1", "ENSMUSG00000070880", "Gad1"),
    ("ENSG00000241837", "ATP5PO", "ENSMUSG00000022956", "Atp5po"),  # symbol drift, id present
    ("ENSG00000136750", "GAD2", "ENSMUSG00000099999", "Gad2"),  # id absent -> symbol fallback
    ("ENSG00000157005", "SST", "ENSMUSG00000004366", "Sst"),
    ("ENSG00000146469", "VIP", "ENSMUSG00000019772", "Vip"),
    ("ENSG00000000001", "NOTINGENETABLE", "ENSMUSG00000088888", "Notingenetable"),
]


def _write_tables(d: Path, genes: list[tuple], peaks: list[tuple], orthologs: list[tuple]) -> None:
    import torch
    from safetensors.torch import save_file

    rng = np.random.default_rng(0)
    g = pd.DataFrame(
        genes, columns=["emb_idx", "gene_name", "gene_id", "chrom", "start", "end", "strand"]
    ).sort_values("emb_idx")
    g["emb_idx"] = g["emb_idx"].astype(np.int32)
    g["obs_name"] = [f"{c}:{s}-{e}" for c, s, e in zip(g.chrom, g.start, g.end, strict=True)]
    pq.write_table(pa.Table.from_pandas(g, preserve_index=False), d / "genes.parquet")
    save_file(
        {
            "cseq": torch.from_numpy(rng.standard_normal((len(g), 768), dtype=np.float32)),
            "chr": torch.from_numpy(rng.integers(0, 21, len(g), dtype=np.int8)),
            "bp": torch.from_numpy(((g.start + g.end) // 2).to_numpy(copy=True).astype(np.int32)),
            "emb_idx": torch.from_numpy(g["emb_idx"].to_numpy(copy=True)),
        },
        str(d / "genes.safetensors"),
        metadata={"reference_id": REFERENCE_ID, "kind": "gene", "n_rows": str(len(g))},
    )
    p = pd.DataFrame(peaks, columns=["emb_idx", "chrom", "start", "end"])
    p["emb_idx"] = p["emb_idx"].astype(np.int32)
    p["name"] = [f"{c}:{s}-{e}" for c, s, e in zip(p.chrom, p.start, p.end, strict=True)]
    pq.write_table(pa.Table.from_pandas(p, preserve_index=False), d / "peaks.parquet")
    save_file(
        {
            "cseq": torch.from_numpy(rng.standard_normal((len(p), 768), dtype=np.float32)),
            "chr": torch.from_numpy(rng.integers(0, 21, len(p), dtype=np.int8)),
            "bp": torch.from_numpy(((p.start + p.end) // 2).to_numpy(copy=True).astype(np.int32)),
        },
        str(d / "peaks.safetensors"),
        metadata={"reference_id": REFERENCE_ID, "kind": "peak", "n_rows": str(len(p))},
    )
    lines = [
        "# orthologs_human_mouse_one2one.tsv (synthetic test slice)",
        "# source: synthetic",
        "# ensembl_release: 0",
        "human_gene_id\thuman_gene_name\tmouse_gene_id\tmouse_gene_name",
        *["\t".join(r) for r in orthologs],
    ]
    (d / "orthologs_human_mouse_one2one.tsv").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="session")
def synthetic_reference_dir(tmp_path_factory) -> Path:
    """Release-layout directory with a 8-gene / 4-peak synthetic reference and a 7-row ortholog slice."""
    d = tmp_path_factory.mktemp("ref") / "references" / REFERENCE_ID
    d.mkdir(parents=True)
    _write_tables(d, REAL_GENES + SYNTH_GENES, SYNTH_PEAKS, SYNTH_ORTHOLOGS)
    return d


def bloom_home() -> Path:
    return Path(os.environ.get("BLOOM_HOME", Path.home() / ".cache" / "bloom")).expanduser()


def _try_fetch(paths: list[str]) -> bool:
    """One `bloom download` attempt of the given manifest paths into `BLOOM_HOME` (default source).

    Returns False when disabled (`BLOOM_FETCH_FIXTURES=0`), offline, or when any file is not
    delivered; never raises, so a fixture can fall back to `pytest.skip`.
    """
    if os.environ.get("BLOOM_FETCH_FIXTURES", "1") == "0":
        return False
    try:
        from bloom import assets

        by_path = assets.manifest().by_path()
        results = assets.download(
            files=[by_path[p] for p in paths], dest=bloom_home(), source=assets.default_source()
        )
    except Exception:  # any network / IO problem means "not available"
        return False
    return all(r.ok for r in results)


GENE_TABLE_PATHS = [
    f"references/{REFERENCE_ID}/genes.parquet",
    f"references/{REFERENCE_ID}/genes.safetensors",
]
ORTHOLOGS_PATH = f"references/{REFERENCE_ID}/orthologs_human_mouse_one2one.tsv"
MULTIOME_20_PATHS = [
    "examples/multiome_20/rna.h5ad",
    "examples/multiome_20/atac.h5ad",
    "examples/multiome_20/cells.tsv",
    "examples/multiome_20/reference_embeddings.npz",
    "examples/multiome_20/reference_embeddings.json",
]


@pytest.fixture(scope="session")
def gene_table_dir() -> Path:
    """`$BLOOM_HOME/references/mm10-evo2-v1` holding genes.{parquet,safetensors}; skip if unavailable."""
    d = bloom_home() / "references" / REFERENCE_ID
    files = ["genes.parquet", "genes.safetensors"]
    if not all((d / f).exists() for f in files):
        _try_fetch(GENE_TABLE_PATHS)
    if not all((d / f).exists() for f in files):
        pytest.skip(
            f"gene table not found under {d} and `bloom download --what genes` from the public bucket "
            "did not fetch it (offline?); see docs/assets.md"
        )
    return d


@pytest.fixture(scope="session")
def gene_table_paths(gene_table_dir) -> tuple[Path, Path]:
    """``(genes.safetensors, genes.parquet)`` of the real ``mm10-evo2-v1`` gene table."""
    return gene_table_dir / "genes.safetensors", gene_table_dir / "genes.parquet"


@pytest.fixture(scope="session")
def shipped_orthologs_tsv() -> Path:
    """The shipped ortholog table under `$BLOOM_HOME/references/mm10-evo2-v1`; skip if unavailable."""
    d = bloom_home() / "references" / REFERENCE_ID
    f = d / "orthologs_human_mouse_one2one.tsv"
    if not f.exists():
        _try_fetch([ORTHOLOGS_PATH])
    if not f.exists():
        pytest.skip(f"{f} not found; `bloom download --what orthologs`, see docs/assets.md")
    return f


@pytest.fixture(scope="session")
def multiome_20_dir() -> Path:
    """`$BLOOM_HOME/examples/multiome_20` (rna/atac h5ad, cells.tsv, reference_embeddings.{npz,json}); skip if unavailable."""
    d = bloom_home() / "examples" / "multiome_20"
    names = [p.rsplit("/", 1)[-1] for p in MULTIOME_20_PATHS]
    if not all((d / n).exists() for n in names):
        _try_fetch(MULTIOME_20_PATHS)
    if not all((d / n).exists() for n in names):
        pytest.skip(
            f"multiome_20 fixture not found under {d}; `bloom download --what examples` (docs/assets.md)"
        )
    return d


PEAK_PARQUET_PATH = f"references/{REFERENCE_ID}/peaks.parquet"
PEAK_SAFETENSORS_PATH = f"references/{REFERENCE_ID}/peaks.safetensors"
DA_PANEL_PATH = f"references/{REFERENCE_ID}/panels/da_2127.txt"


@pytest.fixture(scope="session")
def peak_table_dir() -> Path:
    """`$BLOOM_HOME/references/mm10-evo2-v1` with the gene table + `peaks.parquet` (72 MB); skip if unavailable.

    Matching (R7-R10, `prepare --atac`) needs only the parquet; `peaks.safetensors` (5.4 GB) is
    required by the embedding tests through :func:`peak_tensors_dir`.
    """
    d = bloom_home() / "references" / REFERENCE_ID
    files = ["genes.parquet", "genes.safetensors", "peaks.parquet"]
    if not all((d / f).exists() for f in files):
        _try_fetch([*GENE_TABLE_PATHS, PEAK_PARQUET_PATH])
    if not all((d / f).exists() for f in files):
        pytest.skip(
            f"peak table not found under {d}; `bloom download --what genes,peaks` (docs/assets.md)"
        )
    return d


@pytest.fixture(scope="session")
def peak_tensors_dir(peak_table_dir) -> Path:
    """`peak_table_dir` with `peaks.safetensors` present too (5.4 GB; never auto-fetched here)."""
    if not (peak_table_dir / "peaks.safetensors").exists():
        pytest.skip(
            f"{peak_table_dir / 'peaks.safetensors'} not found; `bloom download --what peaks` (5.4 GB)"
        )
    return peak_table_dir


@pytest.fixture(scope="session")
def da_panel_path() -> Path:
    """The shipped DA peak panel `panels/da_2127.txt`; skip if unavailable."""
    f = bloom_home() / DA_PANEL_PATH
    if not f.exists():
        _try_fetch([DA_PANEL_PATH])
    if not f.exists():
        pytest.skip(f"{f} not found; `bloom download --what panels`")
    return f


RNA_100_REFS = Path(__file__).resolve().parent / "data" / "ref" / "rna_100_reference_embeddings.npz"


def load_reference(npz: Path, name: str) -> np.ndarray:
    """One reference array (float32 ``(n_cells, 1024)``) from a consolidated ``reference_embeddings.npz``."""
    with np.load(npz) as z:
        if name not in z.files:
            raise KeyError(f"{name!r} not in {npz} (arrays: {sorted(z.files)})")
        return np.asarray(z[name], dtype=np.float32)


@pytest.fixture(scope="session")
def rna_100_refs() -> dict[str, np.ndarray]:
    """The committed rna_100 references: ``rna_top4096``, ``rna_top4096_tf32off``, ``rna_random_8192x4``, ``obs_names``."""
    with np.load(RNA_100_REFS) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------
# model track (#13): hyacinth config / weights / gene table locations and a tiny ModelConfig
# --------------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
HYACINTH_CONFIG = REPO_ROOT / "tests" / "data" / "hyacinth_config.json"


@pytest.fixture(scope="session")
def hyacinth_config_path() -> Path:
    """Committed copy of the released ``models/hyacinth/config.json`` (byte-identical to the staged file)."""
    return HYACINTH_CONFIG


@pytest.fixture(scope="session")
def hyacinth_model_dir() -> Path:
    """``$BLOOM_HOME/models/hyacinth`` with ``model.safetensors``; skips when absent (not on the release)."""
    d = bloom_home() / "models" / "hyacinth"
    if not (d / "model.safetensors").is_file():
        pytest.skip(
            f"hyacinth weights not found at {d / 'model.safetensors'}; run "
            "`bloom download --what model` (497 MB from the public bucket) or set BLOOM_HOME to a "
            "tree that mirrors the release layout"
        )
    return d


@pytest.fixture(scope="session")
def tiny_config():
    """A ~100 KB ``ModelConfig`` for construction / device / masking tests without assets."""
    import math

    from bloom.config import AttentionConfig, ModelConfig

    return ModelConfig(
        d_cseq=16,
        d_descriptor=12,
        d_metadata=6,
        d_measurement=4,
        d_model=32,
        n_latents=4,
        n_transformer_layers=1,
        n_heads=2,
        decoder_hidden_dim=8,
        dropout=0.0,
        attention=AttentionConfig(
            d_val=4,
            d_u=4,
            K=4,
            d_v=8,
            n_sequence_heads=2,
            n_position_heads=2,
            n_standard_heads=1,
            max_G=4,
            seq_head_scale=1.0 / math.sqrt(8.0),
        ),
        state_dict_remap={
            "model.bloom_first_layer.": "first_layer.",
            "model.encoder.": "encoder.",
            "model.decoder.": "decoder.",
            "model.feature_emb.": "feature_emb.",
        },
    )


def total_ram_gb() -> float:
    """Physical RAM in GB (0.0 when the platform does not expose it).

    The CPU path materialises ``B x H x L x L`` attention per batch (`top` 4096: ~1.4 GB per view), so the
    regression tests pick their CPU batch size from this: 8 on the maintainers' host, 1 on a 16 GB CI runner
    (results do not depend on batch size, G8). G8's batch-32 case (~42 GB) skips below 48 GB.
    """
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def cpu_batch_size(default: int = 8, per_view_gb: float = 1.4, headroom_gb: float = 6.0) -> int:
    """``default`` when the machine has RAM for it, else 1 (CI runners have ~16 GB)."""
    ram = total_ram_gb()
    if ram and ram < default * per_view_gb + headroom_gb:
        return 1
    return default
