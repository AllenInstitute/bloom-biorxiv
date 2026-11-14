"""``bloom`` command-line interface.

Commands: ``bloom info``, ``bloom download``, ``bloom prepare``, ``bloom embed`` (docs/architecture.md
section 3.1). Inputs are one AnnData per modality (``--rna``, ``--atac``, or both); per-modality
strategy overrides are ``--rna-lmax`` / ``--atac-strategy`` ... (also spelled ``--rna.lmax``).
Exit codes: 0 ok, 1 user error with a one-line fix, 2 not implemented.
"""

import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import typer

from bloom import __version__, assets
from bloom.assets import bloom_home

app = typer.Typer(
    name="bloom",
    help="Bloom cell embeddings from single-cell RNA / ATAC AnnData (inference only).",
    no_args_is_help=True,
    add_completion=False,
)

NOT_IMPLEMENTED_EXIT_CODE = 2

__all__ = ["NOT_IMPLEMENTED_EXIT_CODE", "app", "bloom_home"]


def _fmt_bytes(n: int) -> str:
    """Human-readable size (``1.2 GB``)."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{n} B"


@app.command()
def info(
    verify: bool = typer.Option(
        False, "--verify", help="Hash every cached asset (slow for the 5.4 GB peak table)."
    ),
) -> None:
    """Print package, Python and torch versions, CUDA availability, BLOOM_HOME and asset status."""
    import torch

    cuda = torch.cuda.is_available()
    m = assets.manifest()
    rows = [
        ("bloom", __version__),
        ("python", f"{platform.python_version()} ({sys.executable})"),
        ("platform", platform.platform()),
        ("torch", torch.__version__),
        ("cuda available", str(cuda)),
    ]
    if cuda:
        rows.append(("cuda device", torch.cuda.get_device_name(0)))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        rows.append(("mps available", str(mps.is_available())))
    rows.append(("BLOOM_HOME", str(bloom_home())))
    rows.append(("model id", m.model_id))
    rows.append(("reference id", m.reference_id))
    rows.append(("examples version", m.examples_version))
    rows.append(("manifest", f"v{m.manifest_version} sha256 {m.sha256[:16]}... ({m.created})"))
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        typer.echo(f"{key.ljust(width)}  {value}")

    typer.echo("")
    mode = "sha256 verified" if verify else "size check only; `bloom info --verify` hashes"
    typer.echo(f"assets in BLOOM_HOME ({mode}):")
    statuses = assets.group_status(checksum=verify)
    for st in statuses:
        detail = (
            f"{st.n_present}/{st.n_files} files, "
            f"{_fmt_bytes(st.bytes_present)} of {_fmt_bytes(st.bytes_total)}"
        )
        if st.n_corrupt:
            bad = ", ".join(r.asset.name for r in st.results if r.status.endswith("mismatch"))
            detail += f"; corrupt: {bad}"
        if st.n_partial:
            detail += f"; {st.n_partial} partial download(s)"
        typer.echo(f"  {st.group.ljust(10)} {st.state.ljust(8)} {detail}")
    missing = [st.group for st in statuses if st.state != "present"]
    if missing:
        typer.echo(f"  -> fetch with `bloom download --what {','.join(missing)}`")


@app.command()
def download(
    what: str = typer.Option(
        "all",
        "--what",
        help="Comma-separated groups: " + ",".join(assets.GROUPS) + " (or `all`).",
    ),
    dir: Path | None = typer.Option(
        None, "--dir", help="Cache root (default $BLOOM_HOME or ~/.cache/bloom)."
    ),
    force: bool = typer.Option(False, "--force", help="Re-fetch files even if they verify."),
    source: str | None = typer.Option(
        None,
        "--source",
        help=(
            "bucket (default; the public bucket, anonymous), url:<base> (any mirror of the "
            "layout), or file://<root> (staging directory; symlinks, append ?copy to copy). "
            "Also read from $BLOOM_ASSET_SOURCE."
        ),
    ),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Hash cached files before skipping them (default on)."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 1 if any file of a requested group is not in the source."
    ),
) -> None:
    """Download and verify asset groups (model, genes, peaks, orthologs, panels, examples).

    Idempotent: files that match MANIFEST.json (size + sha256) are kept, corrupt ones re-fetched,
    partial downloads resumed (HTTP Range). The default source is the public bucket
    https://bloom-biorxiv.s3.us-west-2.amazonaws.com/ (no credentials). Exits 1 when a requested
    group has no file in the source (e.g. a file:// tree without the weights) or a transfer fails
    verification; a group the source carries only partially is completed with a warning (use
    --strict to fail instead).
    """
    try:
        groups = assets.parse_groups(what)
        src = assets.source_from_spec(source)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    root = dir if dir is not None else bloom_home()
    typer.echo(f"source  {src.name}")
    typer.echo(f"dest    {root}")
    failed_groups: list[str] = []
    partial_groups: list[str] = []
    for g in groups:
        entries = assets.manifest().group(g)
        total = _fmt_bytes(sum(e.size_bytes for e in entries))
        typer.echo(f"[{g}] {len(entries)} file(s), {total}")
        try:
            results = assets.download(
                files=entries,
                dest=root,
                force=force,
                source=src,
                verify_existing=verify,
                log=lambda msg: typer.echo(f"  {msg}"),
            )
        except assets.AssetError as exc:
            typer.echo(f"  error: {exc}", err=True)
            failed_groups.append(g)
            continue
        unavailable = [r.asset for r in results if r.status == "unavailable"]
        failed = [r for r in results if r.status == "failed"]
        if unavailable:
            typer.echo(f"  {src.describe_unavailable(unavailable)}", err=True)
        for r in failed:
            typer.echo(f"  {r.message}", err=True)
        if failed or (unavailable and (strict or len(unavailable) == len(results))):
            failed_groups.append(g)
        elif unavailable:
            partial_groups.append(g)
    if partial_groups:
        typer.echo(
            f"partial group(s): {', '.join(partial_groups)} (files not in this source skipped; "
            "`--strict` to fail instead)",
            err=True,
        )
    if failed_groups:
        typer.echo(
            f"incomplete group(s): {', '.join(failed_groups)} (see messages above)", err=True
        )
        raise typer.Exit(code=1)
    typer.echo(
        "done: "
        + ", ".join(f"{g} ok" if g not in partial_groups else f"{g} partial" for g in groups)
    )


SPECIES = ("mouse", "human")
PEAK_MATCH = ("auto", "exact", "overlap")


def _fail(message: str, code: int = 1) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=code)


@app.command()
def prepare(
    rna: Path | None = typer.Option(None, "--rna", help="RNA AnnData (.h5ad), cells x genes."),
    atac: Path | None = typer.Option(
        None, "--atac", help="ATAC AnnData (.h5ad), cells x peaks (mm10 coordinates in var_names)."
    ),
    species: str | None = typer.Option(
        None,
        "--species",
        help="mouse | human (required with --rna; ATAC alone implies mouse, peaks are mm10 only).",
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output bundle directory."),
    gene_col: str | None = typer.Option(
        None, "--gene-col", help="var column with gene identifiers (default: var_names)."
    ),
    layer: str | None = typer.Option(
        None,
        "--layer",
        help="RNA layer to take values from (default precedence: layers['counts'] > raw.X > X).",
    ),
    peak_match: str = typer.Option(
        "auto",
        "--peak-match",
        help="auto | exact | overlap (auto = exact when >= 95% of the peaks match exactly).",
    ),
    min_overlap: float = typer.Option(
        0.5, "--min-overlap", help="Overlap mode: fraction of the 501-bp reference peak covered."
    ),
    keep_unmatched: bool = typer.Option(
        False,
        "--keep-unmatched",
        help="With --rna and --atac: keep cells present in one file only (rna_only / atac_only).",
    ),
    force_species: bool = typer.Option(
        False, "--force-species", help="Skip the species-mismatch guard on the gene identifiers."
    ),
    reference_dir: Path | None = typer.Option(
        None,
        "--reference-dir",
        help="Reference tables directory (default $BLOOM_HOME/references/mm10-evo2-v1).",
    ),
) -> None:
    """Map AnnData(s) onto the reference tables and write a prepared bundle (rna.h5ad / atac.h5ad + manifest.json).

    Prints the mapping report (mapping rate, detected mass mapped, detected features per cell,
    peak matching mode, warnings). Only `X > 0` is used downstream; values are kept for `top`.
    """
    import warnings

    from bloom.prepare import prepare as _prepare
    from bloom.reference import ReferenceTables

    if species is not None and species not in SPECIES:
        _fail(f"--species must be mouse or human, got {species!r}", code=2)
    if rna is None and atac is None:
        _fail("give at least one of --rna / --atac", code=2)
    if species is None:
        if rna is not None:
            _fail("--species mouse|human is required with --rna", code=2)
        species = "mouse"  # ATAC alone: mm10 peaks
    if peak_match not in PEAK_MATCH:
        _fail(f"--peak-match must be one of {', '.join(PEAK_MATCH)}, got {peak_match!r}")
    if not 0.0 < min_overlap <= 1.0:
        _fail(f"--min-overlap must be in (0, 1], got {min_overlap}")
    for p, flag in ((rna, "--rna"), (atac, "--atac")):
        if p is not None and not p.exists():
            _fail(f"{flag} file {p} does not exist")
    try:
        ref = ReferenceTables.from_dir(reference_dir, require_peaks=atac is not None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # every warning is also in the printed report
            ds = _prepare(
                rna=rna,
                atac=atac,
                species=species,
                reference=ref,
                gene_col=gene_col,
                layer=layer,
                peak_match=peak_match,
                min_overlap=min_overlap,
                keep_unmatched=keep_unmatched,
                force_species=force_species,
            )
        ds.write(output)
    except NotImplementedError as exc:
        _fail(str(exc), code=NOT_IMPLEMENTED_EXIT_CODE)
    except KeyError as exc:
        _fail(str(exc.args[0]) if exc.args else str(exc))
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    typer.echo(ds.report_text())
    files = list(ds.manifest()["files"].values()) + list(ds.manifest()["unmatched_files"].values())
    typer.echo(f"wrote {output / 'manifest.json'} and {', '.join(files)}")


def _modality_overrides(
    generic: dict[str, object], rna: dict[str, object], atac: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Generic flags apply to every modality; ``--rna.*`` / ``--atac.*`` override per modality."""
    out: dict[str, dict[str, object]] = {}
    for mod, spec in (("rna", rna), ("atac", atac)):
        block = dict(generic)
        block.update({k: v for k, v in spec.items() if v is not None})
        out[mod] = block
    return out


def _warn_overrides_for_absent_modalities(
    modalities: Sequence[str], **per_modality: dict[str, object]
) -> None:
    """Issue #47: ``--rna-*`` / ``--atac-*`` given for a modality that is not in the input are ignored; say so."""
    for mod, spec in per_modality.items():
        given = [f"--{mod}-{k}" for k, v in spec.items() if v is not None]
        if given and mod not in modalities:
            typer.echo(
                f"warning: {', '.join(given)} ignored: the input has no {mod} modality "
                f"(modalities: {', '.join(modalities)})",
                err=True,
            )


def _check_strategy_flag(strategy: str | None, panel: Path | None, flag: str) -> str | None:
    from bloom.strategies import ALIASES, PRESETS

    if strategy is None:
        return None
    key = strategy.lower()
    if key in ALIASES:
        typer.echo(
            f"warning: {flag} {strategy} was renamed to {ALIASES[key]}; the old name is accepted "
            "for one release",
            err=True,
        )
        key = ALIASES[key]
    if key not in PRESETS:
        _fail(f"{flag} must be one of {', '.join(PRESETS)}, got {strategy!r}")
    if key == "panel" and panel is None:
        _fail(f"{flag} panel needs a --panel (or --{flag[2:].split('.')[0]}.panel) feature list")
    return key


@app.command()
def embed(
    prepared_dir: Path | None = typer.Argument(
        None, help="Prepared bundle from `bloom prepare` (alternative to --rna / --atac)."
    ),
    rna: Path | None = typer.Option(None, "--rna", help="RNA AnnData (.h5ad), cells x genes."),
    atac: Path | None = typer.Option(
        None, "--atac", help="ATAC AnnData (.h5ad), cells x peaks (mm10 coordinates)."
    ),
    species: str | None = typer.Option(
        None,
        "--species",
        help="mouse | human (required with --rna on a raw .h5ad; ATAC alone implies mouse).",
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output embeddings .h5ad."),
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        help="random | top | panel | tile for every modality given (default: random, the paper's).",
    ),
    lmax: int | None = typer.Option(
        None, "--lmax", help="Tokens per view (default 8192; 4096 for `top`)."
    ),
    draws: int | None = typer.Option(
        None,
        "--draws",
        help=(
            "Views per cell: `random` draws exactly this many (default 4 RNA / 16 ATAC); `tile` emits "
            "ceil(n_detected / lmax) disjoint draws capped at this number (default 4)."
        ),
    ),
    universe: Path | None = typer.Option(
        None, "--universe", help="Feature list (one id per line) that restricts the detected set."
    ),
    panel: Path | None = typer.Option(
        None, "--panel", help="Ordered feature list that *is* the view (strategy panel)."
    ),
    rna_strategy: str | None = typer.Option(
        None, "--rna.strategy", "--rna-strategy", help="RNA override of --strategy."
    ),
    rna_lmax: int | None = typer.Option(None, "--rna.lmax", "--rna-lmax", help="RNA override."),
    rna_draws: int | None = typer.Option(None, "--rna.draws", "--rna-draws", help="RNA override."),
    rna_universe: Path | None = typer.Option(
        None, "--rna.universe", "--rna-universe", help="RNA override (gene list)."
    ),
    rna_panel: Path | None = typer.Option(
        None, "--rna.panel", "--rna-panel", help="RNA override (gene panel)."
    ),
    atac_strategy: str | None = typer.Option(
        None, "--atac.strategy", "--atac-strategy", help="ATAC override of --strategy."
    ),
    atac_lmax: int | None = typer.Option(None, "--atac.lmax", "--atac-lmax", help="ATAC override."),
    atac_draws: int | None = typer.Option(
        None, "--atac.draws", "--atac-draws", help="ATAC override."
    ),
    atac_universe: Path | None = typer.Option(
        None, "--atac.universe", "--atac-universe", help="ATAC override (peak list)."
    ),
    atac_panel: Path | None = typer.Option(
        None, "--atac.panel", "--atac-panel", help="ATAC override (peak panel)."
    ),
    peak_match: str = typer.Option(
        "auto", "--peak-match", help="ATAC: auto | exact | overlap (see `bloom prepare --help`)."
    ),
    min_overlap: float = typer.Option(
        0.5, "--min-overlap", help="ATAC overlap mode: fraction of the reference peak covered."
    ),
    keep_unmatched: bool = typer.Option(
        False,
        "--keep-unmatched",
        help=(
            "With --rna and --atac: also embed cells present in one file only, written to "
            "<output>_rna_only.h5ad / <output>_atac_only.h5ad."
        ),
    ),
    force_species: bool = typer.Option(
        False, "--force-species", help="Skip the species-mismatch guard on the gene identifiers."
    ),
    seed: int = typer.Option(0, "--seed", help="Base seed; cell i uses default_rng([seed, i])."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | cuda | mps."),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Views per forward pass (default 1 on CPU, 8 on CUDA)."
    ),
    precision: str = typer.Option("fp32", "--precision", help="fp32 (paper) | bf16 (GPU)."),
    keep_views: bool = typer.Option(
        False, "--keep-views", help="Also write per-view embeddings to obsm['X_bloom_<mod>_views']."
    ),
    gene_col: str | None = typer.Option(
        None, "--gene-col", help="var column with gene identifiers (default: var_names)."
    ),
    layer: str | None = typer.Option(
        None,
        "--layer",
        help="RNA layer to take values from (default: layers['counts'] > raw.X > X).",
    ),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", help="Model directory (default $BLOOM_HOME/models/hyacinth)."
    ),
    reference_dir: Path | None = typer.Option(
        None,
        "--reference-dir",
        help="Reference tables directory (default $BLOOM_HOME/references/mm10-evo2-v1).",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="No progress line / report."),
) -> None:
    """Embed cells from RNA and/or ATAC AnnData (or a prepared bundle) with the released model `hyacinth`.

    Each modality file is embedded independently with its own strategy block (defaults: random,
    8192 tokens, 4 draws for RNA / 16 for ATAC) into obsm['X_bloom_rna'] / obsm['X_bloom_atac'].
    With both files one output holds the cells present in both (aligned by obs_names).
    Exit codes: 0 ok; 1 user error (message names the fix); 2 not implemented.
    """
    import warnings

    from bloom import assets as _assets
    from bloom.embed import default_strategies, embed_unmatched, load_model
    from bloom.embed import embed as _embed
    from bloom.prepare import PreparedDataset
    from bloom.prepare import prepare as _prepare

    raw_inputs = [p for p in (rna, atac) if p is not None]
    if prepared_dir is not None and raw_inputs:
        _fail("give either PREPARED_DIR or --rna / --atac, not both")
    if prepared_dir is None and not raw_inputs:
        _fail("give --rna RNA.h5ad and/or --atac ATAC.h5ad (or a prepared bundle directory)")
    for p, flag in ((prepared_dir, "PREPARED_DIR"), (rna, "--rna"), (atac, "--atac")):
        if p is not None and not p.exists():
            _fail(f"{flag} input {p} does not exist")
    if species is not None and species not in SPECIES:
        _fail(f"--species must be mouse or human, got {species!r}")
    if peak_match not in PEAK_MATCH:
        _fail(f"--peak-match must be one of {', '.join(PEAK_MATCH)}, got {peak_match!r}")
    if not 0.0 < min_overlap <= 1.0:
        _fail(f"--min-overlap must be in (0, 1], got {min_overlap}")
    strategy = _check_strategy_flag(strategy, panel or rna_panel or atac_panel, "--strategy")
    rna_strategy = _check_strategy_flag(rna_strategy, rna_panel or panel, "--rna.strategy")
    atac_strategy = _check_strategy_flag(atac_strategy, atac_panel or panel, "--atac.strategy")
    if precision not in ("fp32", "bf16"):
        _fail(f"--precision must be fp32 or bf16, got {precision!r}")
    if device not in ("auto", "cpu", "cuda", "mps"):
        _fail(f"--device must be auto, cpu, cuda or mps, got {device!r}")
    for p, flag in (
        (universe, "--universe"),
        (panel, "--panel"),
        (rna_universe, "--rna.universe"),
        (rna_panel, "--rna.panel"),
        (atac_universe, "--atac.universe"),
        (atac_panel, "--atac.panel"),
    ):
        if p is not None and not p.is_file():
            _fail(f"{flag} file {p} does not exist")
    if output.suffix != ".h5ad":
        typer.echo(f"warning: output {output} does not end in .h5ad", err=True)

    try:
        model = load_model(
            "hyacinth", device=device, model_dir=model_dir, reference_dir=reference_dir
        )
    except _assets.AssetMissingError as exc:
        _fail(str(exc))
    except Exception as exc:  # scale mismatch, corrupt weights, ...
        _fail(f"could not load the model: {exc}")

    try:
        if prepared_dir is not None:
            ds = PreparedDataset.read(prepared_dir)
            if species is not None and species != ds.species:
                _fail(
                    f"bundle {prepared_dir} was prepared for species {ds.species!r}, not {species!r}"
                )
        else:
            if species is None:
                if rna is not None:
                    _fail("--species mouse|human is required for a raw RNA .h5ad input")
                species = "mouse"  # ATAC alone: mm10 peaks
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the report below prints every warning
                ds = _prepare(
                    rna=rna,
                    atac=atac,
                    species=species,
                    reference=model.reference,
                    gene_col=gene_col,
                    layer=layer,
                    peak_match=peak_match,
                    min_overlap=min_overlap,
                    keep_unmatched=keep_unmatched,
                    force_species=force_species,
                )
            if not quiet:
                typer.echo(ds.report_text(), err=True)
        _warn_overrides_for_absent_modalities(
            ds.modalities,
            rna={
                "strategy": rna_strategy,
                "lmax": rna_lmax,
                "draws": rna_draws,
                "universe": rna_universe,
                "panel": rna_panel,
            },
            atac={
                "strategy": atac_strategy,
                "lmax": atac_lmax,
                "draws": atac_draws,
                "universe": atac_universe,
                "panel": atac_panel,
            },
        )
        overrides = _modality_overrides(
            {
                "strategy": strategy,
                "lmax": lmax,
                "draws": draws,
                "universe": universe,
                "panel": panel,
            },
            {
                "strategy": rna_strategy,
                "lmax": rna_lmax,
                "draws": rna_draws,
                "universe": rna_universe,
                "panel": rna_panel,
            },
            {
                "strategy": atac_strategy,
                "lmax": atac_lmax,
                "draws": atac_draws,
                "universe": atac_universe,
                "panel": atac_panel,
            },
        )
        strategies = default_strategies(ds, model, overrides)
        if not quiet:
            for mod in ds.modalities:
                typer.echo(
                    f"strategy {mod}: {strategies[mod]!r}; device {model.device}; precision "
                    f"{precision}; seed {seed}",
                    err=True,
                )
        out = _embed(
            model,
            ds,
            strategies,
            batch_size=batch_size,
            seed=seed,
            device=None,
            precision=precision,
            keep_views=keep_views,
            progress=not quiet,
        )
        extra = (
            embed_unmatched(
                model,
                ds,
                strategies,
                batch_size=batch_size,
                seed=seed,
                device=None,
                precision=precision,
                keep_views=keep_views,
                progress=not quiet,
            )
            if ds.unmatched
            else {}
        )
    except typer.Exit:  # a `_fail` inside the block (click's Exit subclasses RuntimeError)
        raise
    except NotImplementedError as exc:
        _fail(str(exc), code=NOT_IMPLEMENTED_EXIT_CODE)
    except KeyError as exc:  # str(KeyError) wraps the message in quotes
        _fail(str(exc.args[0]) if exc.args else str(exc))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        _fail(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output)
    for mod in ds.modalities:
        run = out.uns["bloom"]["modalities"][mod]["run"]
        key = f"X_bloom_{mod}"
        typer.echo(
            f"wrote {output}: {out.n_obs} cells x {out.obsm[key].shape[1]} (obsm['{key}']); "
            f"{run['n_views']} views in {run['wall_s']} s ({run['s_per_cell']} s/cell); "
            f"{run['n_empty_cells']} empty cell(s)",
            err=True,
        )
    for mod, a in extra.items():
        path = output.with_name(f"{output.stem}_{mod}_only.h5ad")
        a.write_h5ad(path)
        typer.echo(f"wrote {path}: {a.n_obs} {mod}-only cell(s) (obsm['X_bloom_{mod}'])", err=True)


if __name__ == "__main__":
    app()
