"""CLI smoke tests (E6: `bloom info` exits 0 without assets; `bloom embed` is covered in test_embed.py)."""

import pytest
from typer.testing import CliRunner

from bloom import __version__
from bloom.cli import NOT_IMPLEMENTED_EXIT_CODE, app

runner = CliRunner()


@pytest.mark.unit
def test_info_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOM_HOME", str(tmp_path))
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert "torch" in result.output
    assert str(tmp_path) in result.output
    assert "missing" in result.output  # asset groups listed (details in test_assets.py)


@pytest.mark.unit
def test_atac_missing_file_is_a_user_error():
    """`--atac` is accepted (Phase 2 landed); a missing file is exit 1 with the path named."""
    result = runner.invoke(app, ["embed", "--atac", "x.h5ad", "--species", "mouse", "-o", "o.h5ad"])
    assert result.exit_code == 1
    assert "x.h5ad does not exist" in result.output
    assert NOT_IMPLEMENTED_EXIT_CODE == 2
