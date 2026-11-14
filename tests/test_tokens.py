"""tokens.py: CellFeatures construction, TokenSet validation, padding/collate (test-plan S8)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bloom.tokens import (
    GENE,
    PAD_EMB_IDX,
    PAD_KIND,
    PEAK,
    CellFeatures,
    TokenSet,
    collate,
    detected_from_csr_row,
    key_padding_mask,
)

pytestmark = pytest.mark.unit


class TestCellFeatures:
    def test_from_csr_row_drops_unmapped_and_zero_and_ors_duplicates(self):
        # columns 0..5; col 2 unmapped; cols 0 and 4 both map to emb 5 (duplicate symbol, R3)
        col_to_emb = np.array([5, 7, -1, 2, 5, 9])
        col_idx = np.array([0, 1, 2, 3, 4, 5])
        values = np.array([1.0, 0.0, 3.0, 2.0, 4.0, 0.5], dtype=np.float32)
        emb, vals = detected_from_csr_row(col_idx, values, col_to_emb)
        assert emb.dtype == np.int32 and vals.dtype == np.float32
        assert emb.tolist() == [2, 5, 9]  # sorted; 7 dropped (value 0), -1 dropped
        assert vals.tolist() == [2.0, 4.0, 0.5]  # duplicate emb 5 keeps the max value
        cell = CellFeatures.from_csr_row(3, col_idx, values, col_to_emb, "gene")
        assert cell.modality == "gene" and cell.n_detected == 3 and cell.cell_index == 3

    def test_peaks_row_ignores_values(self):
        emb, vals = detected_from_csr_row(np.array([2, 0, 1]), None, np.array([10, -1, 4]))
        assert emb.tolist() == [4, 10] and vals is None
        cell = CellFeatures.from_csr_row(
            0, np.array([2, 0, 1]), None, np.array([10, -1, 4]), "peak"
        )
        assert cell.modality == "peak" and cell.detected_peak_emb_idx.tolist() == [4, 10]

    def test_validation(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            CellFeatures.genes(0, [3, 1], [1.0, 1.0])
        with pytest.raises(ValueError, match="negative"):
            CellFeatures.genes(0, [-1, 1], [1.0, 1.0])
        with pytest.raises(ValueError, match="align"):
            CellFeatures.genes(0, [1, 2], [1.0])
        with pytest.raises(ValueError, match="one modality"):
            CellFeatures(0, np.array([1]), np.array([1.0]), np.array([2]))
        empty = CellFeatures.genes(0, [], [])
        assert empty.modality is None and empty.n_detected == 0

    def test_restrict(self):
        cell = CellFeatures.genes(0, [1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0])
        r = cell.restrict(np.array([4, 2, 99]))
        assert r.detected_gene_emb_idx.tolist() == [2, 4] and r.gene_values.tolist() == [2.0, 4.0]
        assert cell.restrict(None) is cell


class TestTokenSet:
    def test_constructors_and_validation(self):
        g = TokenSet.genes([5, 1])
        assert len(g) == 2 and g.kind.tolist() == [GENE, GENE] and g.emb_idx.dtype == np.int32
        p = TokenSet.peaks(np.array([7]))
        assert p.kind.tolist() == [PEAK] and p.kind.dtype == np.int8
        with pytest.raises(ValueError, match="-1"):
            TokenSet(np.array([-1]), np.array([GENE]))
        with pytest.raises(ValueError, match="kind"):
            TokenSet(np.array([1]), np.array([3]))
        with pytest.raises(
            ValueError, match="one modality"
        ):  # S11 (#46): no gene + peak mix in a view
            TokenSet(np.array([1, 2]), np.array([GENE, PEAK]))
        with pytest.raises(ValueError, match="same length"):
            TokenSet(np.array([1, 2]), np.array([GENE]))


class TestCollate:
    """S8: lengths correct; pads have emb_idx == -1 and kind == 0; mask True exactly at pads."""

    def test_padding_and_mask(self):
        views = [TokenSet.genes([3, 1, 2]), TokenSet.peaks([9]), TokenSet.genes([])]
        batch = collate(views, lmax=5, cell_index=[10, 11, 12], view_index=[0, 0, 1])
        assert batch.emb_idx.shape == (3, 5) and batch.emb_idx.dtype == torch.int32
        assert batch.kind.shape == (3, 5) and batch.kind.dtype == torch.int8
        assert batch.lengths.tolist() == [3, 1, 0] and batch.lengths.dtype == torch.int64
        assert batch.cell_index.tolist() == [10, 11, 12]
        assert batch.view_index.tolist() == [0, 0, 1]
        assert batch.emb_idx[0].tolist() == [3, 1, 2, PAD_EMB_IDX, PAD_EMB_IDX]
        assert batch.kind[0].tolist() == [GENE, GENE, GENE, PAD_KIND, PAD_KIND]
        assert batch.kind[1].tolist() == [PEAK, 0, 0, 0, 0]
        mask = key_padding_mask(batch)
        assert mask.dtype == torch.bool and mask.shape == (3, 5)
        expected_pad = torch.arange(5)[None, :] >= batch.lengths[:, None]
        assert torch.equal(mask, expected_pad)
        assert torch.equal(mask, batch.key_padding_mask())
        # pads are exactly where emb_idx == -1 and kind == 0
        assert torch.equal(mask, batch.emb_idx == PAD_EMB_IDX)
        assert torch.equal(mask, batch.kind == PAD_KIND)
        assert len(batch) == 3 and batch.lmax == 5

    def test_lmax_defaults_and_errors(self):
        batch = collate([TokenSet.genes([1, 2]), TokenSet.genes([4])])
        assert batch.lmax == 2 and batch.cell_index.tolist() == [0, 1]
        assert collate([]).lmax == 1 and len(collate([])) == 0
        with pytest.raises(ValueError, match="> lmax"):
            collate([TokenSet.genes([1, 2, 3])], lmax=2)
        with pytest.raises(ValueError, match="one entry"):
            collate([TokenSet.genes([1])], lmax=2, cell_index=[0, 1])

    def test_to_device_keeps_dtypes(self):
        batch = collate([TokenSet.genes([1])], lmax=2).to("cpu")
        assert batch.emb_idx.dtype == torch.int32 and batch.kind.dtype == torch.int8
