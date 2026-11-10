# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuned fused-MoE tier fallback at the 32768 and 131072 buckets."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs, get_padded_M

CONFIGS = Path(__file__).resolve().parents[1] / "aiter/configs/model_configs"
FLASH_CSV = CONFIGS / "a8w8_blockscale_tuned_fmoe_glm5_3_flash_gfx950.csv"
SHAPE = (4096, 1024, 288, 8)
FP8_BLOCKSCALE = (
    dtypes.bf16,
    dtypes.fp8,
    dtypes.fp8,
    QuantType.per_1x128,
    True,
    ActivationType.Silu,
)
KIMI_CSV = CONFIGS / "kimik3_a16w4_tuned_fmoe.csv"
KIMI_SHAPE = (3584, 384, 896, 16)
A16W4_SITUV2 = (
    dtypes.bf16,
    dtypes.bf16,
    dtypes.fp4x2,
    QuantType.per_1x32,
    True,
    ActivationType.Situv2,
)


@pytest.fixture
def tuned(monkeypatch):
    fused_moe = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG", raising=False)

    def use(config):
        monkeypatch.setenv("AITER_CONFIG_FMOE", str(config))
        monkeypatch.setattr(fused_moe, "cfg_2stages", None)
        fused_moe.AITER_CONFIGS.get_config_file.cache_clear()
        get_2stage_cfgs.cache_clear()

    yield use
    get_2stage_cfgs.cache_clear()
    fused_moe.AITER_CONFIGS.get_config_file.cache_clear()


def _rows(tokens, csv=FLASH_CSV, shape=SHAPE):
    df = pd.read_csv(csv)
    model_dim, inter_dim, expert, topk = shape
    df = df[
        (df.gfx == "gfx950")
        & (df.model_dim == model_dim)
        & (df.inter_dim == inter_dim)
        & (df.expert == expert)
        & (df.topk == topk)
    ]
    return df[df.token.isin(tokens)]


def _write(tmp_path, df):
    path = tmp_path / "tuned_fmoe.csv"
    df.to_csv(path, index=False)
    return path


def _selection(tier, shape=SHAPE, quant=FP8_BLOCKSCALE):
    meta = get_2stage_cfgs(tier, *shape, *quant, False, 0, 0, is_shuffled=True)
    names = tuple(
        stage.keywords.get("kernelName") if stage is not None else None
        for stage in (meta.stage1, meta.stage2)
    )
    return meta.run_1stage, meta.block_m, names


def _default(monkeypatch, tier, *case):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    try:
        return _selection(tier, *case)
    finally:
        monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG")
        get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize(
    ("padded_m", "expected"),
    (
        (1, []),
        (8192, []),
        (16384, []),
        (32768, [16384]),
        (131072, [32768, 16384]),
    ),
)
def test_fallback_padded_ms(padded_m, expected):
    fused_moe = importlib.import_module("aiter.fused_moe")
    assert fused_moe.get_fallback_padded_Ms(padded_m) == expected


@pytest.mark.parametrize("m", (16385, 32768, 50000, 131071, 131072, 200000))
def test_large_m_reuses_16384_row(tuned, monkeypatch, m):
    tuned(FLASH_CSV)
    tier = get_padded_M(m)
    row = _rows([16384]).iloc[0]
    run_1stage, block_m, names = _selection(tier)
    assert not run_1stage
    assert block_m == row.block_m
    assert names == (row.kernelName1, row.kernelName2)
    assert _selection(tier) != _default(monkeypatch, tier)


def test_131072_prefers_32768_row(tuned, tmp_path):
    near = _rows([2048]).assign(token=32768)
    tuned(_write(tmp_path, pd.concat([_rows([16384]), near])))
    run_1stage, block_m, _ = _selection(131072)
    assert run_1stage
    assert block_m == near.iloc[0].block_m
    assert _selection(131072) == _selection(32768)


@pytest.mark.parametrize("tier", (32768, 131072))
def test_no_reuse_of_rows_below_16384(tuned, monkeypatch, tmp_path, tier):
    tuned(_write(tmp_path, _rows([2, 4, 8, 16, 64, 512, 2048, 4096, 8192])))
    assert _selection(tier) == _default(monkeypatch, tier)


@pytest.mark.parametrize("tier", (32768, 131072))
def test_no_reuse_of_route_reduce_rows(tuned, monkeypatch, tier):
    case = (KIMI_SHAPE, A16W4_SITUV2)
    tuned(KIMI_CSV)
    stage2 = _rows([16384], KIMI_CSV, KIMI_SHAPE).iloc[0].kernelName2
    assert stage2.endswith("_reduce")
    assert _selection(16384, *case)[2][1] == stage2
    assert _selection(tier, *case) == _default(monkeypatch, tier, *case)


@pytest.mark.parametrize("tier", (64, 8192, 16384))
def test_no_fallback_below_32768(tuned, monkeypatch, tmp_path, tier):
    tuned(_write(tmp_path, _rows([16, 4096])))
    assert _selection(tier) == _default(monkeypatch, tier)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
