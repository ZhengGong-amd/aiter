# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reuse of a tuned FlyDSL 2-stage fused-MoE row from another inter_dim shard."""

import importlib
import re
from pathlib import Path

import pandas as pd
import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import cktile_moe_stage1, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

_SKIP = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="MXFP4 FlyDSL MoE dispatch needs gfx950"
)

QWEN_CSV = (
    Path(__file__).resolve().parents[1]
    / "aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv"
)
MODEL_DIM, TUNED_INTER, EXPERT, TOPK = 4096, 512, 512, 10
MINIMAX_CSV = QWEN_CSV.with_name("minimax_m3_a16w4_tuned_fmoe.csv")
TILE_RE = re.compile(r"_t(\d+)x(\d+)x(\d+)")


@pytest.fixture
def table(monkeypatch, tmp_path):
    fused_moe = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG", raising=False)

    def use(df):
        path = tmp_path / "tuned_fmoe.csv"
        df.to_csv(path, index=False)
        monkeypatch.setenv("AITER_CONFIG_FMOE", str(path))
        monkeypatch.setattr(fused_moe, "cfg_2stages", None)
        fused_moe.AITER_CONFIGS.get_config_file.cache_clear()
        get_2stage_cfgs.cache_clear()

    yield use
    get_2stage_cfgs.cache_clear()
    fused_moe.AITER_CONFIGS.get_config_file.cache_clear()


def _qwen_rows(tokens=(1, 16, 32)):
    df = pd.read_csv(QWEN_CSV)
    df = df[
        (df.model_dim == MODEL_DIM)
        & (df.inter_dim == TUNED_INTER)
        & (df.expert == EXPERT)
        & (df.topk == TOPK)
        & df.token.isin(tokens)
    ]
    assert len(df) == len(tokens)
    return df.reset_index(drop=True)


def _selection(token, inter_dim, model_dim=MODEL_DIM, expert=EXPERT, topk=TOPK):
    get_2stage_cfgs.cache_clear()
    meta = get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
    )
    names = tuple(
        stage.keywords.get("kernelName") or stage.keywords.get("kernelName2")
        for stage in (meta.stage1, meta.stage2)
    )
    return meta.block_m, names


def _heuristic(monkeypatch, *args, **kwargs):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    try:
        return _selection(*args, **kwargs)
    finally:
        monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG")


def _row(df, token):
    row = df[df.token == token].iloc[0]
    return int(row.block_m), (row.kernelName1, row.kernelName2)


def _without_xcd(names):
    return tuple(re.sub(r"_xcd\d+", "", n) for n in names)


@_SKIP
@pytest.mark.parametrize("token", (1, 16, 32))
def test_other_shard_row_is_reused_without_xcd(table, monkeypatch, token):
    df = _qwen_rows()
    table(df)
    block_m, names = _row(df, token)
    got = _selection(token, 1024)
    assert got == (block_m, _without_xcd(names))
    assert got != _heuristic(monkeypatch, token, 1024)


@_SKIP
@pytest.mark.parametrize("token", (1, 32))
def test_xcd_hint_is_dropped(table, token):
    df = _qwen_rows()
    assert "_xcd" in df[df.token == token].iloc[0].kernelName1
    table(df)
    assert not any("_xcd" in n for n in _selection(token, 1024)[1])


@_SKIP
def test_exact_row_is_unchanged(table):
    df = _qwen_rows()
    table(df)
    assert _selection(32, TUNED_INTER) == _row(df, 32)


@_SKIP
def test_other_model_dim_is_not_reused(table, monkeypatch):
    table(_qwen_rows())
    got = _selection(16, 1024, model_dim=6144)
    assert got == _heuristic(monkeypatch, 16, 1024, model_dim=6144)


@_SKIP
@pytest.mark.parametrize("expert,topk", ((256, 10), (512, 8)))
def test_other_expert_or_topk_is_not_reused(table, monkeypatch, expert, topk):
    table(_qwen_rows())
    got = _selection(16, 1024, expert=expert, topk=topk)
    assert got == _heuristic(monkeypatch, 16, 1024, expert=expert, topk=topk)


@_SKIP
def test_other_token_bucket_is_not_reused(table, monkeypatch):
    table(_qwen_rows(tokens=(16,)))
    assert _selection(64, 1024) == _heuristic(monkeypatch, 64, 1024)


@_SKIP
def test_non_flydsl_rows_are_not_reused(table, monkeypatch):
    df = _qwen_rows(tokens=(16,))
    df.loc[0, "kernelName2"] = (
        "moe_ck2stages_gemm2_256x32x128x256_1x4_MulABScaleExpertWeightShuffled"
        "_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
    )
    table(df)
    assert _selection(16, 1024) == _heuristic(monkeypatch, 16, 1024)


@_SKIP
def test_bypass_tune_config_disables_reuse(table, monkeypatch):
    df = _qwen_rows()
    table(df)
    reused = _without_xcd(_row(df, 16)[1])
    assert _selection(16, 1024)[1] == reused
    assert _heuristic(monkeypatch, 16, 1024)[1] != reused


@_SKIP
@pytest.mark.parametrize("gate_up_interleave", (True, False))
def test_untuned_backend_is_kept(table, monkeypatch, gate_up_interleave):
    df = pd.read_csv(MINIMAX_CSV)
    df = df[
        (df.model_dim == 6144)
        & (df.inter_dim == 768)
        & (df.expert == 129)
        & (df.topk == 5)
        & (df.token == 128)
        & (df.q_dtype_a == str(dtypes.bf16))
    ]
    assert len(df) == 1 and df.iloc[0].kernelName1.startswith("flydsl_moe1_")
    table(df)

    def stage1():
        get_2stage_cfgs.cache_clear()
        meta = get_2stage_cfgs(
            128,
            6144,
            1536,
            129,
            5,
            dtypes.bf16,
            dtypes.bf16,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Swiglu,
            False,
            0,
            0,
            True,
            GateMode.INTERLEAVE.value,
            gate_up_interleave=gate_up_interleave,
        )
        return getattr(meta.stage1, "func", meta.stage1)

    assert stage1() is cktile_moe_stage1
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    assert stage1() is cktile_moe_stage1


@_SKIP
@pytest.mark.parametrize("token", (4096, 32768))
def test_larger_token_buckets_keep_the_heuristic(table, monkeypatch, token):
    table(_qwen_rows(tokens=(token,)))
    assert _selection(token, 1024) == _heuristic(monkeypatch, token, 1024)


@_SKIP
def test_inter_dim_tiles_are_rederived(table):
    df = _qwen_rows(tokens=(16,))
    (tm1, tn1, tk1), (tm2, tn2, tk2) = (
        _tiles(df.loc[0, c]) for c in ("kernelName1", "kernelName2")
    )
    assert tk2 == 256 and 640 % tn1 == 0
    table(df)
    _, names = _selection(16, 640)
    (m1, n1, k1), (m2, n2, k2) = (_tiles(n) for n in names)
    assert (m1, n1, k1) == (tm1, tn1, tk1)
    assert (m2, n2, k2) == (tm2, tn2, 128)
    for got, tuned in zip(names, (df.loc[0, "kernelName1"], df.loc[0, "kernelName2"])):
        assert TILE_RE.sub("", got) == TILE_RE.sub("", re.sub(r"_xcd\d+", "", tuned))


def _tiles(name):
    return tuple(int(v) for v in TILE_RE.search(name).groups())
