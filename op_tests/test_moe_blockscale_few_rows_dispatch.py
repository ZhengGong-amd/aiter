# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Default dispatch of gfx950 fp8 block-scale MoE by rows per expert."""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType
from aiter import dtypes as aiter_dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

_GFX950_ONLY = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="Exercises the gfx950 fp8 asm dispatch"
)


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _dispatch(token, model_dim, inter_dim, expert, topk):
    return get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        torch.bfloat16,
        aiter_dtypes.fp8,
        aiter_dtypes.fp8,
        QuantType.per_1x128,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
    )


_CASES = [
    (32, 3072, 1024, 256, 8, False, 16),
    (64, 3072, 1024, 256, 8, False, 16),
    (128, 3072, 1024, 256, 8, False, 16),
    (256, 3072, 1024, 256, 8, False, 16),
    (512, 3072, 1024, 256, 8, True, 32),
    (4096, 3072, 1024, 256, 8, True, 32),
    (512, 7168, 2048, 256, 4, False, 16),
    (1024, 7168, 2048, 256, 4, True, 32),
    (256, 2560, 640, 512, 10, False, 16),
    (512, 2560, 640, 512, 10, True, 32),
]


@_GFX950_ONLY
@pytest.mark.parametrize(
    "token,model_dim,inter_dim,expert,topk,run_1stage,block_m", _CASES
)
def test_gfx950_blockscale_default_dispatch(
    token, model_dim, inter_dim, expert, topk, run_1stage, block_m
):
    metadata = _dispatch(token, model_dim, inter_dim, expert, topk)
    assert metadata.run_1stage == run_1stage
    assert metadata.block_m == block_m


@_GFX950_ONLY
def test_blockscale_default_dispatch_other_arch_keeps_asm(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx942")
    metadata = _dispatch(64, 3072, 1024, 256, 8)
    assert metadata.run_1stage
    assert metadata.block_m == 32


def _quant_128x128(weight):
    experts, dim1, dim2 = weight.shape
    blocks = weight.view(experts, dim1 // 128, 128, dim2 // 128, 128)
    blocks = blocks.permute(0, 1, 3, 2, 4).contiguous().view(experts, -1, 128 * 128)
    qt, scale = aiter.pertoken_quant(blocks, quant_dtype=aiter_dtypes.fp8)
    qt = qt.view(experts, dim1 // 128, dim2 // 128, 128, 128)
    qt = qt.permute(0, 1, 3, 2, 4).contiguous().view(experts, dim1, dim2)
    return qt, scale.view(experts, dim1 // 128, dim2 // 128)


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


@_GFX950_ONLY
@pytest.mark.parametrize("token,run_1stage", [(64, False), (256, False), (512, True)])
def test_gfx950_blockscale_few_rows_matches_reference(token, run_1stage):
    model_dim, inter_dim, expert, topk = 1024, 512, 64, 2
    metadata = _dispatch(token, model_dim, inter_dim, expert, topk)
    assert metadata.run_1stage == run_1stage
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(token, model_dim, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(expert, 2 * inter_dim, model_dim, device=dev) / 10
    w2 = torch.randn(expert, model_dim, inter_dim, device=dev) / 10
    w1q, w1s = _quant_128x128(w1.to(torch.bfloat16))
    w2q, w2s = _quant_128x128(w2.to(torch.bfloat16))
    score = torch.randn(token, expert, device=dev, dtype=torch.bfloat16)
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    out = fused_moe(
        x,
        shuffle_weight(w1q, layout=(16, 16)),
        shuffle_weight(w2q, layout=(16, 16)),
        topk_weights,
        topk_ids,
        w1_scale=fp4_utils.e8m0_shuffle(w1s),
        w2_scale=fp4_utils.e8m0_shuffle(w2s),
        quant_type=QuantType.per_128x128,
        activation=ActivationType.Silu,
    )

    a1, a1_scale = aiter.pertoken_quant(
        x.view(token, -1, 128), quant_dtype=aiter_dtypes.fp8
    )
    out1 = torch_moe_stage1(
        a1.view(token, model_dim),
        w1q,
        w2q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_128x128,
        a1_scale=a1_scale.squeeze(-1),
        w1_scale=w1s,
    )
    a2, a2_scale = aiter.pertoken_quant(
        out1.view(token, -1, 128), quant_dtype=aiter_dtypes.fp8
    )
    ref = torch_moe_stage2(
        a2.view(token, topk, -1),
        w1q,
        w2q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_128x128,
        w2_scale=w2s,
        a2_scale=a2_scale.view(token, topk, -1),
    )
    assert torch.isfinite(out).all()
    assert _cos_diff(out, ref) < 1e-3
