# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""block_m selection for the untuned CK 2-stage MoE path."""

import itertools

import pytest

import aiter.fused_moe as fused_moe_module
import aiter.fused_moe_dp_shared_expert as dp_shared_expert_module
from aiter.fused_moe import get_block_size_M

CU_NUM = 256
SUPPORTED = (32, 64, 128)


def _round_count_block_m(token, topk, expert, inter_dim, cu_num=CU_NUM):
    tgN = (inter_dim + 127) // 128
    tmp = []
    for el in SUPPORTED:
        max_num_tokens = token * topk + expert * el - topk
        tg_num = tgN * (max_num_tokens + el - 1) // el
        rnd = (tg_num + cu_num - 1) // cu_num
        empty = cu_num - tg_num % cu_num
        tmp.append((rnd, empty, el))
    return min(tmp, key=lambda x: x[:2])[-1]


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: CU_NUM)
    get_block_size_M.cache_clear()
    yield
    get_block_size_M.cache_clear()


@pytest.fixture
def gfx942(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx942")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: CU_NUM)
    get_block_size_M.cache_clear()
    yield
    get_block_size_M.cache_clear()


@pytest.mark.parametrize(
    "inter_dim, expert, topk, token, expected",
    [
        (512, 256, 8, 256, 32),
        (512, 256, 8, 512, 32),
        (512, 256, 8, 1024, 64),
        (512, 256, 8, 1536, 64),
        (512, 256, 8, 2048, 128),
        (512, 256, 8, 4096, 128),
        (512, 256, 6, 768, 32),
        (512, 256, 6, 2048, 64),
        (14336, 8, 2, 128, 64),
        (14336, 8, 2, 256, 128),
    ],
)
def test_measured_shapes(gfx950, inter_dim, expert, topk, token, expected):
    assert get_block_size_M(token, topk, expert, inter_dim) == expected


def test_cap_only_narrows(gfx950):
    shapes = itertools.product(
        (256, 512, 1024, 2048, 4096, 7168),
        (8, 32, 128, 256),
        (1, 2, 4, 6, 8),
        (1, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384),
    )
    for inter_dim, expert, topk, token in shapes:
        got = get_block_size_M(token, topk, expert, inter_dim)
        assert got in SUPPORTED
        assert got <= _round_count_block_m(token, topk, expert, inter_dim)


def test_dense_routing_is_unchanged(gfx950):
    for inter_dim in (256, 1024, 4096, 7168):
        for expert, topk in ((8, 2), (128, 8), (256, 8)):
            token = 1024 * expert // topk
            assert get_block_size_M(token, topk, expert, inter_dim) == (
                _round_count_block_m(token, topk, expert, inter_dim)
            )


def test_other_arch_keeps_round_count_choice(gfx942):
    for inter_dim, expert, topk, token in (
        (512, 256, 8, 512),
        (512, 256, 8, 1024),
        (14336, 8, 2, 128),
    ):
        assert get_block_size_M(token, topk, expert, inter_dim) == (
            _round_count_block_m(token, topk, expert, inter_dim)
        )


def test_dp_shared_expert_uses_the_same_helper(gfx950):
    assert dp_shared_expert_module.get_block_size_M is get_block_size_M
    assert dp_shared_expert_module.get_block_size_M(512, 8, 256, 512) == 32
