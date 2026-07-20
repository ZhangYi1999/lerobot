# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the DAgger (Dataset Aggregation) training pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.dagger import BETA_SCHEDULES, DAggerConfig  # noqa: E402
from lerobot.scripts.lerobot_dagger import (  # noqa: E402
    BetaScheduler,
    _extract_successes,
    _flatten_envs,
    append_episodes,
)


# ---------------------------------------------------------------------------
# BetaScheduler
# ---------------------------------------------------------------------------
def test_beta_scheduler_exponential():
    s = BetaScheduler(init_beta=1.0, schedule="exponential", decay=0.9, min_beta=0.05)
    assert s.value(0) == pytest.approx(1.0)
    assert s.value(1) == pytest.approx(0.9)
    assert s.value(2) == pytest.approx(0.81)


def test_beta_scheduler_linear():
    s = BetaScheduler(init_beta=1.0, schedule="linear", decay=0.8, min_beta=0.1)
    # linear: init - (1 - decay) * iter = 1 - 0.2 * iter
    assert s.value(0) == pytest.approx(1.0)
    assert s.value(1) == pytest.approx(0.8)
    assert s.value(2) == pytest.approx(0.6)


def test_beta_scheduler_constant():
    s = BetaScheduler(init_beta=0.7, schedule="constant")
    assert all(s.value(i) == pytest.approx(0.7) for i in range(10))


def test_beta_scheduler_respects_min_and_max():
    s = BetaScheduler(init_beta=1.0, schedule="exponential", decay=0.5, min_beta=0.05)
    values = [s.value(i) for i in range(50)]
    assert all(0.05 <= v <= 1.0 for v in values)
    # Deep into decay, beta pins to the floor.
    assert s.value(40) == pytest.approx(0.05)


@pytest.mark.parametrize("schedule", BETA_SCHEDULES)
def test_beta_scheduler_all_schedules_run(schedule):
    s = BetaScheduler(init_beta=1.0, schedule=schedule, decay=0.9, min_beta=0.05)
    assert 0.05 <= s.value(5) <= 1.0


# ---------------------------------------------------------------------------
# DAggerConfig validation
# ---------------------------------------------------------------------------
def test_dagger_config_defaults_auto_rollout_batch_size():
    cfg = DAggerConfig(episodes_per_iteration=8, rollout_batch_size=0)
    assert cfg.rollout_batch_size == 8


def test_dagger_config_caps_rollout_batch_size():
    cfg = DAggerConfig(episodes_per_iteration=4, rollout_batch_size=16)
    assert cfg.rollout_batch_size == 4


def test_dagger_config_rejects_bad_schedule():
    with pytest.raises(ValueError, match="beta_schedule"):
        DAggerConfig(beta_schedule="bogus")


def test_dagger_config_rejects_beta_min_above_init():
    with pytest.raises(ValueError, match="beta_min"):
        DAggerConfig(init_beta=0.3, beta_min=0.5)


def test_dagger_config_rejects_bad_decay():
    with pytest.raises(ValueError, match="beta_decay"):
        DAggerConfig(beta_decay=1.5)


# ---------------------------------------------------------------------------
# _extract_successes
# ---------------------------------------------------------------------------
def test_extract_successes_from_dict_final_info():
    info = {"final_info": {"is_success": np.array([True, False])}}
    assert _extract_successes(info, 2) == [True, False]


def test_extract_successes_from_sequence_final_info():
    info = {"final_info": [{"is_success": True}, None]}
    assert _extract_successes(info, 2) == [True, False]


def test_extract_successes_from_top_level():
    info = {"is_success": np.array([False, True, True])}
    assert _extract_successes(info, 3) == [False, True, True]


def test_extract_successes_missing_defaults_false():
    assert _extract_successes({}, 3) == [False, False, False]


# ---------------------------------------------------------------------------
# _flatten_envs
# ---------------------------------------------------------------------------
def test_flatten_envs():
    envs = {"suite_a": {0: "env0", 1: "env1"}, "suite_b": {0: "env2"}}
    assert _flatten_envs(envs) == ["env0", "env1", "env2"]


# ---------------------------------------------------------------------------
# append_episodes
# ---------------------------------------------------------------------------
def _make_episode(n_frames: int, success: bool) -> dict:
    return {"frames": [{"i": i} for i in range(n_frames)], "success": success}


def test_append_episodes_writes_all_by_default():
    dataset = MagicMock()
    episodes = [_make_episode(3, True), _make_episode(2, False)]
    n_eps, n_frames = append_episodes(dataset, episodes, only_success=False)
    assert (n_eps, n_frames) == (2, 5)
    assert dataset.add_frame.call_count == 5
    assert dataset.save_episode.call_count == 2


def test_append_episodes_only_success_filters():
    dataset = MagicMock()
    episodes = [_make_episode(3, True), _make_episode(2, False)]
    n_eps, n_frames = append_episodes(dataset, episodes, only_success=True)
    assert (n_eps, n_frames) == (1, 3)
    assert dataset.add_frame.call_count == 3
    assert dataset.save_episode.call_count == 1


def test_append_episodes_skips_empty():
    dataset = MagicMock()
    episodes = [_make_episode(0, True), _make_episode(2, True)]
    n_eps, n_frames = append_episodes(dataset, episodes, only_success=False)
    assert (n_eps, n_frames) == (1, 2)
    assert dataset.save_episode.call_count == 1
