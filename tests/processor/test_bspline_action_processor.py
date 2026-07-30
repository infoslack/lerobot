#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import BsplineActionResamplerStep
from lerobot.processor.converters import create_transition
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.types import TransitionKey


def test_output_length_preserves_chunk_duration():
    """A 30 Hz chunk resampled to 100 Hz keeps its duration: round(T * 100 / 30) actions."""
    step = BsplineActionResamplerStep(source_fps=30.0, target_fps=100.0)
    chunk = torch.randn(50, 6)

    resampled = step.action(chunk)

    assert resampled.shape == (round(50 * 100 / 30), 6)


def test_endpoints_are_preserved():
    """The clamped spline reproduces the first and last actions of the chunk exactly."""
    step = BsplineActionResamplerStep(source_fps=30.0, target_fps=100.0)
    chunk = torch.randn(20, 4)

    resampled = step.action(chunk)

    torch.testing.assert_close(resampled[0], chunk[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(resampled[-1], chunk[-1], atol=1e-5, rtol=0)


def test_linear_chunk_is_reproduced_exactly():
    """A straight-line chunk resamples to the same straight line at the new rate."""
    step = BsplineActionResamplerStep(source_fps=10.0, target_fps=40.0)
    chunk_size = 10
    start = torch.tensor([0.0, -1.0, 2.0])
    end = torch.tensor([1.0, 1.0, -2.0])
    times = torch.linspace(0.0, 1.0, chunk_size).unsqueeze(1)
    chunk = start + times * (end - start)

    resampled = step.action(chunk)

    new_times = torch.linspace(0.0, 1.0, resampled.shape[0]).unsqueeze(1)
    expected = start + new_times * (end - start)
    torch.testing.assert_close(resampled, expected, atol=1e-4, rtol=0)


def test_cubic_curve_matches_closed_form():
    """A cubic polynomial lies in the span of a cubic spline, so the fit reproduces it."""
    step = BsplineActionResamplerStep(source_fps=25.0, target_fps=100.0, degree=3)
    chunk_size = 40
    times = torch.linspace(0.0, 1.0, chunk_size)
    chunk = (times**3 - 0.5 * times).unsqueeze(1)

    resampled = step.action(chunk)

    new_times = torch.linspace(0.0, 1.0, resampled.shape[0])
    expected = (new_times**3 - 0.5 * new_times).unsqueeze(1)
    torch.testing.assert_close(resampled, expected, atol=1e-4, rtol=0)


def test_batched_and_unbatched_agree():
    """A batched chunk resamples each batch element like the unbatched call."""

    def fresh_step():
        return BsplineActionResamplerStep(source_fps=30.0, target_fps=90.0)

    chunk = torch.randn(2, 15, 5)

    batched = fresh_step().action(chunk)
    per_item = torch.stack([fresh_step().action(chunk[0]), fresh_step().action(chunk[1])])

    assert batched.ndim == 3
    torch.testing.assert_close(batched, per_item)


def test_downsampling_works():
    """target_fps below source_fps produces a shorter chunk over the same duration."""
    step = BsplineActionResamplerStep(source_fps=100.0, target_fps=30.0)
    chunk = torch.randn(100, 6)

    resampled = step.action(chunk)

    assert resampled.shape == (30, 6)


def test_fewer_control_points_smooth_the_chunk():
    """Fitting with fewer control points low-pass filters a noisy chunk."""
    times = torch.linspace(0.0, 1.0, 60)
    clean = torch.sin(2 * torch.pi * times)
    noisy = (clean + 0.05 * torch.randn_like(clean)).unsqueeze(1)

    faithful = BsplineActionResamplerStep(source_fps=30.0, target_fps=30.0).action(noisy)
    smoothed = BsplineActionResamplerStep(source_fps=30.0, target_fps=30.0, num_control_points=10).action(
        noisy
    )

    def roughness(chunk: torch.Tensor) -> torch.Tensor:
        return torch.diff(chunk, n=2, dim=0).square().sum()

    assert roughness(smoothed) < roughness(faithful)


def test_dtype_and_device_are_preserved():
    step = BsplineActionResamplerStep(source_fps=30.0, target_fps=60.0, align_chunks=False)

    for dtype in (torch.float32, torch.float64, torch.float16):
        chunk = torch.randn(12, 3, dtype=dtype)
        resampled = step.action(chunk)
        assert resampled.dtype == dtype
        assert resampled.device == chunk.device


def test_short_chunks_clamp_degree_and_still_resample():
    """A 2-action chunk cannot support a cubic, so the fit degrades to linear instead of failing."""
    step = BsplineActionResamplerStep(source_fps=10.0, target_fps=50.0, degree=3)
    chunk = torch.tensor([[0.0, 0.0], [1.0, -1.0]])

    resampled = step.action(chunk)

    new_times = torch.linspace(0.0, 1.0, resampled.shape[0]).unsqueeze(1)
    expected = new_times * torch.tensor([1.0, -1.0])
    torch.testing.assert_close(resampled, expected, atol=1e-5, rtol=0)


def test_invalid_inputs_raise():
    step = BsplineActionResamplerStep()

    with pytest.raises(ValueError, match="tensor action"):
        step.action({"joint": 1.0})
    with pytest.raises(ValueError, match="chunk"):
        step.action(torch.randn(6))
    with pytest.raises(ValueError, match="at least 2 actions"):
        step.action(torch.randn(1, 6))


def test_invalid_config_raises():
    with pytest.raises(ValueError, match="must be positive"):
        BsplineActionResamplerStep(source_fps=0.0)
    with pytest.raises(ValueError, match="degree"):
        BsplineActionResamplerStep(degree=0)
    with pytest.raises(ValueError, match="num_control_points"):
        BsplineActionResamplerStep(num_control_points=1)


def test_step_processes_transitions():
    """The step plugs into the transition flow like any other ActionProcessorStep."""
    step = BsplineActionResamplerStep(source_fps=30.0, target_fps=60.0)
    chunk = torch.randn(10, 4)
    transition = create_transition(observation={}, action=chunk)

    result = step(transition)

    assert result[TransitionKey.ACTION].shape == (20, 4)


def test_registry_and_config_roundtrip():
    """The step is registered and can be rebuilt from its own config."""
    step_cls = ProcessorStepRegistry.get("bspline_action_resampler")
    assert step_cls is BsplineActionResamplerStep

    step = BsplineActionResamplerStep(source_fps=25.0, target_fps=75.0, degree=2, num_control_points=8)
    rebuilt = step_cls(**step.get_config())

    assert rebuilt == step


def test_transform_features_is_identity():
    step = BsplineActionResamplerStep()
    features = {PipelineFeatureType.ACTION: {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}}

    assert step.transform_features(features) == features


def test_consecutive_chunks_join_with_continuous_position_and_velocity():
    """With alignment on, chunk N+1 starts exactly where chunk N ended, with the same slope.

    The slopes are measured by finite differences, so the curves are sampled densely
    (high target_fps) to approximate the analytic boundary derivatives well.
    """
    step = BsplineActionResamplerStep(source_fps=20.0, target_fps=2000.0)
    times = torch.linspace(0.0, 1.0, 40)
    trajectory = torch.stack([torch.sin(2 * torch.pi * times), torch.cos(torch.pi * times)], dim=1)

    first = step.action(trajectory[:20])
    second = step.action(trajectory[20:])

    torch.testing.assert_close(second[0], first[-1], atol=1e-5, rtol=0)
    slope_end = (first[-1] - first[-2]) * step.target_fps
    slope_start = (second[1] - second[0]) * step.target_fps
    torch.testing.assert_close(slope_start, slope_end, atol=0.05, rtol=0.1)


def test_reset_clears_the_alignment_state():
    """After reset() the next chunk is fit fresh and keeps its own first action."""
    step = BsplineActionResamplerStep(source_fps=30.0, target_fps=60.0)
    step.action(torch.randn(10, 4))
    step.reset()
    chunk = torch.randn(10, 4)

    resampled = step.action(chunk)

    torch.testing.assert_close(resampled[0], chunk[0], atol=1e-5, rtol=0)


def test_alignment_disabled_keeps_every_chunk_endpoint():
    step = BsplineActionResamplerStep(source_fps=30.0, target_fps=60.0, align_chunks=False)
    step.action(torch.randn(10, 4))
    chunk = torch.randn(10, 4)

    resampled = step.action(chunk)

    torch.testing.assert_close(resampled[0], chunk[0], atol=1e-5, rtol=0)


def test_alignment_starts_fresh_when_batch_shape_changes():
    """A stored boundary from another batch shape is ignored instead of misapplied."""
    step = BsplineActionResamplerStep(source_fps=30.0, target_fps=60.0)
    step.action(torch.randn(10, 4))
    chunk = torch.randn(2, 10, 4)

    resampled = step.action(chunk)

    torch.testing.assert_close(resampled[:, 0, :], chunk[:, 0, :], atol=1e-5, rtol=0)
