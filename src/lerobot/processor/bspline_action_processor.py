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

"""B-spline resampling of policy action chunks.

Chunked policies (ACT, Diffusion Policy, VLAs) predict actions at the rate their training data was
recorded, which is often much lower than the rate the robot is controlled at (e.g. a 30 Hz chunk
driving a 100 Hz control loop). Replaying the chunk with zero-order hold or linear interpolation
produces velocity discontinuities at every waypoint and at chunk boundaries.

This processor fits a B-spline curve to the predicted chunk and resamples it at the control rate,
following the action representation of B-spline Policy (https://arxiv.org/abs/2607.09648). The
fitted curve is time-continuous, so the resampled trajectory is smooth, and its density is
decoupled from the policy's prediction rate. Fitting with fewer control points than waypoints
additionally low-pass filters the chunk. Consecutive chunks are joined with continuous position and
velocity by constraining each new fit to start where the previous curve ended.

The fit is solved in closed form with a linear least-squares system built from the Cox-de Boor
basis, entirely in PyTorch.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import PolicyAction

from .pipeline import ActionProcessorStep, ProcessorStepRegistry


def _clamped_uniform_knots(
    num_control_points: int, degree: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Builds a clamped, uniformly spaced knot vector on [0, 1].

    Clamping (repeating the boundary knots ``degree + 1`` times) makes the curve start exactly at
    the first control point and end exactly at the last one.

    Args:
        num_control_points: Number of control points the spline will have.
        degree: Polynomial degree of the spline.
        dtype: Dtype of the returned knot vector.
        device: Device of the returned knot vector.

    Returns:
        A knot vector of shape ``(num_control_points + degree + 1,)``.
    """
    num_interior = num_control_points - degree - 1
    boundary = torch.linspace(0.0, 1.0, num_interior + 2, dtype=dtype, device=device)
    zeros = torch.zeros(degree, dtype=dtype, device=device)
    ones = torch.ones(degree, dtype=dtype, device=device)
    return torch.cat([zeros, boundary, ones])


def _bspline_basis(times: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluates all B-spline basis functions at the given times (Cox-de Boor recursion).

    Args:
        times: Evaluation times of shape ``(num_times,)``, within ``[knots[0], knots[-1]]``.
        knots: Knot vector of shape ``(num_knots,)``.
        degree: Polynomial degree of the basis.

    Returns:
        Basis matrix of shape ``(num_times, num_knots - degree - 1)`` whose rows sum to 1.
    """
    t = times.unsqueeze(-1)
    left, right = knots[:-1], knots[1:]

    # Degree 0: indicator of the half-open knot span [knots[i], knots[i+1]).
    basis = ((t >= left) & (t < right)).to(times.dtype)

    # Close the domain at the final knot so times equal to knots[-1] fall in the last
    # non-degenerate span instead of nowhere.
    last_span = torch.nonzero(right > left).max()
    basis[times >= knots[-1], :] = 0.0
    basis[times >= knots[-1], last_span] = 1.0

    for p in range(1, degree + 1):
        left_den = knots[p:-1] - knots[: -p - 1]
        right_den = knots[p + 1 :] - knots[1:-p]
        safe_left_den = torch.where(left_den > 0, left_den, torch.ones_like(left_den))
        safe_right_den = torch.where(right_den > 0, right_den, torch.ones_like(right_den))
        left_term = torch.where(left_den > 0, (t - knots[: -p - 1]) / safe_left_den, torch.zeros_like(t))
        right_term = torch.where(right_den > 0, (knots[p + 1 :] - t) / safe_right_den, torch.zeros_like(t))
        basis = left_term * basis[:, :-1] + right_term * basis[:, 1:]

    return basis


@ProcessorStepRegistry.register("bspline_action_resampler")
@dataclass
class BsplineActionResamplerStep(ActionProcessorStep):
    """Resamples a chunk of policy actions to the control rate through a B-spline fit.

    The step expects an action chunk of shape ``(chunk_size, action_dim)`` or
    ``(batch, chunk_size, action_dim)``, fits one B-spline per action dimension and returns the
    curve sampled at the target rate, with shape ``(new_chunk_size, action_dim)`` (respectively
    batched). The chunk's duration is preserved: ``new_chunk_size`` is
    ``round(chunk_size * target_fps / source_fps)``.

    With ``align_chunks`` enabled (the default), the step remembers where the previous curve ended
    and constrains each new fit to start there with the same velocity, so consecutive chunks join
    without position or velocity discontinuities. This assumes chunks are executed to completion,
    one after the other; for overlapping-chunk schemes (e.g. async inference with an aggregator),
    disable it. Call :meth:`reset` at episode boundaries to clear the alignment state. The first
    chunk after construction or reset, and every chunk when alignment is disabled, reproduces its
    own first and last actions exactly (clamped spline).

    Attributes:
        source_fps: Rate at which the policy predicts actions (the chunk's native rate).
        target_fps: Rate at which the robot consumes actions (the control loop rate).
        degree: Polynomial degree of the spline. Cubic (3) gives continuous velocity and
            acceleration, which is what makes the resampled trajectory smooth to execute.
        num_control_points: Number of spline control points. ``None`` uses one per waypoint, which
            closely follows the chunk. Fewer control points low-pass filter the chunk on top of
            resampling it. Chunk alignment needs at least ``degree + 2`` control points and is
            skipped otherwise.
        align_chunks: Whether to join consecutive chunks with continuous position and velocity.
    """

    source_fps: float = 30.0
    target_fps: float = 100.0
    degree: int = 3
    num_control_points: int | None = None
    align_chunks: bool = True

    _boundary_position: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)
    _boundary_velocity: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.source_fps <= 0 or self.target_fps <= 0:
            raise ValueError(
                f"source_fps and target_fps must be positive, got {self.source_fps} and {self.target_fps}."
            )
        if self.degree < 1:
            raise ValueError(f"degree must be at least 1, got {self.degree}.")
        if self.num_control_points is not None and self.num_control_points < 2:
            raise ValueError(f"num_control_points must be at least 2, got {self.num_control_points}.")

    def action(self, action: PolicyAction) -> PolicyAction:
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"{self.__class__.__name__} expects a tensor action, got {type(action)}.")
        if action.ndim not in (2, 3):
            raise ValueError(
                f"{self.__class__.__name__} expects an action chunk of shape (chunk_size, action_dim) "
                f"or (batch, chunk_size, action_dim), got shape {tuple(action.shape)}. Single actions "
                "cannot be resampled; place this step where the full chunk is available."
            )

        batched = action.ndim == 3
        chunk = action if batched else action.unsqueeze(0)
        batch_size, chunk_size, action_dim = chunk.shape
        if chunk_size < 2:
            raise ValueError(f"Resampling needs a chunk of at least 2 actions, got {chunk_size}.")

        degree = min(self.degree, chunk_size - 1)
        num_control_points = self.num_control_points if self.num_control_points is not None else chunk_size
        num_control_points = max(degree + 1, min(num_control_points, chunk_size))
        new_chunk_size = max(2, round(chunk_size * self.target_fps / self.source_fps))
        duration_s = chunk_size / self.source_fps

        compute_dtype = torch.promote_types(chunk.dtype, torch.float32)
        curve = chunk.to(compute_dtype)
        knots = _clamped_uniform_knots(num_control_points, degree, compute_dtype, chunk.device)

        fit_times = torch.linspace(0.0, 1.0, chunk_size, dtype=compute_dtype, device=chunk.device)
        fit_basis = _bspline_basis(fit_times, knots, degree)
        targets = curve.permute(1, 0, 2).reshape(chunk_size, batch_size * action_dim)

        control_points = self._fit_control_points(
            fit_basis, targets, knots, degree, num_control_points, duration_s, batch_size, action_dim
        )

        sample_times = torch.linspace(0.0, 1.0, new_chunk_size, dtype=compute_dtype, device=chunk.device)
        sample_basis = _bspline_basis(sample_times, knots, degree)
        resampled = sample_basis @ control_points
        resampled = resampled.reshape(new_chunk_size, batch_size, action_dim).permute(1, 0, 2)
        resampled = resampled.to(chunk.dtype)

        if self.align_chunks:
            self._store_boundary_state(
                control_points, knots, degree, num_control_points, duration_s, batch_size, action_dim
            )

        return resampled if batched else resampled.squeeze(0)

    def _fit_control_points(
        self,
        fit_basis: torch.Tensor,
        targets: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        num_control_points: int,
        duration_s: float,
        batch_size: int,
        action_dim: int,
    ) -> torch.Tensor:
        """Solves for the control points, constrained by the previous chunk's end when aligning.

        For a clamped spline the curve starts at the first control point with a velocity set by the
        first two, so pinning those two enforces position (C0) and velocity (C1) continuity with the
        previous chunk; the remaining control points are fit to the new chunk by least squares.
        """
        can_align = (
            self.align_chunks
            and self._boundary_position is not None
            and self._boundary_position.shape == (batch_size, action_dim)
            and num_control_points >= degree + 2
        )
        if not can_align:
            return torch.linalg.lstsq(fit_basis, targets).solution

        start = self._boundary_position.to(dtype=targets.dtype, device=targets.device)
        velocity = self._boundary_velocity.to(dtype=targets.dtype, device=targets.device)
        start_span = (knots[degree + 1] - knots[1]) / degree
        second = start + velocity * duration_s * start_span
        pinned = torch.stack([start, second]).reshape(2, batch_size * action_dim)

        residual = targets - fit_basis[:, :2] @ pinned
        free = torch.linalg.lstsq(fit_basis[:, 2:], residual).solution
        return torch.cat([pinned, free])

    def _store_boundary_state(
        self,
        control_points: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        num_control_points: int,
        duration_s: float,
        batch_size: int,
        action_dim: int,
    ) -> None:
        """Remembers the curve's end position and velocity for the next chunk's fit."""
        last = control_points[-1].reshape(batch_size, action_dim)
        second_last = control_points[-2].reshape(batch_size, action_dim)
        end_span = (knots[num_control_points + degree - 1] - knots[num_control_points - 1]) / degree
        velocity_s = (last - second_last) / (end_span * duration_s)
        self._boundary_position = last.detach()
        self._boundary_velocity = velocity_s.detach()

    def get_config(self) -> dict[str, Any]:
        """Returns the serializable configuration of the processor.

        Returns:
            A dictionary with the resampling rates and spline parameters.
        """
        return {
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "degree": self.degree,
            "num_control_points": self.num_control_points,
            "align_chunks": self.align_chunks,
        }

    def reset(self) -> None:
        """Clears the chunk-alignment state. Call between episodes."""
        self._boundary_position = None
        self._boundary_velocity = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Resampling changes how many actions a chunk holds, not the shape of each action."""
        return features
