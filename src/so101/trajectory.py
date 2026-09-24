"""Trajectories: turning a list of poses into a smooth motion in time (lesson 04).

IK gives us *poses*. A robot needs a *trajectory*: a joint angle for every instant, smooth enough
that the servos can follow it and the object doesn't get knocked over on the way.

Two pieces:
  * a **time scaling** s(t) that goes 0 -> 1 smoothly (the "when"),
  * a **path** that the scaling moves along (the "where").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------- time scaling

def cubic(tau):
    """s(tau) = 3 tau^2 - 2 tau^3.  Starts and ends at zero VELOCITY."""
    tau = np.clip(tau, 0.0, 1.0)
    return 3 * tau ** 2 - 2 * tau ** 3


def quintic(tau):
    """s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5.  Zero velocity AND zero acceleration at both ends.

    Derivation: we want a polynomial with s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    Six conditions -> five degrees + constant -> a quintic. Solving the linear system gives the
    coefficients above. Zero end-acceleration matters for a servo: acceleration is proportional to
    torque, so a cubic asks for a torque step at the start of every segment, which the motor
    answers with a jolt.
    """
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def quintic_derivatives(tau, duration: float = 1.0):
    """(s, ds/dt, d2s/dt2) for the quintic, at normalized time tau."""
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / duration
    dds = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / duration ** 2
    return s, ds, dds


# ------------------------------------------------------------------------- segments

@dataclass
class Segment:
    """One piece of a motion: a function of normalized time, plus how long it lasts."""

    duration: float
    q_of_s: Callable[[float], np.ndarray]
    gripper: float | None = None           # commanded gripper angle during this segment
    label: str = ""
    scaling: Callable = quintic

    def q(self, t: float) -> np.ndarray:
        return np.asarray(self.q_of_s(float(self.scaling(t / self.duration))), float)


def joint_segment(q_start, q_end, duration: float, *, gripper=None, label="") -> Segment:
    """Straight line in JOINT space: simple, fast, and the tip takes a curved path."""
    q_start, q_end = np.asarray(q_start, float), np.asarray(q_end, float)
    return Segment(duration, lambda s: q_start + s * (q_end - q_start),
                   gripper=gripper, label=label)


def hold_segment(q, duration: float, *, gripper=None, label="") -> Segment:
    """Stay put (used while the gripper opens or closes)."""
    q = np.asarray(q, float)
    return Segment(duration, lambda s: q, gripper=gripper, label=label)


def cartesian_segment(ik, p_start, p_end, psi: float, duration: float, *, q_seed=None,
                      samples: int = 25, gripper=None, label="") -> Segment:
    """Straight line in CARTESIAN space for the grasp point, solved by IK at `samples` knots.

    Needed whenever the shape of the path matters: lowering onto a cube must go straight down,
    not along whatever curve the joints happen to trace.

    Each knot picks the IK branch closest to the previous one, so the arm can't flip posture
    half way down. The knots are then interpolated linearly in joint space, which is accurate
    because consecutive knots are millimetres apart.
    """
    p_start, p_end = np.asarray(p_start, float), np.asarray(p_end, float)
    qs, seed = [], None if q_seed is None else np.asarray(q_seed, float)
    for u in np.linspace(0.0, 1.0, samples):
        point = p_start + u * (p_end - p_start)
        sols = [s for s in ik.grasp(point, psi, all_solutions=True) if s.within_limits]
        if not sols:
            raise ValueError(f"unreachable point on the path: {point}")
        best = min(sols, key=lambda s: np.linalg.norm(s.q - seed)) if seed is not None else \
            max(sols, key=lambda s: s.limit_margin)
        seed = best.q
        qs.append(best.q)
    qs = np.array(qs)
    us = np.linspace(0.0, 1.0, samples)

    def q_of_s(s):
        return np.array([np.interp(s, us, qs[:, j]) for j in range(qs.shape[1])])

    return Segment(duration, q_of_s, gripper=gripper, label=label)


# ---------------------------------------------------------------------- trajectory

@dataclass
class Trajectory:
    """A list of segments played back to back."""

    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def at(self, t: float) -> tuple[np.ndarray, float | None, str]:
        """(joint angles, gripper command, segment label) at time t."""
        t = max(0.0, float(t))
        for seg in self.segments:
            if t <= seg.duration:
                return seg.q(t), seg.gripper, seg.label
            t -= seg.duration
        last = self.segments[-1]
        return last.q(last.duration), last.gripper, last.label

    def sample(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        ts = np.arange(0.0, self.duration + dt, dt)
        return ts, np.array([self.at(t)[0] for t in ts])
