"""Inverse kinematics for the SO-101 (lesson 03).

Two solvers:

1. `top_down_ik` - a closed-form solution for the grasps we actually need: gripper pointing
   straight down at a point (x, y, z) with the jaws lined up at yaw psi. Exact, instant, and it
   returns *all* solutions so we can pick one.
2. `dls_ik` - a general numerical solver (Jacobian + damped least squares) for any target pose.
   Slower and iterative, but it doesn't care about the robot's structure.

The geometry constants are read from the model at construction time rather than hard-coded, so a
change in the model file can't silently invalidate the maths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101.kinematics import Chain, common_normal_feet, invert_T, make_T


def yaw_to_rotation(psi: float) -> np.ndarray:
    """Target tip rotation for a top-down grasp with the jaws at yaw `psi`.

    Tip frame convention (checked in tests):
      x = approach direction, out of the jaws -> must point straight down, (0, 0, -1)
      y = the jaw hinge axis
      z = the direction the jaws close along -> horizontal, at angle psi in the xy plane
    """
    x = np.array([0.0, 0.0, -1.0])
    z = np.array([np.cos(psi), np.sin(psi), 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


@dataclass
class Solution:
    """One IK solution. `elbow` and `shoulder` say which branch it came from."""

    q: np.ndarray
    elbow: str
    shoulder: str
    within_limits: bool
    limit_margin: float  # radians to the nearest joint limit (negative = outside)


class TopDownIK:
    """Closed-form IK for top-down grasps.

    Structure we exploit (lesson 02 found it in the DH table):
      * joints 2, 3, 4 have parallel axes  -> they move the wrist inside one vertical plane;
      * joints 4 and 5 axes intersect      -> the 'wrist centre' W does not move when joint 5
                                              turns, so position and orientation decouple;
      * joint 5's axis is the approach axis -> it only sets the yaw of the jaws.

    So the problem splits into: (a) which way to face -> joint 1; (b) a two-link planar arm
    reaching W -> joints 2 and 3 by the law of cosines; (c) point the gripper down -> joint 4;
    (d) line the jaws up -> joint 5.
    """

    def __init__(self, model, chain: Chain | None = None):
        self.model = model
        self.chain = chain or Chain(model)
        c = self.chain
        self.limits = model.jnt_range[: len(c.joints)].copy()

        F = c.frames(np.zeros(len(c.joints)))
        axes = [(T[:3, 2].copy(), T[:3, 3].copy()) for T in F[1:6]]

        # --- the wrist centre: where the wrist_flex and wrist_roll axes cross
        f1, f2 = common_normal_feet(*axes[3], *axes[4])
        W0 = 0.5 * (f1 + f2)
        assert np.linalg.norm(f1 - f2) < 1e-6, "wrist axes do not intersect"
        T_tip0 = F[-1]
        self.W_in_tip = invert_T(T_tip0)[:3, :3] @ W0 + invert_T(T_tip0)[:3, 3]

        # --- frame 1 = the link that joint 1 turns. Everything below lives in it and is constant.
        T1 = c.joint_transform(0, 0.0)
        to_1 = invert_T(T1)
        self.pan_axis_dir = axes[0][0].copy()
        self.pan_axis_point = axes[0][1].copy()

        n = to_1[:3, :3] @ axes[1][0]                    # pitch-axis direction in frame 1
        A2 = to_1[:3, :3] @ axes[1][1] + to_1[:3, 3]     # shoulder_lift axis point in frame 1
        A3 = to_1[:3, :3] @ axes[2][1] + to_1[:3, 3]     # elbow_flex axis point
        W1 = to_1[:3, :3] @ W0 + to_1[:3, 3]             # wrist centre
        self.n = n
        self.lateral = float(n @ W1)                     # W's fixed offset out of the arm plane

        # in-plane basis (e1, e2), both perpendicular to n
        e1 = np.array([0.0, 0.0, 1.0])
        e1 = e1 - (e1 @ n) * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        self.e1, self.e2 = e1, e2
        self.A2_planar = np.array([A2 @ e1, A2 @ e2])

        v1 = self._planar(A3) - self.A2_planar          # upper arm, at q = 0
        v2 = self._planar(W1) - self._planar(A3)        # forearm + wrist, at q = 0
        self.L1, self.L2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        self.theta_a = float(np.arctan2(v1[1], v1[0]))  # absolute in-plane angle at q = 0
        self.theta_b = float(np.arctan2(v2[1], v2[0]))

        approach0 = to_1[:3, :3] @ T_tip0[:3, 0]        # approach direction in frame 1, at q = 0
        ap = self._planar_dir(approach0)
        self.theta_c = float(np.arctan2(ap[1], ap[0]))

        # --- signs: does a positive joint angle increase or decrease the in-plane angle?
        eps = 1e-3
        F_eps = c.frames([0.0, eps, 0.0, 0.0, 0.0])
        v1_eps = self._planar(to_1[:3, :3] @ F_eps[3][:3, 3] + to_1[:3, 3]) - self.A2_planar
        self.s = 1.0 if np.arctan2(v1_eps[1], v1_eps[0]) > self.theta_a else -1.0

        F_eps = c.frames([0.0, 0.0, 0.0, 0.0, eps])     # roll: which way does the jaw axis turn?
        z_eps = F_eps[-1][:3, 2]
        z_0 = T_tip0[:3, 2]
        ang = np.arctan2(np.cross(z_0, z_eps) @ T_tip0[:3, 0], z_0 @ z_eps)
        self.s_roll = 1.0 if ang > 0 else -1.0

    # ------------------------------------------------------------------ helpers

    def _planar(self, p) -> np.ndarray:
        return np.array([p @ self.e1, p @ self.e2])

    def _planar_dir(self, v) -> np.ndarray:
        return np.array([v @ self.e1, v @ self.e2])

    def wrist_centre(self, position, psi: float) -> np.ndarray:
        """Where the wrist centre must be for the tip to sit at `position` with yaw `psi`."""
        R = yaw_to_rotation(psi)
        return np.asarray(position, float) + R @ self.W_in_tip

    # ------------------------------------------------------------------ the solver

    def solve(self, position, psi: float, *, all_solutions: bool = False,
              half_turn: bool = True):
        """Joint angles that put the tip at `position` pointing down, jaws at yaw `psi`.

        Returns the best solution (inside the joint limits, furthest from them), or None.
        With all_solutions=True, returns every branch, including ones outside the limits.
        half_turn=False suppresses the psi+180 alternative; `grasp()` needs that, because the
        grasp point is off the roll axis and so moves when the jaws turn over.
        """
        out: list[Solution] = []
        # A parallel gripper grasping a box looks the same after half a turn, so psi and psi+180
        # are both valid. They are *different* targets, though: the tip sits 7.9 mm off the roll
        # axis, so each one needs its own wrist centre.
        variants = (psi, self._wrap(psi + np.pi)) if half_turn else (psi,)
        for psi_variant in variants:
            W = self.wrist_centre(position, psi_variant)
            R_target = yaw_to_rotation(psi_variant)

            for q1, shoulder in self._pan_angles(W):
                T1 = self.chain.joint_transform(0, q1)
                to_1 = invert_T(T1)
                W1 = to_1[:3, :3] @ W + to_1[:3, 3]
                target = self._planar(W1) - self.A2_planar

                for q2, q3 in self._planar_2r(target):
                    q4 = self._wrist_angle(q1, q2, q3, to_1)
                    q5 = self._roll_angle([q1, q2, q3, q4], R_target)
                    q = np.array([q1, q2, q3, q4, q5])
                    elbow = self._elbow_branch(q)
                    margin = float(np.min(np.minimum(q - self.limits[:, 0],
                                                     self.limits[:, 1] - q)))
                    out.append(Solution(q, elbow, shoulder, margin >= 0, margin))

        if all_solutions:
            return out
        ok = [s for s in out if s.within_limits]
        return max(ok, key=lambda s: s.limit_margin) if ok else None

    # --------------------------------------------------------------- the four steps

    def _pan_angles(self, W):
        """Step 1: turn the arm's plane so that it contains the wrist centre.

        The plane sits `lateral` metres to the side of the pan axis, so the condition is
            v . n(q1) = lateral,   i.e.   A sin q1 + B cos q1 = C,
        which has two solutions (arm in front / arm behind) whenever |C| <= sqrt(A^2 + B^2).
        """
        v = np.asarray(W, float) - self.pan_axis_point
        A, B, C = v[0], v[1], self.lateral
        r = np.hypot(A, B)
        if r < 1e-12 or abs(C) > r:
            return []
        phi = np.arctan2(B, A)
        base = np.arcsin(np.clip(C / r, -1.0, 1.0))
        return [(self._wrap(base - phi), "front"), (self._wrap(np.pi - base - phi), "back")]

    def _planar_2r(self, target):
        """Step 2: two-link planar arm, solved with the law of cosines.

        With D = |target|, the triangle (L1, L2, D) gives the interior elbow angle; the two signs
        are the 'elbow up' and 'elbow down' postures.
        """
        D = float(np.linalg.norm(target))
        if D > self.L1 + self.L2 or D < abs(self.L1 - self.L2) or D < 1e-9:
            return []
        cos_gamma = (D ** 2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        gamma = float(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))
        alpha = float(np.arctan2(target[1], target[0]))
        out = []
        for g in (gamma, -gamma):
            phi1 = alpha - np.arctan2(self.L2 * np.sin(g), self.L1 + self.L2 * np.cos(g))
            phi2 = phi1 + g
            q2 = self._wrap((phi1 - self.theta_a) / self.s)
            q3 = self._wrap((phi2 - self.theta_b) / self.s - q2)
            out.append((q2, q3))
        return out

    def _elbow_branch(self, q) -> str:
        """Is the elbow above or below the straight line from the shoulder to the wrist?"""
        F = self.chain.frames(q)
        shoulder, elbow, wrist = F[2][:3, 3], F[3][:3, 3], F[4][:3, 3]
        span = wrist - shoulder
        t = float((elbow - shoulder) @ span / (span @ span))
        return "up" if elbow[2] > shoulder[2] + t * span[2] else "down"

    def _wrist_angle(self, q1, q2, q3, to_1) -> float:
        """Step 3: whatever is left over to make the approach point straight down."""
        down_1 = to_1[:3, :3] @ np.array([0.0, 0.0, -1.0])
        ap = self._planar_dir(down_1)
        phi_target = np.arctan2(ap[1], ap[0])
        return self._wrap((phi_target - self.theta_c) / self.s - (q2 + q3))

    def _roll_angle(self, q4, R_target) -> float:
        """Step 4: spin the gripper about its own axis until the jaws line up with the yaw."""
        R_now = self.chain.fk([*q4, 0.0])[:3, :3]
        M = R_now.T @ R_target                      # residual rotation, about the approach axis
        return self._wrap(self.s_roll * np.arctan2(M[2, 1], M[1, 1]))

    @staticmethod
    def _wrap(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    # ------------------------------------------------------------------ workspace

    def reachable(self, position, psi: float = 0.0) -> bool:
        return self.solve(position, psi) is not None

    # ------------------------------------------------------------------ grasp targets

    def grasp(self, point, psi: float = 0.0, *, all_solutions: bool = False):
        """Like solve(), but `point` is where the OBJECT should sit between the jaws.

        The half-turn alternative is handled here rather than inside solve(), because the grasp
        centre is offset from the roll axis: turning the jaws over moves it, so each yaw needs
        its own tip target. Getting this wrong puts the object down 2.8 cm from the target.
        """
        out = []
        for psi_k in (psi, self._wrap(psi + np.pi)):
            out += self.solve(_tip_for_grasp(point, psi_k), psi_k,
                              all_solutions=True, half_turn=False)
        if all_solutions:
            return out
        ok = [s for s in out if s.within_limits]
        return max(ok, key=lambda s: s.limit_margin) if ok else None

    def can_grasp(self, point, psi: float = 0.0) -> bool:
        return self.grasp(point, psi) is not None


# ------------------------------------------------------------ numerical IK (general)

def geometric_jacobian(chain: Chain, q) -> np.ndarray:
    """6 x n Jacobian of the tip pose: [linear; angular] velocity per unit joint rate.

    For a revolute joint with world axis z_i through point p_i, turning it at 1 rad/s moves the
    tip at z_i x (p_tip - p_i) and rotates it at z_i. That is the whole derivation.
    """
    frames = chain.frames(q)
    p_tip = frames[-1][:3, 3]
    J = np.zeros((6, len(chain.joints)))
    for i in range(len(chain.joints)):
        z_i = frames[i + 1][:3, 2]
        p_i = frames[i + 1][:3, 3]
        J[:3, i] = np.cross(z_i, p_tip - p_i)
        J[3:, i] = z_i
    return J


def pose_error(T_now: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """6-vector error: position difference, then orientation as a rotation vector."""
    e_p = T_target[:3, 3] - T_now[:3, 3]
    R_err = T_target[:3, :3] @ T_now[:3, :3].T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    if angle < 1e-12:
        e_r = np.zeros(3)
    else:
        axis = np.array([R_err[2, 1] - R_err[1, 2],
                         R_err[0, 2] - R_err[2, 0],
                         R_err[1, 0] - R_err[0, 1]]) / (2 * np.sin(angle))
        e_r = axis * angle
    return np.concatenate([e_p, e_r])


def dls_ik(chain: Chain, T_target: np.ndarray, q0, *, damping: float = 0.05,
           iters: int = 200, tol: float = 1e-6, limits=None):
    """Damped least squares (Levenberg-Marquardt) IK: iterate q <- q + J^T (J J^T + k^2 I)^-1 e.

    Plain least squares (the pseudo-inverse) blows up near singularities, where J loses rank and
    tiny errors ask for enormous joint rates. The damping term k^2 I keeps the matrix invertible,
    trading a little accuracy for stability.
    """
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        e = pose_error(chain.fk(q), T_target)
        if np.linalg.norm(e) < tol:
            break
        J = geometric_jacobian(chain, q)
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        q = q + np.clip(dq, -0.2, 0.2)
        if limits is not None:
            q = np.clip(q, limits[:, 0], limits[:, 1])
    return q, float(np.linalg.norm(pose_error(chain.fk(q), T_target)))


# --------------------------------------------------------- where the grasp actually happens

#: The centre of the jaw opening, in the tip ("gripperframe") frame, in metres.
#: The model's site sits at the wrist output, not between the fingers, so every grasp target has
#: to be shifted by this. Measured from the finger meshes and confirmed by lifting the cube from
#: a grid of candidate offsets (see docs/lessons/04-trajectories-and-control.md).
GRASP_OFFSET = np.array([-0.030, 0.0, 0.014])

#: Fallback gripper angles [rad]. Prefer `pick_place.jaw_angle_for_gap`, which computes the angle
#: from the width you actually want between the pads.
GRIPPER_OPEN = np.radians(60.0)     # 4.4 cm gap: clears a 3 cm cube
GRIPPER_CLOSED = np.radians(25.0)   # 2.85 cm gap: squeezes a 3 cm cube by ~1.5 mm


def _tip_for_grasp(point, psi: float) -> np.ndarray:
    return np.asarray(point, float) - yaw_to_rotation(psi) @ GRASP_OFFSET
