"""Lesson 03 checks: inverse kinematics. Run with:  pytest"""

import numpy as np
import pytest

from so101 import ARM_JOINTS, load_model
from so101.ik import TopDownIK, dls_ik, geometric_jacobian, pose_error, yaw_to_rotation
from so101.kinematics import Chain, make_T

GRASP_Z = 0.015


@pytest.fixture(scope="module")
def ik():
    model, _ = load_model()
    chain = Chain(model)
    return TopDownIK(model, chain), chain


def grasp_errors(chain, q, target, psi):
    T = chain.fk(q)
    pos = np.linalg.norm(T[:3, 3] - target)
    approach = np.degrees(np.arccos(np.clip(T[:3, 0] @ [0, 0, -1], -1, 1)))
    yaw = np.arctan2(T[1, 2], T[0, 2])
    yaw_err = np.degrees(abs((yaw - psi + np.pi / 2) % np.pi - np.pi / 2))
    return pos, approach, yaw_err


def test_yaw_to_rotation_is_a_rotation():
    for psi in np.linspace(-np.pi, np.pi, 9):
        R = yaw_to_rotation(psi)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(R[:, 0], [0, 0, -1])            # approach points down
        assert R[2, 2] == pytest.approx(0.0, abs=1e-12)    # jaw direction is horizontal


def test_geometry_constants(ik):
    solver, _ = ik
    assert solver.L1 == pytest.approx(0.116, abs=1e-4)
    assert solver.L2 == pytest.approx(0.135, abs=1e-4)
    assert abs(solver.lateral) < 1e-3          # the arm plane nearly contains the pan axis


def test_round_trip_random_targets(ik):
    """IK then FK must return the target, for every target we claim to solve."""
    solver, chain = ik
    rng = np.random.default_rng(3)
    solved = 0
    for _ in range(200):
        p = np.array([rng.uniform(0.05, 0.32), rng.uniform(-0.25, 0.25), GRASP_Z])
        psi = rng.uniform(-np.pi, np.pi)
        s = solver.solve(p, psi)
        if s is None:
            continue
        solved += 1
        e_p, e_a, e_y = grasp_errors(chain, s.q, p, psi)
        assert e_p < 1e-9
        assert e_a < 1e-4
        assert e_y < 1e-4
    assert solved > 100, "expected most table targets to be solvable"


def test_solutions_respect_joint_limits(ik):
    solver, _ = ik
    lo, hi = solver.limits.T
    for _ in range(50):
        s = solver.solve([0.22, 0.08, GRASP_Z], np.random.default_rng(1).uniform(-np.pi, np.pi))
        if s is not None:
            assert np.all(s.q >= lo - 1e-12) and np.all(s.q <= hi + 1e-12)


def test_branches_are_distinct_but_equivalent(ik):
    """Different postures, same tip pose."""
    solver, chain = ik
    target, psi = np.array([0.22, 0.08, GRASP_Z]), np.radians(30)
    sols = solver.solve(target, psi, all_solutions=True)
    assert len(sols) >= 4
    assert {s.elbow for s in sols} == {"up", "down"}
    assert {s.shoulder for s in sols} == {"front", "back"}
    for s in sols:
        e_p, e_a, e_y = grasp_errors(chain, s.q, target, psi)
        assert e_p < 1e-9 and e_a < 1e-4 and e_y < 1e-4


def test_unreachable_targets_return_none(ik):
    solver, _ = ik
    assert solver.solve([0.60, 0.0, GRASP_Z], 0.0) is None     # beyond reach
    assert solver.solve([0.0, 0.0, 0.6], 0.0) is None          # above the head, too high
    assert solver.solve([0.02, 0.0, GRASP_Z], 0.0) is None     # inside the base


def test_elbow_up_is_actually_higher(ik):
    solver, chain = ik
    sols = solver.solve([0.25, 0.0, GRASP_Z], 0.0, all_solutions=True)
    ups = [chain.frames(s.q)[3][2, 3] for s in sols if s.elbow == "up"]
    downs = [chain.frames(s.q)[3][2, 3] for s in sols if s.elbow == "down"]
    assert min(ups) > max(downs)


# ------------------------------------------------------------------- numerical IK

def test_jacobian_matches_finite_differences(ik):
    solver, chain = ik
    q = np.radians([20, -30, 40, -20, 35])
    J = geometric_jacobian(chain, q)
    eps = 1e-6
    for i in range(len(ARM_JOINTS)):
        dq = np.zeros(5)
        dq[i] = eps
        num = pose_error(chain.fk(q), chain.fk(q + dq)) / eps
        assert np.allclose(J[:, i], num, atol=1e-5)


def test_dls_finds_the_same_pose(ik):
    solver, chain = ik
    target, psi = np.array([0.22, 0.08, GRASP_Z]), np.radians(30)
    closed = solver.solve(target, psi)
    q_num, err = dls_ik(chain, make_T(yaw_to_rotation(psi), target), np.zeros(5),
                        limits=solver.limits)
    assert err < 1e-5
    assert np.abs(q_num - closed.q).max() < np.radians(0.5)


def test_top_down_height_ceiling(ik):
    """wrist_flex's +-95 deg limit caps how high the gripper can hover pointing down (~9 cm)."""
    solver, _ = ik
    assert solver.reachable([0.22, 0.08, 0.05], 0.0)
    assert not solver.reachable([0.22, 0.08, 0.12], 0.0)
    # the ceiling is a joint-limit effect, not a geometric one: the branches exist, but are outside
    assert len(solver.solve([0.22, 0.08, 0.12], 0.0, all_solutions=True)) > 0
