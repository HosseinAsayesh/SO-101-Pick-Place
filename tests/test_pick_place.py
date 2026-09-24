"""Lesson 04 checks: time scaling, paths, the gripper, and a full pick-and-place. Run: pytest"""

import mujoco
import numpy as np
import pytest

from so101 import load_model
from so101.ik import GRASP_OFFSET, TopDownIK, yaw_to_rotation
from so101.kinematics import Chain
from so101.pick_place import (
    jaw_angle_for_gap,
    judge,
    pad_gap,
    plan_pick_place,
    run_plan,
    run_trial,
    set_cube,
    tracking_error,
)
from so101.trajectory import (
    Trajectory,
    cartesian_segment,
    cubic,
    hold_segment,
    joint_segment,
    quintic,
    quintic_derivatives,
)

PLACE = (0.15, -0.15)


@pytest.fixture(scope="module")
def sim():
    model, data = load_model()
    chain = Chain(model)
    return model, data, chain, TopDownIK(model, chain)


# ------------------------------------------------------------------- time scaling

def test_time_scaling_endpoints():
    for fn in (cubic, quintic):
        assert fn(0.0) == pytest.approx(0.0)
        assert fn(1.0) == pytest.approx(1.0)
        assert np.all(np.diff(fn(np.linspace(0, 1, 50))) >= -1e-12)   # monotone


def test_quintic_starts_and_stops_gently():
    """Zero velocity AND zero acceleration at both ends - that is why we prefer it to the cubic."""
    for tau in (0.0, 1.0):
        s, ds, dds = quintic_derivatives(tau, duration=2.0)
        assert ds == pytest.approx(0.0, abs=1e-12)
        assert dds == pytest.approx(0.0, abs=1e-12)
    # the cubic does NOT have zero acceleration at the ends
    eps = 1e-5
    accel = (cubic(eps) - 2 * cubic(0.0) + cubic(-eps)) / eps ** 2
    assert abs(accel) > 1.0


def test_quintic_peak_speed_is_1_875_times_average():
    s, ds, _ = quintic_derivatives(0.5, duration=1.0)
    assert ds == pytest.approx(1.875, abs=1e-9)


# ---------------------------------------------------------------------- segments

def test_joint_segment_hits_both_ends():
    a, b = np.zeros(5), np.radians([10, -20, 30, -40, 50])
    seg = joint_segment(a, b, 2.0)
    assert np.allclose(seg.q(0.0), a)
    assert np.allclose(seg.q(2.0), b)


def test_hold_segment_never_moves():
    q = np.radians([5, 5, 5, 5, 5])
    seg = hold_segment(q, 1.0)
    assert np.allclose(seg.q(0.3), q) and np.allclose(seg.q(1.0), q)


def test_cartesian_segment_is_straight(sim):
    """The whole point of a Cartesian segment: the grasp point travels in a straight line."""
    _, _, chain, ik = sim
    p0 = np.array([0.22, 0.08, 0.07])
    p1 = np.array([0.22, 0.08, 0.02])
    seg = cartesian_segment(ik, p0, p1, 0.0, 1.0)
    deviations = []
    for s in np.linspace(0, 1, 21):
        T = chain.fk(seg.q_of_s(s))
        point = T[:3, 3] + T[:3, :3] @ GRASP_OFFSET
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        deviations.append(np.linalg.norm((point - p0) - ((point - p0) @ u) * u))
    assert max(deviations) < 1e-3          # under a millimetre off the line


def test_trajectory_concatenates(sim):
    a, b = np.zeros(5), np.radians([10, 0, 0, 0, 0])
    traj = Trajectory([joint_segment(a, b, 1.0), hold_segment(b, 0.5)])
    assert traj.duration == pytest.approx(1.5)
    assert np.allclose(traj.at(0.0)[0], a)
    assert np.allclose(traj.at(1.4)[0], b)
    assert np.allclose(traj.at(99.0)[0], b)   # clamps at the end


# ----------------------------------------------------------------------- gripper

def test_pad_gap_opens_with_the_jaw(sim):
    model, data, _, _ = sim
    gaps = [pad_gap(model, data, np.radians(a)) for a in (0, 15, 30, 45, 60)]
    assert all(np.diff(gaps) > 0)
    assert gaps[0] < 0.01 and gaps[-1] > 0.04


def test_jaw_angle_for_gap_round_trip(sim):
    model, data, _, _ = sim
    for gap in (0.02, 0.03, 0.04):
        angle = jaw_angle_for_gap(model, data, gap)
        assert pad_gap(model, data, angle) == pytest.approx(gap, abs=1e-3)


def test_pads_clear_the_table_at_the_grasp_pose(sim):
    """A pad that reaches below the table stalls the arm and the grasp misses (lesson 04 bug 3)."""
    model, data, _, ik = sim
    mujoco.mj_resetData(model, data)
    set_cube(model, data, 0.22, 0.08, 0.0)
    plan = plan_pick_place(model, Chain(model), ik, (0.22, 0.08), PLACE, data=data)
    from so101.model import set_joint_angles
    set_joint_angles(model, data, [*plan.grasp_q, np.radians(55)])
    for name in ("pad_fixed", "pad_moving"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        half = model.geom_size[gid]
        corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        world = (data.geom_xmat[gid].reshape(3, 3) @ corners.T).T + data.geom_xpos[gid]
        assert world[:, 2].min() > 0.002, f"{name} is inside the table"


def test_grasp_offset_survives_the_half_turn(sim):
    """psi and psi+180 are both valid grasps, and both must put the OBJECT in the same place."""
    _, _, chain, ik = sim
    point = np.array([0.22, 0.08, 0.02])
    for sol in ik.grasp(point, np.radians(30), all_solutions=True):
        if not sol.within_limits:
            continue
        T = chain.fk(sol.q)
        assert np.allclose(T[:3, 3] + T[:3, :3] @ GRASP_OFFSET, point, atol=1e-9)


# ------------------------------------------------------------------- integration

@pytest.mark.parametrize("pick,yaw", [((0.22, 0.08), 0.0), ((0.19, -0.05), np.radians(40))])
def test_full_pick_and_place_succeeds(sim, pick, yaw):
    model, data, chain, ik = sim
    result = run_trial(model, data, chain, ik, pick, PLACE, yaw=yaw)
    assert result["success"], result["reason"]
    assert result["final_xy_error_mm"] < 20


def test_servos_track_the_trajectory(sim):
    model, data, chain, ik = sim
    mujoco.mj_resetData(model, data)
    set_cube(model, data, 0.22, 0.08, 0.0)
    plan = plan_pick_place(model, chain, ik, (0.22, 0.08), PLACE, data=data)
    log = run_plan(model, data, chain, plan, reset=False)
    assert np.degrees(np.abs(tracking_error(log))).max() < 1.0
    assert judge(log, PLACE)["success"]
