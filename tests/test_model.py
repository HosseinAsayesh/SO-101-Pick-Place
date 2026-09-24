"""Sanity checks for the robot model. Run with:  pytest"""

import mujoco
import numpy as np
import pytest

from so101 import ALL_JOINTS, ARM_JOINTS, load_model, set_joint_angles, tip_pose


@pytest.fixture
def sim():
    model, data = load_model()
    set_joint_angles(model, data, [0.0] * 6)
    return model, data


def test_joint_names_and_types(sim):
    model, _ = sim
    for name in ALL_JOINTS:
        assert model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert len(ARM_JOINTS) == 5
    assert model.nu == 6  # 5 arm servos + gripper servo


def test_axis_geometry(sim):
    """Pan is vertical; lift/elbow/wrist_flex are parallel and horizontal; roll is perpendicular."""
    _, data = sim
    ax = {n: data.joint(n).xaxis.copy() for n in ALL_JOINTS}
    assert abs(abs(ax["shoulder_pan"] @ [0, 0, 1]) - 1) < 1e-6
    for n in ("elbow_flex", "wrist_flex"):
        assert abs(abs(ax[n] @ ax["shoulder_lift"]) - 1) < 1e-6
    assert abs(ax["shoulder_lift"] @ [0, 0, 1]) < 1e-6
    assert abs(ax["wrist_roll"] @ ax["wrist_flex"]) < 1e-6


def test_pitch_joints_keep_tip_in_plane(sim):
    model, data = sim
    rng = np.random.default_rng(1)
    ys = []
    for _ in range(50):
        set_joint_angles(model, data, [0.0, *rng.uniform(-1, 1, 3), 0.0, 0.0])
        ys.append(tip_pose(model, data)[0][1])
    assert np.ptp(ys) < 1e-6


def test_pan_preserves_height_and_radius(sim):
    """Radius is measured from the pan AXIS, which is offset from the world origin."""
    model, data = sim
    axis_xy = data.joint("shoulder_pan").xanchor[:2].copy()
    p0, _ = tip_pose(model, data)
    set_joint_angles(model, data, [np.radians(50), 0, 0, 0, 0, 0])
    p1, _ = tip_pose(model, data)
    assert p1[2] == pytest.approx(p0[2], abs=1e-9)
    r0 = np.linalg.norm(p0[:2] - axis_xy)
    r1 = np.linalg.norm(p1[:2] - axis_xy)
    assert r1 == pytest.approx(r0, abs=1e-9)


def test_arm_holds_pose_under_gravity(sim):
    """Position servos commanded to home should keep the arm near home for 1 s."""
    model, data = sim
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    q = np.array([data.qpos[model.joint(n).qposadr[0]] for n in ARM_JOINTS])
    assert np.max(np.abs(q)) < np.radians(5)
