"""Lesson 02 checks: our own kinematics must agree with MuJoCo. Run with:  pytest"""

import mujoco
import numpy as np
import pytest

from so101 import ARM_JOINTS, load_model, set_joint_angles, tip_pose
from so101.kinematics import (
    Chain,
    T_to_dh,
    common_normal_feet,
    derive_dh,
    dh_fk,
    dh_transform,
    invert_T,
    make_T,
    quat_to_R,
    rodrigues,
    rot_x,
    rot_y,
    rot_z,
)


@pytest.fixture(scope="module")
def sim():
    model, data = load_model()
    return model, data, Chain(model)


def random_qs(model, n, seed=0):
    lo, hi = model.jnt_range[: len(ARM_JOINTS)].T
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, len(ARM_JOINTS)))


# ------------------------------------------------------------------ building blocks

def test_rotations_are_orthonormal():
    for R in (rot_x(0.3), rot_y(-1.1), rot_z(2.0), rodrigues([1, 2, 3], 0.7)):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_rodrigues_reduces_to_elementary_rotations():
    for axis, fn in (([1, 0, 0], rot_x), ([0, 1, 0], rot_y), ([0, 0, 1], rot_z)):
        assert np.allclose(rodrigues(axis, 0.83), fn(0.83), atol=1e-12)


def test_quat_to_R_matches_mujoco():
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        expected = np.zeros(9)
        mujoco.mju_quat2Mat(expected, q)
        assert np.allclose(quat_to_R(q), expected.reshape(3, 3), atol=1e-12)


def test_invert_T_is_the_inverse():
    T = make_T(rodrigues([0.2, -1, 0.5], 1.2), [0.1, -0.2, 0.3])
    assert np.allclose(invert_T(T) @ T, np.eye(4), atol=1e-12)


# --------------------------------------------------------------------------- FK

def test_fk_matches_mujoco(sim):
    model, data, chain = sim
    for q in random_qs(model, 200, seed=1):
        set_joint_angles(model, data, [*q, 0.0])
        p_mj, R_mj = tip_pose(model, data)
        T = chain.fk(q)
        assert np.linalg.norm(T[:3, 3] - p_mj) < 1e-9   # under a nanometre
        assert np.abs(T[:3, :3] - R_mj).max() < 1e-9


def test_every_frame_matches_mujoco(sim):
    """Not just the tip: each link frame must land on MuJoCo's body frame."""
    model, data, chain = sim
    q = np.radians([25, -35, 45, -30, 60])
    set_joint_angles(model, data, [*q, 0.0])
    frames = chain.frames(q)
    for name, T in zip(("shoulder", "upper_arm", "lower_arm", "wrist", "gripper"), frames[1:]):
        body = data.body(name)
        assert np.allclose(T[:3, 3], body.xpos, atol=1e-9)
        assert np.allclose(T[:3, :3], body.xmat.reshape(3, 3), atol=1e-9)


def test_fk_is_consistent_with_joint_axes(sim):
    """Rotating joint i must rotate exactly the frames after it, and none before it."""
    model, _, chain = sim
    q = np.zeros(5)
    base_frames = chain.frames(q)
    q2 = q.copy()
    q2[2] += 0.4
    moved = chain.frames(q2)
    for i in range(3):  # frames 0..2 are before elbow_flex
        assert np.allclose(base_frames[i], moved[i], atol=1e-12)
    assert not np.allclose(base_frames[3], moved[3], atol=1e-6)


# --------------------------------------------------------------------------- DH

def test_common_normal_feet():
    # two perpendicular skew lines 2 apart
    f1, f2 = common_normal_feet([0, 0, 1], [0, 0, 0], [0, 1, 0], [2, 0, 5])
    assert np.allclose(f1, [0, 0, 5])
    assert np.allclose(f2, [2, 0, 5])


def test_T_to_dh_roundtrip():
    theta, d, a, alpha = 0.3, -0.12, 0.25, np.pi / 2
    out = T_to_dh(dh_transform(theta, d, a, alpha))
    assert out[:4] == pytest.approx((theta, d, a, alpha))
    assert out[4] < 1e-12


def test_dh_table_reproduces_fk(sim):
    model, _, chain = sim
    base, table, tool = derive_dh(chain)
    assert len(table) == len(ARM_JOINTS)
    for q in random_qs(model, 100, seed=5):
        assert np.abs(dh_fk(table, q, base=base, tool=tool) - chain.fk(q)).max() < 1e-9
