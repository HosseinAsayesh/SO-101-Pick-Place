"""Forward kinematics, written from scratch (lesson 02).

Everything here is built from three ideas:
  1. A rotation is a 3x3 orthonormal matrix R.
  2. A pose (rotation + translation) is a 4x4 homogeneous transform T.
  3. The pose of link i in the world = the product of the transforms along the chain.

Nothing in this file calls MuJoCo's kinematics; `scripts/04_forward_kinematics.py` and
`tests/test_kinematics.py` compare our numbers against MuJoCo's as an independent check.

Convention used everywhere: T maps a point expressed in the CHILD frame to the PARENT frame,
    p_parent = T @ p_child        (points written as [x, y, z, 1])
"""

from __future__ import annotations

import numpy as np

from so101.model import ARM_JOINTS

# --------------------------------------------------------------------------- basics


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by `angle` [rad] about a unit `axis` through the origin (Rodrigues' formula).

    R = I + sin(t) K + (1 - cos(t)) K^2,  where K = skew(axis).
    rot_x/rot_y/rot_z are the special cases axis = e_x / e_y / e_z.
    """
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> rotation matrix. MuJoCo stores orientations this way.

    A quaternion encodes 'rotate by angle t about unit axis u' as
        q = (cos(t/2), sin(t/2) * u),
    which is why the formula below is full of squared terms: the half-angle doubles.
    """
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def make_T(R: np.ndarray = np.eye(3), p: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Build the 4x4 homogeneous transform [[R, p], [0, 1]]."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform: [[R^T, -R^T p], [0, 1]]. No matrix inverse needed."""
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


# ------------------------------------------------------- the SO-101's fixed geometry

class Chain:
    """The constant part of the robot's geometry, read once from the model file.

    For the SO-101 every joint sits at its body's origin and rotates about that body's local
    z axis, so each step of the chain is:

        T_parent_child(q) = Trans(offset) @ Rot(quat) @ Rot_z(q)
                            \\________ fixed, from the model ______/   \\__ the joint __/
    """

    def __init__(self, model, joints=ARM_JOINTS, tip_site: str = "gripperframe"):
        self.joints = tuple(joints)
        self.offsets, self.rotations = [], []
        for name in self.joints:
            body_id = int(model.joint(name).bodyid[0])
            assert np.allclose(model.jnt_pos[model.joint(name).id], 0), "joint not at body origin"
            assert np.allclose(model.jnt_axis[model.joint(name).id], [0, 0, 1]), "axis not local z"
            self.offsets.append(model.body_pos[body_id].copy())
            self.rotations.append(quat_to_R(model.body_quat[body_id]))
        site = model.site(tip_site)
        self.tip_offset = site.pos.copy()
        self.tip_rotation = quat_to_R(site.quat)

    def joint_transform(self, i: int, q: float) -> np.ndarray:
        """Transform from link i-1's frame to link i's frame at joint angle q [rad]."""
        return make_T(self.rotations[i], self.offsets[i]) @ make_T(rot_z(q))

    def frames(self, q) -> list[np.ndarray]:
        """World pose of every link frame, base first, tip frame last."""
        q = np.asarray(q, dtype=float).ravel()[: len(self.joints)]
        T = np.eye(4)  # base is bolted to the world at the origin
        out = [T]
        for i, qi in enumerate(q):
            T = T @ self.joint_transform(i, float(qi))
            out.append(T)
        out.append(T @ make_T(self.tip_rotation, self.tip_offset))
        return out

    def fk(self, q) -> np.ndarray:
        """World pose (4x4) of the gripper tip for joint angles q [rad]."""
        return self.frames(q)[-1]


# ------------------------------------------------------------------ DH parameters

def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """One row of a classic (Denavit-Hartenberg) table, as a 4x4 transform.

        T = Rot_z(theta) @ Trans_z(d) @ Trans_x(a) @ Rot_x(alpha)

    theta: joint angle (the variable, for a revolute joint)
    d    : offset along the previous z axis
    a    : link length, along the common normal x
    alpha: twist, the angle between consecutive z axes, measured about x
    """
    return make_T(rot_z(theta)) @ make_T(p=[0, 0, d]) @ make_T(p=[a, 0, 0]) @ make_T(rot_x(alpha))


def dh_fk(table, q, base: np.ndarray | None = None, tool: np.ndarray | None = None) -> np.ndarray:
    """Forward kinematics from a DH table: rows of (theta_offset, d, a, alpha)."""
    T = np.eye(4) if base is None else base.copy()
    for (theta0, d, a, alpha), qi in zip(table, np.asarray(q, dtype=float).ravel()):
        T = T @ dh_transform(theta0 + qi, d, a, alpha)
    return T if tool is None else T @ tool


def common_normal_feet(z1, p1, z2, p2):
    """The two points where the common normal of two lines meets them.

    Line i = {p_i + t z_i}. The common normal is the unique segment perpendicular to both;
    its length is the distance between the lines (0 if they intersect). For parallel lines the
    normal is not unique, so we return the foot on line 1 closest to p2.
    """
    z1, p1, z2, p2 = map(np.asarray, (z1, p1, z2, p2))
    w0 = p1 - p2
    a, b, c = z1 @ z1, z1 @ z2, z2 @ z2
    dd, e = z1 @ w0, z2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:  # parallel axes
        return p1 + (z1 @ (p2 - p1)) * z1, p2
    s, t = (b * e - c * dd) / den, (a * e - b * dd) / den
    return p1 + s * z1, p2 + t * z2


def _frame(origin, x, z) -> np.ndarray:
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float)
    x = x - (x @ z) * z
    return make_T(np.column_stack([x / np.linalg.norm(x), np.cross(z, x / np.linalg.norm(x)), z]),
                  np.asarray(origin, float))


def T_to_dh(T: np.ndarray) -> tuple[float, float, float, float, float]:
    """Decompose a transform into (theta, d, a, alpha) + residual.

    Rot_z(t) Trans_z(d) Trans_x(a) Rot_x(al) has rotation Rot_z(t) Rot_x(al) and translation
    Rot_z(t) @ (a, 0, d), so the parameters can be read off directly. The residual is how far
    the transform is from being representable at all (it must be ~0 for proper DH frames).
    """
    R, p = T[:3, :3], T[:3, 3]
    theta = np.arctan2(R[1, 0], R[0, 0])
    alpha = np.arctan2(R[2, 1], R[2, 2])
    a = p[0] * np.cos(theta) + p[1] * np.sin(theta)
    d = p[2]
    return theta, d, a, alpha, float(np.abs(dh_transform(theta, d, a, alpha) - T).max())


def derive_dh(chain: Chain):
    """Build an exact classic-DH description of `chain`: (base, table, tool).

    DH frame rules: z_i is joint (i+1)'s axis; x_i is the common normal from z_{i-1} to z_i;
    the origin is where that normal meets z_i. Then each link transform needs only 4 numbers.
    Reconstruct poses with: dh_fk(table, q, base=base, tool=tool).
    """
    n = len(chain.joints)
    F0 = chain.frames(np.zeros(n))
    axes = [(T[:3, 2].copy(), T[:3, 3].copy()) for T in F0[1:n + 1]]  # (direction, point on axis)
    tip = F0[-1]

    frames = [None] * (n + 1)
    for i in range(1, n):
        (zp, pp), (z, p) = axes[i - 1], axes[i]
        f_prev, f_here = common_normal_feet(zp, pp, z, p)
        x = f_here - f_prev
        if np.linalg.norm(x) < 1e-9:  # axes intersect: normal direction is z_{i-1} x z_i
            x = np.cross(zp, z)
        frames[i] = _frame(f_here, x, z)

    z0, p0 = axes[0]                      # base frame: free choice, aligned with frame 1's x
    foot, _ = common_normal_feet(z0, p0, *axes[1])
    frames[0] = _frame(foot, frames[1][:3, 0], z0)

    zn, pn = axes[-1]                     # last frame: origin = tip projected on the last axis
    origin = pn + (zn @ (tip[:3, 3] - pn)) * zn
    x = tip[:3, 3] - origin
    frames[n] = _frame(origin, x if np.linalg.norm(x) > 1e-9 else frames[n - 1][:3, 0], zn)

    table = []
    for i in range(1, n + 1):
        theta, d, a, alpha, residual = T_to_dh(invert_T(frames[i - 1]) @ frames[i])
        assert residual < 1e-9, f"row {i} is not a valid DH frame pair (residual {residual:.1e})"
        table.append((theta, d, a, alpha))
    return frames[0], table, invert_T(frames[n]) @ tip
