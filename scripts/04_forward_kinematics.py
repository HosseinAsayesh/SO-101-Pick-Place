"""Lesson 02 - forward kinematics written from scratch, checked against MuJoCo.

Run:
    python scripts/04_forward_kinematics.py
"""

import time

import numpy as np

from so101 import ARM_JOINTS, load_model, set_joint_angles, tip_pose
from so101.kinematics import Chain, derive_dh, dh_fk


def fmt(v, nd=4):
    return "[" + ", ".join(f"{x:+.{nd}f}" for x in np.asarray(v).ravel()) + "]"


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    model, data = load_model()
    chain = Chain(model)

    # ------------------------------------------------ 1. the constants of the chain
    section("1. FIXED GEOMETRY: T_parent_child(q) = Trans(offset) @ Rot(quat) @ Rot_z(q)")
    for name, off, R in zip(chain.joints, chain.offsets, chain.rotations):
        print(f"{name:<15} offset = {fmt(off)}   R = {np.array2string(R, precision=3, suppress_small=True).replace(chr(10), '')}")
    print(f"{'tip site':<15} offset = {fmt(chain.tip_offset)}")

    # ------------------------------------------------------ 2. our FK vs MuJoCo's
    section("2. OUR FK vs MuJoCo (the independent check)")
    rng = np.random.default_rng(0)
    lo, hi = model.jnt_range[:len(ARM_JOINTS)].T
    pos_err, rot_err = [], []
    t_ours = t_mj = 0.0
    for _ in range(1000):
        q = rng.uniform(lo, hi)
        t0 = time.perf_counter(); T = chain.fk(q); t_ours += time.perf_counter() - t0
        t0 = time.perf_counter(); set_joint_angles(model, data, [*q, 0.0]); t_mj += time.perf_counter() - t0
        p_mj, R_mj = tip_pose(model, data)
        pos_err.append(np.linalg.norm(T[:3, 3] - p_mj))
        rot_err.append(np.abs(T[:3, :3] - R_mj).max())
    print(f"1000 random poses inside the joint limits:")
    print(f"  max tip position error : {max(pos_err):.2e} m  ({max(pos_err) * 1e9:.3f} nm)")
    print(f"  max rotation error     : {max(rot_err):.2e}")
    print(f"  time: ours {t_ours * 1e6 / 1000:.0f} us/pose, MuJoCo {t_mj * 1e6 / 1000:.0f} us/pose")

    # ------------------------------------------------------- 3. a worked example
    section("3. WORKED EXAMPLE: q = (30, -20, 40, -20, 45) deg")
    q = np.radians([30, -20, 40, -20, 45])
    for i, T in enumerate(chain.frames(q)):
        label = "base" if i == 0 else ("tip " if i == len(chain.joints) + 1 else chain.joints[i - 1])
        print(f"  frame after {label:<15} origin = {fmt(T[:3, 3])}  z axis = {fmt(T[:3, 2], 3)}")
    T = chain.fk(q)
    # the tip frame's x axis points out of the jaws: that is the "approach" direction
    print(f"  approach direction (tip frame x axis) = {fmt(T[:3, 0], 3)}")
    print("  for a top-down grasp we will need this to be [0, 0, -1] (lesson 03)")

    # --------------------------------------------------------- 4. the DH table
    section("4. DENAVIT-HARTENBERG TABLE (derived from the joint axes)")
    base, table, tool = derive_dh(chain)
    print(f"{'row':<5}{'joint':<15}{'theta offset [deg]':>20}{'d [m]':>10}{'a [m]':>10}{'alpha [deg]':>13}")
    for i, ((th, d, a, al), name) in enumerate(zip(table, chain.joints), start=1):
        print(f"{i:<5}{name:<15}{np.degrees(th):>20.3f}{d:>10.5f}{a:>10.5f}{np.degrees(al):>13.3f}")
    print(f"\nbase frame origin = {fmt(base[:3, 3])}, z = {fmt(base[:3, 2], 3)} (pointing down)")
    print(f"tool offset from the last DH frame = {fmt(tool[:3, 3])}")
    err = max(np.abs(dh_fk(table, q_, base=base, tool=tool) - chain.fk(q_)).max()
              for q_ in rng.uniform(lo, hi, size=(300, len(ARM_JOINTS))))
    print(f"max |DH - our FK| over 300 random poses: {err:.2e}  (they are the same robot)")


if __name__ == "__main__":
    main()
