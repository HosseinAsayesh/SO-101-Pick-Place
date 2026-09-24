"""Lesson 01 - read links, joints and DOF straight out of the model, then run 3 small
forward-kinematics experiments that prove the claims in the lesson.

Run:
    python scripts/02_inspect_joints.py
"""

import mujoco
import numpy as np

from so101 import ALL_JOINTS, ARM_JOINTS, load_model, set_joint_angles, tip_pose

JOINT_TYPE = {0: "free (6 DOF)", 1: "ball (3 DOF)", 2: "slide/prismatic", 3: "hinge/revolute"}


def fmt(v, nd=3):
    return "[" + ", ".join(f"{x:+.{nd}f}" for x in v) + "]"


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    model, data = load_model()
    set_joint_angles(model, data, [0.0] * 6)  # "home": every joint at the middle of its range

    # ------------------------------------------------------------------ 1. links
    section("1. LINKS (MuJoCo calls them bodies) - the kinematic tree")
    for b in range(1, model.nbody):  # body 0 is the fixed 'world'
        depth, p = 0, b
        while model.body_parentid[p] != 0:
            p = model.body_parentid[p]
            depth += 1
        mass = model.body_mass[b]
        print(f"{'  ' * depth}- {model.body(b).name:<22} mass = {mass * 1000:6.1f} g")

    # ----------------------------------------------------------------- 2. joints
    section("2. JOINTS - what connects each link to its parent (home pose, world frame)")
    print(f"{'#':<3}{'joint':<15}{'type':<17}{'moves body':<22}{'axis (world)':<26}{'range [deg]'}")
    for i, name in enumerate(ALL_JOINTS, start=1):
        j = model.joint(name)
        child = model.body(j.bodyid[0]).name
        lo, hi = np.degrees(j.range)
        axis = data.joint(name).xaxis
        tag = "" if name in ARM_JOINTS else "   <- gripper (not arm DOF)"
        print(f"{i:<3}{name:<15}{JOINT_TYPE[j.type[0]]:<17}{child:<22}{fmt(axis, 2):<26}"
              f"{lo:+6.1f} .. {hi:+6.1f}{tag}")

    # ------------------------------------------------------------------- 3. DOF
    section("3. DEGREES OF FREEDOM")
    cube_dof = model.nv - len(ALL_JOINTS)
    print(f"Total DOF in the scene (model.nv) = {model.nv}")
    print(f"  arm joints    : {len(ARM_JOINTS)}  (one per hinge)")
    print(f"  gripper jaw   : 1")
    print(f"  cube free-joint: {cube_dof}  (3 translation + 3 rotation)")
    print("Note: model.nq = {} because a free joint stores orientation as a 4-number "
          "quaternion (7 qpos, 6 DOF).".format(model.nq))

    # ------------------------------------------------------- 4. segment lengths
    section("4. DISTANCES BETWEEN JOINT AXES (home pose) - the 'link lengths'")
    anchors = {n: data.joint(n).xanchor.copy() for n in ALL_JOINTS}
    chain = list(ARM_JOINTS)
    for a, b in zip(chain[:-1], chain[1:]):
        print(f"  {a:>13} -> {b:<13}: {np.linalg.norm(anchors[b] - anchors[a]) * 100:5.1f} cm")
    p_tip, _ = tip_pose(model, data)
    print(f"  {'wrist_flex':>13} -> {'tip':<13}: "
          f"{np.linalg.norm(p_tip - anchors['wrist_flex']) * 100:5.1f} cm")
    print(f"Tip position at home: {fmt(p_tip)} m")

    # -------------------------------------------------------- 5. experiments
    section("5. FORWARD-KINEMATICS EXPERIMENTS")
    rng = np.random.default_rng(0)

    # A) pitch joints only -> tip stays in one vertical plane (y constant)
    ys = []
    for _ in range(200):
        q = [0.0, *rng.uniform(-1.0, 1.0, 3), 0.0, 0.0]
        set_joint_angles(model, data, q)
        ys.append(tip_pose(model, data)[0][1])
    print(f"A) 200 random poses of shoulder_lift/elbow_flex/wrist_flex (pan = 0):\n"
          f"   tip y ranges {min(ys) * 1000:+.2f} .. {max(ys) * 1000:+.2f} mm "
          f"-> joints 2,3,4 act like a planar 3-link arm.")

    # B) pan only -> tip circles the pan axis: height and radius constant, azimuth changes
    #    Careful: the pan axis sits at x = +3.9 cm, NOT at the world origin. Measure from it.
    #    Also: the axis points DOWN (-z), so positive pan turns the arm clockwise seen from above.
    print("B) shoulder_pan only (radius/azimuth measured about the pan axis):")
    c = data.joint("shoulder_pan").xanchor[:2].copy()
    for deg in (0, 30, 60):
        set_joint_angles(model, data, [np.radians(deg), 0, 0, 0, 0, 0])
        p, _ = tip_pose(model, data)
        dx, dy = p[0] - c[0], p[1] - c[1]
        print(f"   pan {deg:>3} deg: tip = {fmt(p)}  height = {p[2] * 100:.1f} cm, "
              f"radius = {np.hypot(dx, dy) * 100:.1f} cm, "
              f"azimuth = {np.degrees(np.arctan2(dy, dx)):+.1f} deg")

    # C) wrist_roll only -> orientation changes, position (almost) doesn't
    set_joint_angles(model, data, [0, 0, 0, 0, 0, 0])
    p0, R0 = tip_pose(model, data)
    print("C) wrist_roll only:")
    for deg in (0, 45, 90):
        set_joint_angles(model, data, [0, 0, 0, 0, np.radians(deg), 0])
        p, R = tip_pose(model, data)
        dR = R0.T @ R
        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        print(f"   roll {deg:>3} deg: tip moved {np.linalg.norm(p - p0) * 1000:5.1f} mm, "
              f"gripper frame rotated {angle:5.1f} deg")
    print("   (the tip moves a little because the 'gripperframe' site is not exactly on the "
          "roll axis)")


if __name__ == "__main__":
    main()
