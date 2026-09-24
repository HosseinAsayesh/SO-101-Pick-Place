"""A scripted pick-and-place, built from IK poses and trajectory segments (lesson 04).

The motion is a little state machine of segments:

    home -> above the cube -> down onto it -> close -> lift
         -> above the target -> down -> open -> retreat -> home

Positions are given for the GRASP POINT (the centre of the jaw opening), not the model's tip
site; `TopDownIK.grasp` applies the offset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from so101.ik import GRIPPER_CLOSED, GRIPPER_OPEN, TopDownIK
from so101.kinematics import Chain
from so101.model import ALL_JOINTS, ARM_JOINTS, set_joint_angles, tip_pose
from so101.trajectory import Trajectory, cartesian_segment, hold_segment, joint_segment


def pad_gap(model, data, angle: float) -> float:
    """Distance between the two grasp pads [m] at gripper joint `angle` [rad].

    Measured from the pad boxes' corner positions, so it stays correct if the pads are moved.
    """
    import mujoco as _mj
    qpos_adr = model.joint("gripper").qposadr[0]
    saved = float(data.qpos[qpos_adr])
    data.qpos[qpos_adr] = angle
    _mj.mj_forward(model, data)
    faces = []
    for name, sign in (("pad_fixed", +1), ("pad_moving", -1)):
        gid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_GEOM, name)
        half = model.geom_size[gid]
        corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        world = (data.geom_xmat[gid].reshape(3, 3) @ corners.T).T + data.geom_xpos[gid]
        R_tip = data.site("gripperframe").xmat.reshape(3, 3)
        local_z = (world - data.site("gripperframe").xpos) @ R_tip[:, 2]
        faces.append(local_z.max() if sign > 0 else local_z.min())
    data.qpos[qpos_adr] = saved
    _mj.mj_forward(model, data)
    return float(faces[1] - faces[0])


def jaw_angle_for_gap(model, data, gap: float) -> float:
    """The gripper command [rad] that opens the pads to `gap` metres (clamped to the limits).

    A real gripper would close under force control; with a position servo we have to know how
    wide to go. Commanding 'fully closed' on a 3 cm cube asks the solver for 1.3 cm of
    penetration, and the cube squirts out of the jaws - that was the first bug in lesson 04.
    """
    lo, hi = model.jnt_range[model.joint("gripper").id]
    angles = np.linspace(lo, hi, 60)
    gaps = np.array([pad_gap(model, data, a) for a in angles])
    order = np.argsort(gaps)
    return float(np.clip(np.interp(gap, gaps[order], angles[order]), lo, hi))


@dataclass
class PickPlacePlan:
    trajectory: Trajectory
    grasp_q: np.ndarray
    place_q: np.ndarray


@dataclass
class RunLog:
    """Everything the run recorded, for plots and for judging success."""

    t: list = field(default_factory=list)
    q_cmd: list = field(default_factory=list)
    q_act: list = field(default_factory=list)
    tip: list = field(default_factory=list)
    cube: list = field(default_factory=list)
    phase: list = field(default_factory=list)

    def arrays(self):
        return (np.array(self.t), np.array(self.q_cmd), np.array(self.q_act),
                np.array(self.tip), np.array(self.cube), self.phase)


def plan_pick_place(model, chain: Chain, ik: TopDownIK, pick_xy, place_xy, *,
                    cube_half: float = 0.015, pick_yaw: float = 0.0, place_yaw: float = 0.0,
                    hover: float = 0.055, speed: float = 1.0, data=None,
                    squeeze: float = 0.0015, clearance: float = 0.012) -> PickPlacePlan:
    """Build the whole motion. Heights are for the grasp point, measured from the table."""
    if data is not None:   # size the gripper commands to the object
        width = 2 * cube_half
        grip_closed = jaw_angle_for_gap(model, data, width - squeeze)
        grip_open = jaw_angle_for_gap(model, data, width + clearance)
    else:
        grip_closed, grip_open = GRIPPER_CLOSED, GRIPPER_OPEN
    # The fixed pad sticks out 6 mm below the grasp centre (GRASP_OFFSET z = 14 mm, pad
    # half-thickness 6 mm), so grasping at the cube's mid-height would drive the pad into the
    # table: the arm stalls 1.5 cm high and the grasp misses. Grasp slightly higher up the cube.
    pad_below = 0.020
    grasp_z = max(cube_half, pad_below)
    up_z = grasp_z + hover                    # stays under the top-down ceiling from lesson 03

    pick = np.array([pick_xy[0], pick_xy[1], grasp_z])
    pick_up = np.array([pick_xy[0], pick_xy[1], up_z])
    place = np.array([place_xy[0], place_xy[1], grasp_z])
    place_up = np.array([place_xy[0], place_xy[1], up_z])

    for point, psi in ((pick, pick_yaw), (pick_up, pick_yaw),
                       (place, place_yaw), (place_up, place_yaw)):
        if not ik.can_grasp(point, psi):
            raise ValueError(f"no IK solution for {point} at yaw {np.degrees(psi):.0f} deg")

    q_home = np.zeros(len(ARM_JOINTS))
    q_pick_up = ik.grasp(pick_up, pick_yaw).q
    q_place_up = ik.grasp(place_up, place_yaw).q

    t = lambda seconds: seconds / speed  # noqa: E731

    segs = [
        joint_segment(q_home, q_pick_up, t(1.6), gripper=grip_open, label="approach"),
        cartesian_segment(ik, pick_up, pick, pick_yaw, t(1.0), q_seed=q_pick_up,
                          gripper=grip_open, label="descend"),
    ]
    q_grasp = segs[-1].q(segs[-1].duration)
    segs += [
        hold_segment(q_grasp, t(0.7), gripper=grip_closed, label="close"),
        cartesian_segment(ik, pick, pick_up, pick_yaw, t(1.0), q_seed=q_grasp,
                          gripper=grip_closed, label="lift"),
        joint_segment(q_pick_up, q_place_up, t(2.0), gripper=grip_closed, label="transfer"),
        cartesian_segment(ik, place_up, place, place_yaw, t(1.0), q_seed=q_place_up,
                          gripper=grip_closed, label="lower"),
    ]
    q_place = segs[-1].q(segs[-1].duration)
    segs += [
        hold_segment(q_place, t(0.6), gripper=grip_open, label="release"),
        cartesian_segment(ik, place, place_up, place_yaw, t(1.0), q_seed=q_place,
                          gripper=grip_open, label="retreat"),
        joint_segment(q_place_up, q_home, t(1.6), gripper=grip_open, label="home"),
    ]
    return PickPlacePlan(Trajectory(segs), q_grasp, q_place)


def run_plan(model, data, chain: Chain, plan: PickPlacePlan, *, log_every: int = 10,
             reset: bool = True) -> RunLog:
    """Play the trajectory on the position servos and record what happens.

    reset=False keeps whatever is already in `data` (e.g. a cube you just placed yourself).
    """
    if reset:
        mujoco.mj_resetData(model, data)
    q0, grip0, _ = plan.trajectory.at(0.0)
    set_joint_angles(model, data, [*q0, grip0 if grip0 is not None else GRIPPER_OPEN])

    log = RunLog()
    steps = int(plan.trajectory.duration / model.opt.timestep)
    grip = GRIPPER_OPEN
    for i in range(steps):
        t = i * model.opt.timestep
        q_des, grip_cmd, phase = plan.trajectory.at(t)
        if grip_cmd is not None:
            grip = grip_cmd
        data.ctrl[:len(ARM_JOINTS)] = q_des
        data.ctrl[len(ARM_JOINTS)] = grip
        mujoco.mj_step(model, data)
        if i % log_every == 0:
            log.t.append(t)
            log.q_cmd.append(np.array(q_des))
            log.q_act.append(np.array([data.qpos[model.joint(n).qposadr[0]] for n in ARM_JOINTS]))
            log.tip.append(tip_pose(model, data)[0])
            log.cube.append(data.body("cube").xpos.copy())
            log.phase.append(phase)
    return log


def run_plan_live(model, data, chain: Chain, plan: PickPlacePlan, *, speed: float = 1.0,
                  log_every: int = 10, reset: bool = True) -> RunLog:
    """Same as run_plan, but opens MuJoCo's viewer and runs at (roughly) wall-clock speed.

    The window is passive: the physics is driven by this loop, so you watch rather than interact.
    Close the window to stop early. `speed` > 1 plays faster than real time.
    """
    import mujoco.viewer

    if reset:
        mujoco.mj_resetData(model, data)
    q0, grip0, _ = plan.trajectory.at(0.0)
    set_joint_angles(model, data, [*q0, grip0 if grip0 is not None else GRIPPER_OPEN])

    log = RunLog()
    grip = GRIPPER_OPEN
    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        i = 0
        steps = int(plan.trajectory.duration / model.opt.timestep)
        while viewer.is_running() and i < steps:
            t = i * model.opt.timestep
            q_des, grip_cmd, phase = plan.trajectory.at(t)
            if grip_cmd is not None:
                grip = grip_cmd
            data.ctrl[:len(ARM_JOINTS)] = q_des
            data.ctrl[len(ARM_JOINTS)] = grip
            mujoco.mj_step(model, data)
            if i % log_every == 0:
                log.t.append(t)
                log.q_cmd.append(np.array(q_des))
                log.q_act.append(np.array([data.qpos[model.joint(n).qposadr[0]]
                                           for n in ARM_JOINTS]))
                log.tip.append(tip_pose(model, data)[0])
                log.cube.append(data.body("cube").xpos.copy())
                log.phase.append(phase)
            if i % 5 == 0:
                viewer.sync()
            # pace the loop to wall-clock time
            ahead = t / speed - (time.perf_counter() - wall_start)
            if ahead > 0:
                time.sleep(ahead)
            i += 1
    return log


def judge(log: RunLog, place_xy, *, lift_threshold: float = 0.03, zone: float = 0.04,
          cube_half: float = 0.015) -> dict:
    """Did it work? The success rule from docs/PROJECT_BRIEF.md."""
    _, _, _, _, cube, _ = log.arrays()
    lifted = float(cube[:, 2].max() - cube[0, 2]) >= lift_threshold
    final = cube[-1]
    placed = bool(abs(final[0] - place_xy[0]) <= zone / 2 and abs(final[1] - place_xy[1]) <= zone / 2)
    upright = abs(final[2] - cube_half) < 0.01
    return {
        "success": bool(lifted and placed and upright),
        "lifted": bool(lifted),
        "max_lift_cm": float((cube[:, 2].max() - cube[0, 2]) * 100),
        "placed": placed,
        "resting": bool(upright),
        "final_xy_error_mm": float(np.hypot(final[0] - place_xy[0], final[1] - place_xy[1]) * 1000),
    }


def set_cube(model, data, x: float, y: float, yaw: float = 0.0, half: float = 0.015) -> None:
    """Put the cube on the table at (x, y) turned by `yaw`, at rest."""
    import mujoco as _mj
    adr = model.jnt_qposadr[model.joint("cube_free").id]
    data.qpos[adr:adr + 3] = [x, y, half]
    data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    vadr = model.jnt_dofadr[model.joint("cube_free").id]
    data.qvel[vadr:vadr + 6] = 0.0
    _mj.mj_forward(model, data)


def run_trial(model, data, chain: Chain, ik: TopDownIK, pick_xy, place_xy, *, yaw: float = 0.0,
              **plan_kwargs) -> dict:
    """Spawn the cube, plan, run, and judge. Returns the verdict plus the failure reason."""
    import mujoco as _mj
    _mj.mj_resetData(model, data)
    set_cube(model, data, pick_xy[0], pick_xy[1], yaw)
    try:
        plan = plan_pick_place(model, chain, ik, pick_xy, place_xy, pick_yaw=yaw,
                               data=data, **plan_kwargs)
    except ValueError as exc:
        return {"success": False, "reason": "no IK solution", "detail": str(exc)}
    cube_start = data.body("cube").xpos.copy()
    log = run_plan(model, data, chain, plan, reset=False)
    verdict = judge(log, place_xy)
    verdict["reason"] = ("ok" if verdict["success"]
                         else "grasp failed" if not verdict["lifted"]
                         else "placed off target" if not verdict["placed"]
                         else "not resting")
    verdict["cube_start"] = cube_start
    verdict["log"] = log
    return verdict


def tracking_error(log: RunLog) -> np.ndarray:
    """Commanded minus actual joint angles [rad], per logged sample."""
    _, q_cmd, q_act, _, _, _ = log.arrays()
    return q_cmd - q_act
