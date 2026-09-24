"""Loading the SO-101 scene and reading/writing joint angles.

Vocabulary (see docs/lessons/01-links-joints-dof.md):
  - body  = link  (a rigid part)
  - joint = the connection that lets a body move relative to its parent
  - qpos  = generalized positions (joint angles for hinges; 7 numbers for a free joint)
  - ctrl  = actuator commands (for SO-101: target angle of each position servo)
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = REPO_ROOT / "models" / "so101" / "scene.xml"

# The 5 arm joints, base -> tip. These are the arm's 5 DOF.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",   # 1. base yaw, vertical axis
    "shoulder_lift",  # 2. pitch
    "elbow_flex",     # 3. pitch (parallel to 2)
    "wrist_flex",     # 4. pitch (parallel to 2 and 3)
    "wrist_roll",     # 5. roll about the forearm/gripper axis
)
GRIPPER_JOINT = "gripper"  # moving jaw hinge, not counted as arm DOF
ALL_JOINTS: tuple[str, ...] = ARM_JOINTS + (GRIPPER_JOINT,)

# A reference frame on the gripper, between the jaws. We call it the "tip".
TIP_SITE = "gripperframe"


def load_model(xml_path: str | Path = SCENE_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the scene and run forward kinematics once so positions are valid."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def set_joint_angles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    angles: Mapping[str, float] | Sequence[float],
    *,
    also_set_ctrl: bool = True,
) -> None:
    """Teleport joints to the given angles [rad] and recompute kinematics.

    `angles` is either {joint_name: angle} or a sequence in ALL_JOINTS order
    (a 5-long sequence sets only the arm joints).
    If also_set_ctrl, the position servos are commanded to the same angles so the
    arm holds the pose when the simulation steps.
    """
    if not isinstance(angles, Mapping):
        angles = dict(zip(ALL_JOINTS, angles))
    for name, value in angles.items():
        data.qpos[model.joint(name).qposadr[0]] = value
        if also_set_ctrl:
            data.ctrl[model.actuator(name).id] = value
    mujoco.mj_forward(model, data)


def get_joint_angles(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """Current joint angles [rad] as {name: angle}."""
    return {name: float(data.qpos[model.joint(name).qposadr[0]]) for name in ALL_JOINTS}


def tip_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """World position (3,) [m] and rotation matrix (3, 3) of the gripper tip frame."""
    site = data.site(TIP_SITE)
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()
