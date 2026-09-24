"""Lesson 01 - open the SO-101 in MuJoCo's interactive viewer.

Run from the repo root (with the .venv active):
    python scripts/01_view_robot.py

In the viewer:
  - Right panel -> "Control": one slider per actuator (shoulder_pan ... gripper).
    Each slider is a target angle for that joint's position servo.
  - Left-drag = rotate view, right-drag = pan, scroll = zoom.
  - Space = pause/resume physics. Double-click a part to select it.
  - Left panel -> "Rendering" -> Frame: "Body" shows each link's coordinate frame;
    Left panel -> "Rendering" -> "Joint" draws the joint axes. Turn both on.

Things to try (answers in the lesson doc):
  1. Move only shoulder_lift, elbow_flex and wrist_flex. Which plane does the tip stay in?
  2. Now move shoulder_pan. What happens to that plane?
  3. Move only wrist_roll. Does the tip move? Does the gripper's orientation change?
"""

import mujoco
import mujoco.viewer

from so101 import load_model


def main() -> None:
    model, data = load_model()
    print(f"Loaded {model.nbody - 1} bodies, {model.njnt} joints, {model.nu} actuators.")
    print("Opening viewer... (close the window to exit)")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
