# SO-101 model files

Source: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), folder
`Simulation/SO101`, commit `eecbe3e` (2026-09-06). License: Apache-2.0 (`LICENSE-SO-ARM100`).

| File | Status |
|---|---|
| `so101_new_calib.xml` | Unmodified MuJoCo model ("new calibration": 0 rad = middle of each joint range) |
| `so101_new_calib.urdf` | Unmodified URDF, same robot, for ROS 2 later |
| `assets/*.stl` | The 13 meshes referenced by the model (unused upstream files not copied) |
| `so101_pick_place.xml` | **Derived** — upstream model + grasp pads, written by `tools/derive_grasp_model.py`. Do not edit by hand |
| `scene.xml` | **Ours** — includes the derived model; table, lights, cube, `overview` and `top_cam` cameras |

Why the derived file exists: MuJoCo collides mesh geoms as convex hulls, and the fixed finger's
mesh wraps around the jaw hinge, so its hull fills the space between the jaws — nothing can be
gripped. The derived model keeps the meshes for rendering, switches their collisions off on the two
gripper bodies, and adds one box pad per finger. See docs/lessons/04-trajectories-and-control.md.

Upstream notes: generated from Onshape with onshape-to-robot; base collision meshes removed
upstream; servo parameters for the Feetech STS3215 adapted from the Open Duck Mini project.
