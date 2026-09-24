# SO-101 model files

Source: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), folder
`Simulation/SO101`, commit `eecbe3e` (2026-09-06). License: Apache-2.0 (`LICENSE-SO-ARM100`).

| File | Status |
|---|---|
| `so101_new_calib.xml` | Unmodified MuJoCo model ("new calibration": 0 rad = middle of each joint range) |
| `so101_new_calib.urdf` | Unmodified URDF, same robot, for ROS 2 later |
| `assets/*.stl` | The 13 meshes referenced by the model (unused upstream files not copied) |
| `scene.xml` | **Ours** — table, lights, cube, `overview` and `top_cam` cameras |

Upstream notes: generated from Onshape with onshape-to-robot; base collision meshes removed
upstream; servo parameters for the Feetech STS3215 adapted from the Open Duck Mini project.
