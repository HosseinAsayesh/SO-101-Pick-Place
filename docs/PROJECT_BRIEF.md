# Project Brief — SO-101 Vision Pick-and-Place (simulation)

*Started 2026-09-17. This file holds the decisions; the reasoning behind day-to-day changes lives in
`docs/notebook/`.*

## 1. Goal
Build a simulated robot arm that **finds a small object on a table with a camera, picks it up, and
places it at a target**. Then measure how often it succeeds and explain the failures.

Why this project:
- It covers the core robotics stack end-to-end: kinematics, motion planning, control, perception.
- It produces a single number (success rate) plus failure analysis, which works well in a
  portfolio or research application.
- Its structure (perception → planning → control) maps directly onto ROS 2 nodes for Phase 6.

## 2. Decisions so far

| Topic | Decision | Why |
|---|---|---|
| Hardware | **None — simulation only** | Cost and availability; everything transfers later |
| Robot | **SO-101** (5 DOF + gripper) | Open-source, simulator model ready, 5 DOF is exactly enough for top-down grasps (lesson 01) |
| Simulator | **MuJoCo** (Python, native Windows) | Light on 6 GB RAM, accurate contacts, `pip install`; Gazebo needs Linux and a lot of RAM |
| Language | Python 3.12 | Same as the lesson series |
| Editor / tracking | VS Code + git + GitHub, notebook in `docs/notebook/` | Everything versioned |
| Grasp type | **Top-down** (gripper pointing down), yaw aligned with object | Needs only x, y, z, yaw → 5-DOF arm is sufficient |
| ROS 2 | Deferred to Phase 6 (WSL2 + ROS 2 Jazzy) | Learn ROS with code that already works |

## 3. The robot at a glance
Numbers read from the model by `scripts/02_inspect_joints.py`:

| # | Joint | Axis (at home) | Range | Next segment |
|---|---|---|---|---|
| 1 | `shoulder_pan` | vertical (−z) | ±110° | 6.5 cm to shoulder_lift |
| 2 | `shoulder_lift` | horizontal (y) | ±100° | 11.6 cm (upper arm) |
| 3 | `elbow_flex` | horizontal (y) | ±96.8° | 13.5 cm (forearm) |
| 4 | `wrist_flex` | horizontal (y) | ±95° | 16.0 cm to the tip |
| 5 | `wrist_roll` | along the gripper (x) | −157° … +163° | — |
| – | `gripper` | jaw hinge | −10° … +100° | — |

Total arm mass ≈ 0.63 kg. Servos: Feetech STS3215 (modelled as position actuators).

## 4. Success metric (draft — freeze before Phase 5)
One **trial**:
1. Cube (3 cm) spawned at a random reachable (x, y) on the table and random yaw ∈ [0°, 90°).
2. Target zone = a fixed 4 cm × 4 cm square.
3. The pipeline runs **with no access to ground truth** (camera only).

A trial **succeeds** if, within 20 s of simulated time:
- the cube is lifted ≥ 3 cm above the table at some point, **and**
- it ends at rest with its centre inside the target zone.

Report: success rate over N = 100 trials with a 95 % confidence interval, plus counts per failure
type (detection error, IK unreachable, grasp slip, collision, placement miss, timeout).
Baseline to compare against: the same pipeline using the **ground-truth** cube pose (so we can
separate perception errors from grasping errors).

## 5. Roadmap

| Phase | Milestone | Main theory to learn | Done when |
|---|---|---|---|
| 0 | Repo, model, viewer | Links, joints, DOF, MJCF | `pytest` green, lesson 01 written ✅ |
| 1 | Forward kinematics | Rotation matrices, homogeneous transforms, DH parameters | Our FK matches MuJoCo to 3e-16 m; exact DH table derived ✅ |
| 2 | Inverse kinematics | Geometric IK (planar 3R + base), Jacobian, damped least squares | Tip reaches random reachable poses to < 1 mm |
| 3 | Scripted pick-and-place (ground-truth pose) | Trajectory generation (cubic/quintic), servo control, grasp contacts | Baseline success rate measured |
| 4 | Vision | Pinhole camera model, intrinsics/extrinsics, colour segmentation, pixel → world | Cube pose error < 5 mm, < 5° |
| 5 | Full pipeline + evaluation harness | Statistics of success rates, randomization | Success-rate report with failure breakdown |
| 6 | ROS 2 port | Nodes, topics, services, launch files | Same pipeline as 3 ROS 2 nodes |
| + | Stretch | Clutter, domain randomization, learned grasp detection, depth camera | — |

## 6. Open questions
- Place target: fixed, or also picked from camera (e.g. a coloured pad)?
- Top camera only, or also the wrist camera (model exists: `so101_new_calib_camera.xml` upstream)?
- Do we keep MuJoCo inside ROS 2 in Phase 6 or switch to Gazebo?
