# SO-101 Vision Pick-and-Place (MuJoCo)

A simulated **5-DOF SO-101 robot arm** that finds an object with a camera, picks it up and places it
at a target. The result is a documented **success rate** over randomized trials. Built step by step
as a learning project: every stage comes with a lesson (theory + worked numbers), a script and
tests.

![SO-101 in the scene](docs/images/scene_overview.png)

## Status
| Phase | Topic | Status |
|---|---|---|
| 0 | Repo, robot model, viewer, links/joints/DOF | ✅ |
| 1 | Forward kinematics | ✅ |
| 2 | Inverse kinematics | ⏳ next |
| 3 | Scripted pick-and-place (ground-truth pose) | |
| 4 | Vision: camera model, detection, pixel → world | |
| 5 | Evaluation harness, success rate | |
| 6 | ROS 2 port | |

Details: [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) · Lessons: [`docs/lessons/`](docs/lessons/) ·
Log: [`docs/notebook/`](docs/notebook/)

## Setup (Windows, VS Code)
Requirements: Python 3.10+ (MuJoCo, NumPy and Matplotlib all ship Windows wheels for 3.14), Git.

**Windows, first time only.** PowerShell blocks venv activation scripts by default. Allow them for your
user account once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

```powershell
cd "path\to\SO-101-Pick&Place"
python -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest                  # should print: 15 passed
```
Linux/macOS: `source .venv/bin/activate` instead of the activate line.

## Run
```powershell
python scripts/01_view_robot.py       # interactive viewer, joint sliders in the right panel
python scripts/02_inspect_joints.py   # links, joints, DOF, FK experiments printed to terminal
python scripts/03_render_snapshot.py   # save PNGs from the overview and top-down cameras
python scripts/04_forward_kinematics.py  # our own FK vs MuJoCo, and the DH table
```

## Layout
```
models/so101/     robot model (upstream, unmodified) + our scene.xml
src/so101/        library code (import so101)
scripts/          numbered lesson scripts
tests/            pytest checks
docs/
  PROJECT_BRIEF.md  goals, decisions, success metric, roadmap
  lessons/          theory, one file per lesson
  notebook/         dated work log
  images/           figures
CLAUDE.md         context file for AI assistants
```

## Credits
Robot model: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
Simulator: [MuJoCo](https://mujoco.org).
