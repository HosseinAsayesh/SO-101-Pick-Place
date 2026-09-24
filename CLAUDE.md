# CLAUDE.md — context for AI assistants working in this repo

Read this first, then `docs/PROJECT_BRIEF.md`, then the newest file in `docs/notebook/`.

## What this project is
Vision-based pick-and-place with a **simulated SO-101 arm** (5 DOF + gripper) in **MuJoCo**.
Pipeline: camera → object detection / pose → inverse kinematics + motion plan → joint control.
The deliverable is a documented **success rate** over randomized trials (definition in the brief).
Simulation only, no hardware. A ROS 2 port comes later (Phase 6).

## Who I am and how to help me
- Mechanical engineering bachelor's (robotics, control, dynamics). Knows classical control (PID,
  root locus, Bode). **Many theorems (state-space, kinematics formalisms, optimization, computer
  vision) were not taught in my degree — teach them.**
- Explain step by step and show the reasoning. Use specific numeric examples, ideally with this
  robot's real numbers.
- Every new concept is delivered as: **a lesson doc in `docs/lessons/` + a runnable script in
  `scripts/` + tests in `tests/`**. Derive the math before using a library that hides it, then
  check our derivation against MuJoCo.
- Don't write big chunks of code I haven't seen explained. Prefer small, readable steps.
- The project is a portfolio piece for research/job applications, so keep it clean and honest.
  Record failures and wrong turns in the notebook too.

## Environment
- Windows 10, VS Code, Python 3.14, virtual env in `.venv/` (not committed).
- Laptop: Ryzen 7 3700U, ~6 GB usable RAM, GTX 1050. Keep things light (no Gazebo on Windows).
- Setup: `python -m venv .venv` → `.venv\Scripts\Activate.ps1` → `python -m pip install -r requirements.txt`.
  (PowerShell execution policy must be RemoteSigned for the user; always use `python -m pip`.)
- Run tests: `pytest`. Run a lesson script: `python scripts/NN_name.py` from the repo root.

## Conventions
- SI units everywhere: metres, radians, seconds, kg. Print degrees only for humans.
- World frame: z up, table surface at z = 0, robot base at the origin, arm points +x at home.
- Joint order (always this order in arrays):
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- "Home" = all joints 0 rad = middle of each joint's range ("new calibration" model).
- The tip frame is the MuJoCo site `gripperframe`.
- Robot model files in `models/so101/` come from TheRobotStudio/SO-ARM100 (Apache-2.0). Do not edit
  `so101_new_calib.xml`; put scene changes in `scene.xml`.
- Library code goes in `src/so101/`, experiments/lesson demos go in `scripts/NN_*.py`.
- Numbered scripts match lesson numbers (lesson 02 → `scripts/02_*`... bumping as needed).
- Add a dependency only when it's needed, to both `requirements.txt` and `pyproject.toml`.
- Commits: small, present tense, prefixed by area, e.g. `kinematics: add planar 3R FK`,
  `docs: lesson 02 draft`, `notebook: 2026-09-20`.

## Session ritual
1. Read the latest notebook entry to see where we stopped.
2. Do the work (lesson → script → test).
3. Run `pytest`.
4. Add/extend today's notebook entry (`docs/notebook/YYYY-MM-DD.md`, copy `_template.md`).
5. Commit.
