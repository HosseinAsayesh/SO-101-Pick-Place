"""Lesson 04 - the first full pick-and-place, with the cube's pose known exactly.

Run:
    python scripts/06_pick_and_place.py                      # headless: figures + 12 random trials
    python scripts/06_pick_and_place.py --watch --trials 0   # watch it in the MuJoCo viewer
    python scripts/06_pick_and_place.py --watch --speed 0.5  # half speed, to see the grasp
Writes docs/images/pickplace_run.png and docs/images/pickplace_filmstrip.png
"""

import argparse
import collections
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

from so101 import ARM_JOINTS, load_model  # noqa: E402
from so101.ik import GRASP_OFFSET, TopDownIK  # noqa: E402
from so101.kinematics import Chain  # noqa: E402
from so101.model import REPO_ROOT  # noqa: E402
from so101.pick_place import (  # noqa: E402
    jaw_angle_for_gap,
    judge,
    pad_gap,
    plan_pick_place,
    run_plan,
    run_plan_live,
    run_trial,
    set_cube,
    tracking_error,
)

PICK = (0.22, 0.08)
PLACE = (0.15, -0.15)
PICK_YAW = np.radians(25.0)
IMAGES = REPO_ROOT / "docs" / "images"

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
JOINT_COLORS = ["#1864ab", "#e8590c", "#0ca678", "#7048e8", "#c2255c"]  # CVD-checked


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor("white")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd2d9")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=8, loc="left")


def phase_spans(t, phases):
    spans, start, current = [], t[0], phases[0]
    for ti, p in zip(t, phases):
        if p != current:
            spans.append((current, start, ti))
            start, current = ti, p
    spans.append((current, start, t[-1]))
    return spans


def main(trials: int = 12, watch: bool = False, speed: float = 1.0) -> None:
    model, data = load_model()
    chain = Chain(model)
    ik = TopDownIK(model, chain)

    # ------------------------------------------------------------- 1. the gripper
    section("1. GRIPPER: how wide is the gap, and what do we command?")
    for deg in (0, 15, 26, 40, 55, 90):
        print(f"  jaw at {deg:3d} deg -> pads {pad_gap(model, data, np.radians(deg)) * 100:5.2f} cm apart")
    closed = jaw_angle_for_gap(model, data, 0.030 - 0.004)
    opened = jaw_angle_for_gap(model, data, 0.030 + 0.012)
    print(f"  for a 3 cm cube: open at {np.degrees(opened):.1f} deg, close to {np.degrees(closed):.1f} deg")

    # ---------------------------------------------------------- 2. plan and run it
    section("2. ONE PICK AND PLACE")
    mujoco.mj_resetData(model, data)
    set_cube(model, data, PICK[0], PICK[1], PICK_YAW)
    plan = plan_pick_place(model, chain, ik, PICK, PLACE, pick_yaw=PICK_YAW, data=data)
    print(f"{'segment':<12}{'duration [s]':>13}")
    for seg in plan.trajectory.segments:
        print(f"{seg.label:<12}{seg.duration:>13.2f}")
    print(f"{'total':<12}{plan.trajectory.duration:>13.2f}")

    t0 = time.perf_counter()
    if watch:
        print("\nopening the MuJoCo viewer - close the window when you are done watching")
        log = run_plan_live(model, data, chain, plan, reset=False, log_every=5, speed=speed)
    else:
        log = run_plan(model, data, chain, plan, reset=False, log_every=5)
    wall = time.perf_counter() - t0
    verdict = judge(log, PLACE)
    print(f"\nsimulated {plan.trajectory.duration:.1f} s in {wall:.1f} s of wall clock")
    for k in ("success", "lifted", "max_lift_cm", "placed", "final_xy_error_mm"):
        print(f"  {k:<18} {verdict[k]}")

    t, q_cmd, q_act, tip, cube, phases = log.arrays()
    err = np.degrees(tracking_error(log))
    print(f"\n  worst tracking error {np.abs(err).max():.2f} deg "
          f"(joint {ARM_JOINTS[np.unravel_index(np.abs(err).argmax(), err.shape)[1]]})")
    for name, a, b in phase_spans(t, phases):
        last = max(i for i, ti in enumerate(t) if ti <= b)
        print(f"  end of {name:<9} worst joint error {np.abs(err[last]).max():6.3f} deg")

    # --------------------------------------------------------------- 3. figures
    section("3. FIGURES")
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), dpi=150, sharex=True)
    spans = phase_spans(t, phases)
    for ax in axes:
        for i, (name, a, b) in enumerate(spans):
            if i % 2 == 0:
                ax.axvspan(a, b, color="#f4f6f8", zorder=0)

    ax = axes[0]
    ax.plot(t, cube[:, 2] * 100, lw=2.2, color="#c2255c", label="cube")
    grasp_h = np.array([(chain.fk(q)[:3, 3] + chain.fk(q)[:3, :3] @ GRASP_OFFSET)[2] for q in q_act])
    ax.plot(t, grasp_h * 100, lw=2.0, color="#1864ab", label="grasp point")
    ax.axhline(1.5, color=MUTED, lw=1.0, ls=":")
    ax.annotate("cube resting on the table", (t[-1], 1.5), color=MUTED, fontsize=8,
                ha="right", va="bottom")
    style(ax, "", "height [cm]", "Pick and place: what moves when")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right")
    for name, a, b in spans:
        ax.annotate(name, ((a + b) / 2, ax.get_ylim()[1]), color=MUTED, fontsize=8,
                    ha="center", va="top", rotation=90)

    ax = axes[1]
    for j, (name, colour) in enumerate(zip(ARM_JOINTS, JOINT_COLORS)):
        ax.plot(t, err[:, j], lw=1.8, color=colour, label=name)
    style(ax, "time [s]", "commanded − actual [deg]", "Servo tracking error")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=5, loc="lower center")
    fig.tight_layout()
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / "pickplace_run.png", facecolor="white")
    print(f"  wrote {(IMAGES / 'pickplace_run.png').relative_to(REPO_ROOT)}")

    # filmstrip: re-run, grabbing a frame in the middle of each phase
    want = ["approach", "descend", "close", "lift", "transfer", "lower", "release", "home"]
    frames = {}
    mujoco.mj_resetData(model, data)
    set_cube(model, data, PICK[0], PICK[1], PICK_YAW)
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        q0, g0, _ = plan.trajectory.at(0.0)
        data.ctrl[:5], data.ctrl[5] = q0, g0
        marks = {}
        elapsed = 0.0
        for seg in plan.trajectory.segments:
            marks.setdefault(seg.label, elapsed + seg.duration * 0.6)
            elapsed += seg.duration
        for i in range(int(plan.trajectory.duration / model.opt.timestep)):
            tt = i * model.opt.timestep
            q, g, ph = plan.trajectory.at(tt)
            data.ctrl[:5] = q
            if g is not None:
                data.ctrl[5] = g
            mujoco.mj_step(model, data)
            for label in want:
                if label not in frames and tt >= marks.get(label, 1e9):
                    renderer.update_scene(data, camera="overview")
                    frames[label] = renderer.render()
    cols = 4
    rows = int(np.ceil(len(want) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(11, 5.6), dpi=150)
    for ax, label in zip(axs.ravel(), want):
        ax.imshow(frames[label])
        ax.set_title(label, color=INK, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(IMAGES / "pickplace_filmstrip.png", facecolor="white")
    print(f"  wrote {(IMAGES / 'pickplace_filmstrip.png').relative_to(REPO_ROOT)}")

    # ------------------------------------------------- 4. baseline success rate
    if trials > 0:
        section(f"4. BASELINE SUCCESS RATE over {trials} random trials (cube pose known exactly)")
        rng = np.random.default_rng(0)
        reasons, errors, done = collections.Counter(), [], 0
        while done < trials:
            x, y = rng.uniform(0.15, 0.30), rng.uniform(-0.05, 0.22)
            yaw = rng.uniform(0, np.pi / 2)
            if not (ik.can_grasp([x, y, 0.02], yaw) and ik.can_grasp([x, y, 0.075], yaw)):
                continue
            done += 1
            r = run_trial(model, data, chain, ik, (x, y), PLACE, yaw=yaw)
            reasons[r["reason"]] += 1
            if r["success"]:
                errors.append(r["final_xy_error_mm"])
            print(f"  {done:3d}/{trials}  cube ({x:.3f}, {y:.3f}) yaw {np.degrees(yaw):5.1f} deg"
                  f"  ->  {r['reason']}", flush=True)
        k, rate = reasons["ok"], reasons["ok"] / trials
        # Wilson interval: the normal approximation gives a useless +-0 when every trial passes
        z = 1.96
        centre = (k + z ** 2 / 2) / (trials + z ** 2)
        half = z / (trials + z ** 2) * np.sqrt(k * (trials - k) / trials + z ** 2 / 4)
        print(f"\n  success {k}/{trials} = {rate * 100:.0f}%  "
              f"(95% Wilson CI {max(0, centre - half) * 100:.0f}-{min(1, centre + half) * 100:.0f}%)")
        print(f"  failure reasons: {dict(reasons)}")
        if errors:
            print(f"  placement error: mean {np.mean(errors):.1f} mm, worst {np.max(errors):.1f} mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12, help="random trials for the baseline")
    parser.add_argument("--watch", action="store_true",
                        help="open the MuJoCo viewer and run the pick-and-place at real speed")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed for --watch (2.0 = twice real time)")
    main(**vars(parser.parse_args()))
