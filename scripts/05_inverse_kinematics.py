"""Lesson 03 - inverse kinematics: closed-form top-down grasps, plus a numerical solver.

Run:
    python scripts/05_inverse_kinematics.py           # ~30 s (1 cm workspace grid)
    python scripts/05_inverse_kinematics.py --fine    # ~3 min (5 mm grid, smoother map)
Writes docs/images/workspace_topdown.png, ik_triangle.png and grasp_pose.png
"""

import argparse
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

from so101 import load_model, set_joint_angles  # noqa: E402
from so101.ik import TopDownIK, dls_ik, geometric_jacobian, yaw_to_rotation  # noqa: E402
from so101.kinematics import Chain, make_T  # noqa: E402
from so101.model import REPO_ROOT  # noqa: E402

CUBE = np.array([0.22, 0.08, 0.015])   # cube centre in scene.xml
GRASP_Z = 0.015                        # grasp at the cube's mid-height
IMAGES = REPO_ROOT / "docs" / "images"

INK, MUTED = "#1f2933", "#7b8794"
FILL_SOME, FILL_ALL = "#c3d4f7", "#3b5bdb"


def fmt(v, nd=4):
    return "[" + ", ".join(f"{x:+.{nd}f}" for x in np.asarray(v).ravel()) + "]"


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def grasp_errors(chain, q, target, psi):
    T = chain.fk(q)
    pos = np.linalg.norm(T[:3, 3] - target)
    approach = np.degrees(np.arccos(np.clip(T[:3, 0] @ [0, 0, -1], -1, 1)))
    yaw = np.arctan2(T[1, 2], T[0, 2])
    yaw_err = np.degrees(abs((yaw - psi + np.pi / 2) % np.pi - np.pi / 2))  # 180 deg symmetric
    return pos, approach, yaw_err


def main(step: float = 0.01) -> None:
    model, data = load_model()
    chain = Chain(model)
    ik = TopDownIK(model, chain)

    # ------------------------------------------------- 1. the geometry the solver uses
    section("1. GEOMETRY READ FROM THE MODEL")
    print(f"planar link lengths      L1 = {ik.L1 * 100:.2f} cm, L2 = {ik.L2 * 100:.2f} cm")
    print(f"wrist centre in tip frame  = {fmt(ik.W_in_tip)} m")
    print(f"arm plane offset from pan axis = {ik.lateral * 1000:+.2f} mm (tiny, but not zero)")
    print(f"reach of the planar arm  = {(ik.L1 + ik.L2) * 100:.1f} cm from the shoulder axis")

    # ------------------------------------------------------- 2. all branches for one grasp
    psi = np.radians(30)
    target = np.array([CUBE[0], CUBE[1], GRASP_Z])
    section(f"2. EVERY SOLUTION for tip at {fmt(target)} m, jaws at {np.degrees(psi):.0f} deg")
    print(f"{'shoulder':<10}{'elbow':<7}{'in limits':<11}{'margin':>8}   joint angles [deg]")
    for s in ik.solve(target, psi, all_solutions=True):
        print(f"{s.shoulder:<10}{s.elbow:<7}{str(s.within_limits):<11}"
              f"{np.degrees(s.limit_margin):7.1f}   {np.degrees(s.q).round(1)}")
    best = ik.solve(target, psi)
    pos, approach, yaw_err = grasp_errors(chain, best.q, target, psi)
    print(f"\nchosen: {np.degrees(best.q).round(2)} ({best.shoulder}/{best.elbow} branch)")
    print(f"check by FK -> position error {pos * 1e6:.3f} um, approach {approach:.2e} deg, "
          f"yaw {yaw_err:.2e} deg")

    # ------------------------------------------------------------ 3. round-trip over the table
    section("3. ROUND TRIP: random targets -> IK -> FK")
    rng = np.random.default_rng(0)
    pos_errs, ang_errs, tried, solved = [], [], 0, 0
    t0 = time.perf_counter()
    while tried < 500:
        p = np.array([rng.uniform(0.05, 0.35), rng.uniform(-0.25, 0.25), GRASP_Z])
        psi_i = rng.uniform(-np.pi, np.pi)
        tried += 1
        sol = ik.solve(p, psi_i)
        if sol is None:
            continue
        solved += 1
        e_p, e_a, e_y = grasp_errors(chain, sol.q, p, psi_i)
        pos_errs.append(e_p)
        ang_errs.append(max(e_a, e_y))
    dt = time.perf_counter() - t0
    print(f"{solved}/{tried} targets solvable within the joint limits")
    print(f"max position error {max(pos_errs) * 1e9:.1f} nm, max angle error {max(ang_errs):.2e} deg")
    print(f"{dt / tried * 1e6:.0f} us per call (all branches, including the failures)")

    # --------------------------------------------------- 4. numerical IK for comparison
    section("4. NUMERICAL IK (damped least squares) on the same target")
    T_target = make_T(yaw_to_rotation(psi), target)
    t0 = time.perf_counter()
    q_num, err = dls_ik(chain, T_target, np.zeros(5), limits=ik.limits)
    dt = time.perf_counter() - t0
    print(f"solution {np.degrees(q_num).round(2)}  residual {err:.2e}  ({dt * 1000:.1f} ms)")
    print(f"closed form {np.degrees(best.q).round(2)}  -> difference "
          f"{np.degrees(np.abs(q_num - best.q)).max():.3f} deg")
    J = geometric_jacobian(chain, best.q)
    print(f"Jacobian at the solution: condition number {np.linalg.cond(J):.1f}, "
          f"smallest singular value {np.linalg.svd(J, compute_uv=False)[-1]:.4f}")

    # ------------------------------------------------------------------ 5. workspace map
    section("5. TOP-DOWN WORKSPACE MAP")
    xs = np.arange(-0.10, 0.451, step)
    ys = np.arange(-0.35, 0.351, step)
    yaws = np.radians(np.arange(0, 180, 30))
    grid = np.zeros((len(ys), len(xs)))
    print(f"sweeping {len(xs) * len(ys) * len(yaws)} IK calls on a {step * 100:.1f} cm grid "
          f"(this is the slow part)", flush=True)
    t0 = time.perf_counter()
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            n = sum(ik.reachable([x, y, GRASP_Z], psi_k) for psi_k in yaws)
            grid[iy, ix] = 0 if n == 0 else (2 if n == len(yaws) else 1)
        done = (iy + 1) / len(ys)
        elapsed = time.perf_counter() - t0
        print(f"\r  row {iy + 1}/{len(ys)}  {done * 100:5.1f}%  "
              f"eta {elapsed / done - elapsed:5.1f} s", end="", flush=True)
    print()
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    print(f"grasp height z = {GRASP_Z * 100:.1f} cm")
    print(f"  area reachable at some yaw : {np.sum(grid >= 1) * cell * 1e4:.0f} cm^2")
    print(f"  area reachable at any yaw  : {np.sum(grid == 2) * cell * 1e4:.0f} cm^2")
    cube_n = sum(ik.reachable([CUBE[0], CUBE[1], GRASP_Z], psi_k) for psi_k in yaws)
    print(f"  cube at ({CUBE[0]}, {CUBE[1]}): reachable at {cube_n}/{len(yaws)} yaw angles")

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    ax.set_facecolor("white")
    ax.contourf(xs, ys, grid, levels=[0.5, 1.5, 2.5], colors=[FILL_SOME, FILL_ALL])
    ax.plot([0], [0], marker="s", ms=9, color=INK, zorder=5)
    ax.annotate("robot base", (0, 0), textcoords="offset points", xytext=(10, 8),
                color=INK, fontsize=9)
    ax.plot(CUBE[0], CUBE[1], marker="o", ms=9, color="#d6336c", zorder=5)
    ax.annotate("cube", (CUBE[0], CUBE[1]), textcoords="offset points", xytext=(10, 8),
                color="#d6336c", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", color=MUTED)
    ax.set_ylabel("y [m]", color=MUTED)
    ax.set_title(f"Where the SO-101 can grasp from above (tip at z = {GRASP_Z * 100:.1f} cm)",
                 color=INK, fontsize=11, pad=12)
    ax.grid(color="#e4e7eb", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd2d9")
    ax.tick_params(colors=MUTED, labelsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FILL_ALL),
               plt.Rectangle((0, 0), 1, 1, color=FILL_SOME)]
    ax.legend(handles, ["any jaw angle", "only some jaw angles"], frameon=False,
              loc="upper left", fontsize=9, labelcolor=INK)
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / "workspace_topdown.png", facecolor="white")
    print(f"  wrote {(IMAGES / 'workspace_topdown.png').relative_to(REPO_ROOT)}")

    # ------------------------------------------------- 5b. how high can we hover?
    section("5b. MAXIMUM TOP-DOWN HEIGHT (the wrist_flex limit bites)")

    def max_height(x, y, psi_k=0.0, hi=0.35):
        lo = 0.0
        if not ik.reachable([x, y, lo], psi_k):
            return None
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ik.reachable([x, y, mid], psi_k) else (lo, mid)
        return lo

    for x in (0.10, 0.15, 0.20, 0.25, 0.30):
        h = max_height(x, 0.0)
        print(f"  at (x={x:.2f}, y=0.00): highest top-down pose is z = "
              f"{h * 100:.1f} cm" if h else f"  at (x={x:.2f}, y=0.00): unreachable")
    h_cube = max_height(CUBE[0], CUBE[1])
    print(f"  above the cube: {h_cube * 100:.1f} cm  <- the hover waypoint must stay below this")

    # ------------------------------------------------- 5c. the planar triangle figure
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    sols = [s for s in ik.solve(target, psi, all_solutions=True) if s.shoulder == "front"]
    seen = {}
    for s in sols:
        seen.setdefault(s.elbow, s)
    for (name, s), style in zip(sorted(seen.items(), reverse=True), ("-", "--")):
        F = chain.frames(s.q)
        S, E, W = F[2][:3, 3], F[3][:3, 3], F[4][:3, 3]
        radial = np.array([np.cos(np.arctan2(W[1] - S[1], W[0] - S[0])),
                           np.sin(np.arctan2(W[1] - S[1], W[0] - S[0]))])
        def flat(p):
            return np.array([(p[:2] - S[:2]) @ radial, p[2] - S[2]])
        pts = np.array([flat(S), flat(E), flat(W)])
        colour = FILL_ALL if name == "up" else MUTED
        ax.plot(pts[:, 0], pts[:, 1], style, color=colour, lw=2.2, marker="o", ms=7,
                label=f"elbow {name}")
        ax.annotate("L1", 0.5 * (pts[0] + pts[1]), textcoords="offset points", xytext=(-16, 4),
                    color=colour, fontsize=9)
        ax.annotate("L2", 0.5 * (pts[1] + pts[2]), textcoords="offset points", xytext=(6, 4),
                    color=colour, fontsize=9)
        if name == "up":
            ax.plot([pts[0, 0], pts[2, 0]], [pts[0, 1], pts[2, 1]], ":", color=INK, lw=1.4)
            ax.annotate("D", 0.5 * (pts[0] + pts[2]), textcoords="offset points", xytext=(0, -14),
                        color=INK, fontsize=9)
            ax.annotate("shoulder", pts[0], textcoords="offset points", xytext=(-20, -16),
                        color=INK, fontsize=9)
            ax.annotate("wrist centre", pts[2], textcoords="offset points", xytext=(-30, -18),
                        color=INK, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("distance in the arm's plane [m]", color=MUTED)
    ax.set_ylabel("height above the shoulder axis [m]", color=MUTED)
    ax.set_title("The two-link triangle, both elbow branches", color=INK, fontsize=11, pad=10)
    ax.grid(color="#e4e7eb", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd2d9")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(IMAGES / "ik_triangle.png", facecolor="white")
    print(f"  wrote {(IMAGES / 'ik_triangle.png').relative_to(REPO_ROOT)}")

    # ------------------------------------------------------- 6. picture of the grasp pose
    set_joint_angles(model, data, [*best.q, np.radians(40)])
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data, camera="overview")
        plt.imsave(IMAGES / "grasp_pose.png", r.render())
    print(f"  wrote {(IMAGES / 'grasp_pose.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", action="store_true",
                        help="use a 5 mm workspace grid instead of 1 cm (about 4x slower)")
    args = parser.parse_args()
    main(step=0.005 if args.fine else 0.01)
