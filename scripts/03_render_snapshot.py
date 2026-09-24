"""Save PNG snapshots of the scene from the 'overview' and 'top_cam' cameras.

Run:
    python scripts/03_render_snapshot.py            # writes docs/images/*.png
Optional joint angles in degrees (shoulder_pan ... gripper):
    python scripts/03_render_snapshot.py 30 -20 40 -20 0 20
"""

import sys

import mujoco
import numpy as np

from so101 import load_model, set_joint_angles
from so101.model import REPO_ROOT

OUT_DIR = REPO_ROOT / "docs" / "images"


def save_png(path, rgb):
    try:
        import imageio.v3 as iio  # optional
        iio.imwrite(path, rgb)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.imsave(path, rgb)


def main() -> None:
    model, data = load_model()
    if len(sys.argv) == 7:
        set_joint_angles(model, data, np.radians([float(a) for a in sys.argv[1:]]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for cam in ("overview", "top_cam"):
            renderer.update_scene(data, camera=cam)
            path = OUT_DIR / f"scene_{cam}.png"
            save_png(path, renderer.render())
            print("wrote", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
