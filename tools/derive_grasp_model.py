"""Generate models/so101/so101_pick_place.xml from the upstream SO-101 model.

Why this exists
---------------
MuJoCo collides mesh geoms as their **convex hull**. The SO-101's fixed finger mesh wraps around
the jaw hinge, so its hull fills the whole space between the jaws: nothing can ever be gripped,
and objects get shoved aside instead. (Upstream hit the same class of problem and already removed
the base collision meshes for it.)

Fix: keep the meshes for rendering, turn OFF their collisions on the two gripper bodies, and add
one box "pad" per finger, placed on the real inner faces measured from the mesh vertices:

    fixed finger inner face  ~ tip frame z = 0.000  (pad top)
    moving jaw inner face    ~ tip frame z = +0.017 (pad bottom, at gripper = 0 deg)

The pads are deliberately thin (3 mm) and short (3 cm), and they stop 1.5 cm short of the
fingertip point: a pad that hangs below the grasp centre digs into the table before the jaws
reach an object lying on it, and the shoulder servo saturates trying to push through it.

The moving pad is also pre-rotated by 26 degrees, the jaw angle at which a 3 cm object is held.
The SO-101's jaw swings on a hinge, so flat pads are parallel at exactly one angle; at any other
angle they touch along an edge and wedge the object out of the gripper. Real grippers solve this
with angled or compliant fingertips - this is the simulation equivalent.

Run:  python tools/derive_grasp_model.py
"""

from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models" / "so101"
SRC = MODELS / "so101_new_calib.xml"
DST = MODELS / "so101_pick_place.xml"

# pad geometry, in the frames of the bodies they attach to (see docs/lessons/04-*.md)
FIXED_PAD = ('      <!-- grasp pad, added by tools/derive_grasp_model.py -->\n'
             '      <geom name="pad_fixed" type="box" size="0.015 0.012 0.003" '
             'pos="-0.0109 -0.00022 -0.068127" quat="0.707107 0 0.707107 0" '
             'group="3" friction="2.0 0.05 0.001" condim="4" rgba="0.2 0.8 0.4 1"/>\n')
MOVING_PAD = ('          <!-- grasp pad, added by tools/derive_grasp_model.py -->\n'
              '          <geom name="pad_moving" type="box" size="0.015 0.012 0.003" '
              'pos="-0.016102 -0.04191 0.019018" quat="-0.374709 0.374709 -0.599661 0.599661" '
              'group="3" friction="2.0 0.05 0.001" condim="4" rgba="0.2 0.8 0.4 1"/>\n')

DISABLE = ' contype="0" conaffinity="0"'


def main() -> None:
    lines = SRC.read_text().splitlines(keepends=True)
    out, body = [], None
    for line in lines:
        if '<body name="gripper"' in line:
            body = "gripper"
        elif '<body name="moving_jaw_so101_v1"' in line:
            body = "moving_jaw"
        # turn off collisions for the two bodies' hull meshes
        if body in ("gripper", "moving_jaw") and 'class="collision"' in line:
            line = line.replace('class="collision"', 'class="collision"' + DISABLE)
        out.append(line)
        if body == "gripper" and '<joint' in line and 'wrist_roll' in line:
            out.append(FIXED_PAD)
        if body == "moving_jaw" and '<joint' in line and 'name="gripper"' in line:
            out.append(MOVING_PAD)
    DST.write_text(
        "".join(out).replace('<mujoco model="so101_new_calib">',
                             '<!-- DERIVED FILE - do not edit by hand. See tools/derive_grasp_model.py -->\n'
                             '<mujoco model="so101_pick_place">'))
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
