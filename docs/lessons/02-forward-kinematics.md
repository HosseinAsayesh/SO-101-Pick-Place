# Lesson 02 — Forward Kinematics

**The question:** given the five joint angles, where is the gripper, and which way is it pointing?

**Why it matters:** everything downstream needs it. Inverse kinematics (lesson 03) is the same
equations solved backwards. The Jacobian is its derivative. And when a grasp misses, FK is how you
tell whether the arm went where you asked.

**Run alongside:**
```
python scripts/04_forward_kinematics.py
pytest tests/test_kinematics.py
```
Code: `src/so101/kinematics.py`. It uses no MuJoCo kinematics at all — MuJoCo is only the referee.

---

## 1. Frames and points

A **frame** is an origin plus three perpendicular unit vectors (x, y, z), right-handed:
x × y = z. Every link of the arm carries its own frame, glued to it.

The same physical point has different coordinates in different frames. Write **ᴬp** for the
coordinates of point p in frame A. The job of this lesson is converting between them.

Our convention throughout:

$$ {}^{A}p = T_{AB} \, {}^{B}p $$

$T_{AB}$ takes coordinates in B (the child) and gives coordinates in A (the parent).

---

## 2. Rotation matrices

### Derivation in 2D
Rotate frame B relative to frame A by angle θ about their shared z. Where do B's basis vectors
point, expressed in A?

$$ {}^{A}x_B = (\cos\theta,\ \sin\theta), \qquad {}^{A}y_B = (-\sin\theta,\ \cos\theta) $$

A point with coordinates $(u, v)$ in B is $u\,{}^{A}x_B + v\,{}^{A}y_B$ in A. Stack the two vectors
as columns and that sum is a matrix product:

$$ R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} $$

**This is the key idea and it generalizes: the columns of a rotation matrix are the child frame's
axes, written in the parent frame.** When you print a rotation matrix from our code, read it
column by column: "where does the child's x point, where does its y point, where does its z point".

### In 3D
Rotating about each axis in turn (the axis's own row and column stay as in the identity):

$$
R_x(a)=\begin{bmatrix}1&0&0\\0&\cos a&-\sin a\\0&\sin a&\cos a\end{bmatrix},\quad
R_y(a)=\begin{bmatrix}\cos a&0&\sin a\\0&1&0\\-\sin a&0&\cos a\end{bmatrix},\quad
R_z(a)=\begin{bmatrix}\cos a&-\sin a&0\\\sin a&\cos a&0\\0&0&1\end{bmatrix}
$$

Note the sign pattern in $R_y$ is flipped compared to the other two. That's not a typo: it follows
from the right-hand rule and the cyclic order x → y → z → x. Getting this wrong is the single most
common FK bug.

### Properties (worth memorizing)
1. **Orthonormal:** $R^{\mathsf T}R = I$. The columns are unit length and mutually perpendicular.
2. **Inverse = transpose:** $R^{-1} = R^{\mathsf T}$. Never call `np.linalg.inv` on a rotation.
3. **det R = +1.** If you get −1, you built a mirror, not a rotation.
4. **Composition is matrix multiplication, and order matters:** $R_z(90°)R_x(90°) \ne R_x(90°)R_z(90°)$.
   Rotations do not commute. (Try it with your hand and a book: the book ends up in different
   orientations.)
5. Reading right to left, $R_{AB}R_{BC}$ means "C relative to B, then B relative to A".

### Any axis: Rodrigues' formula
For a rotation by angle $t$ about a unit axis $k$ through the origin:

$$ R = I + \sin t\,K + (1-\cos t)K^2, \qquad
K = \begin{bmatrix}0&-k_z&k_y\\ k_z&0&-k_x\\ -k_y&k_x&0\end{bmatrix} $$

$K$ is the *skew-symmetric* matrix of $k$, defined by $Kv = k \times v$. Setting $k = (0,0,1)$ gives
$R_z$ back, as `test_rodrigues_reduces_to_elementary_rotations` checks.

### Quaternions (because the model file uses them)
MuJoCo stores each link's fixed rotation as a unit quaternion $q = (w, x, y, z)$, which encodes
"rotate by $t$ about unit axis $u$" as $q = (\cos\frac t2,\ \sin\frac t2\, u)$. Four numbers instead
of nine, no drift, no gimbal lock. The conversion to a matrix is in `quat_to_R`. The half-angle is
why the formula is full of squared terms: $\cos t = 1 - 2\sin^2\frac t2$.

Example from our robot: the `shoulder` link's quaternion is $(0, 0, -1, 0)$, i.e. $w = 0 \Rightarrow
t = 180°$, about the axis $(0,-1,0)$. A half turn about y. Sure enough `quat_to_R` returns
diag(−1, 1, −1).

---

## 3. Adding translation: homogeneous transforms

A link is rotated *and* offset from its parent:

$$ {}^{A}p = R\,{}^{B}p + t $$

That's an affine map, not a linear one, so it can't be a single 3×3 matrix — which would be
annoying, because chaining links would then alternate multiplications and additions. The fix is a
classic trick: add a fourth coordinate equal to 1.

$$
\begin{bmatrix}{}^{A}p\\1\end{bmatrix} =
\underbrace{\begin{bmatrix}R & t\\ 0\ 0\ 0 & 1\end{bmatrix}}_{T_{AB}}
\begin{bmatrix}{}^{B}p\\1\end{bmatrix}
$$

Now **composition is just matrix multiplication**, and it chains to any depth:

$$ T_{AC} = T_{AB} \, T_{BC} $$

The inverse has a closed form (derive it by asking what undoes "rotate then translate"):

$$ T^{-1} = \begin{bmatrix} R^{\mathsf T} & -R^{\mathsf T}t \\ 0 & 1\end{bmatrix} $$

That's `invert_T`, and `test_invert_T_is_the_inverse` checks it.

A 4×4 transform is also called a **pose**: position + orientation in one object. "The pose of the
gripper" means this matrix.

---

## 4. The forward-kinematics chain

Each joint contributes one transform that depends on its angle. Multiply along the chain:

$$ T_{\text{world},\,\text{tip}}(q) = T_{01}(q_1)\,T_{12}(q_2)\,T_{23}(q_3)\,T_{34}(q_4)\,T_{45}(q_5)\,T_{5,\text{tip}} $$

For the SO-101 the model gives each link a fixed offset and fixed rotation from its parent, and
**every joint sits at its link's origin and turns about that link's local z** (verified by the
asserts in `Chain.__init__`). So each step is:

$$ T_{i-1,i}(q_i) = \mathrm{Trans}(\text{offset}_i)\ \mathrm{Rot}(\text{quat}_i)\ R_z(q_i) $$

The fixed part places the joint; $R_z(q_i)$ turns it. That is the whole of `Chain.joint_transform`.

### Worked example by hand: where is the shoulder_lift axis at q₁ = 30°?
Constants for the first link (printed by the script):
offset₁ = (0.0388, 0, 0.0624), R₁ = diag(−1, 1, −1), and offset₂ = (−0.0304, −0.0183, −0.0542).

Step 1, the rotation part:

$$ R_1R_z(30°) = \begin{bmatrix}-1&0&0\\0&1&0\\0&0&-1\end{bmatrix}
\begin{bmatrix}0.866&-0.5&0\\0.5&0.866&0\\0&0&1\end{bmatrix}
= \begin{bmatrix}-0.866&0.5&0\\0.5&0.866&0\\0&0&-1\end{bmatrix} $$

Step 2, apply it to offset₂ and add offset₁:

- x: (−0.866)(−0.0304) + (0.5)(−0.0183) = 0.0263 − 0.0091 = **0.0172** → + 0.0388 = **0.0560**
- y: (0.5)(−0.0304) + (0.866)(−0.0183) = −0.0152 − 0.0158 = **−0.0310**
- z: (−1)(−0.0542) = **0.0542** → + 0.0624 = **0.1166**

So the shoulder_lift axis passes through (0.0560, −0.0310, 0.1166) m. The script's worked example
prints exactly that. Two things to notice: the height 0.1166 doesn't change with q₁ (the pan axis is
vertical), and y went negative for a positive angle, which is the "pan axis points down" quirk from
lesson 01.

### Does it work?
`scripts/04_forward_kinematics.py` runs 1000 random poses inside the joint limits and compares our
tip pose with MuJoCo's:

```
max tip position error : 3.28e-16 m
max rotation error     : 1.33e-15
```

That's floating-point round-off, far smaller than any physical scale in the robot. Our FK and MuJoCo's agree
exactly. `test_every_frame_matches_mujoco` goes further and checks every *link* frame, not only the
tip, which is what catches an error in the middle of the chain.

(The script also times both. Ignore the comparison: MuJoCo's call does far more than kinematics.)

---

## 5. Denavit–Hartenberg parameters

Our FK needs 6 numbers per link (3 offset + 3 rotation). Denavit and Hartenberg showed in 1955 that
**4 are enough per joint**, if you are willing to let the convention choose where the link frames
sit. Papers, textbooks and robot datasheets almost always describe an arm as a DH table, so you need
to be able to read one.

### The frame rules
For joint $i$ with axis $z_{i-1}$:

1. $z_i$ lies along joint $i{+}1$'s axis.
2. $x_i$ lies along the **common normal** — the unique line segment perpendicular to both $z_{i-1}$
   and $z_i$. If the axes intersect, use $z_{i-1} \times z_i$. If they're parallel, any common
   perpendicular works, so pick one and be consistent.
3. The origin of frame $i$ is where that common normal meets $z_i$.
4. $y_i$ follows from the right-hand rule.

Those constraints remove two of the six degrees of freedom, leaving four parameters:

| Parameter | Meaning | Measured |
|---|---|---|
| $\theta_i$ | joint angle | about $z_{i-1}$, from $x_{i-1}$ to $x_i$ |
| $d_i$ | link offset | along $z_{i-1}$ |
| $a_i$ | link length | along $x_i$ (the common-normal distance) |
| $\alpha_i$ | link twist | about $x_i$, from $z_{i-1}$ to $z_i$ |

and the link transform is always the same product:

$$ T_{i-1,i} = R_z(\theta_i)\ \mathrm{Trans}_z(d_i)\ \mathrm{Trans}_x(a_i)\ R_x(\alpha_i) $$

For a revolute joint $\theta_i$ is the variable and the other three are constants. (For a prismatic
joint it's the other way round: $d_i$ varies.)

### Deriving the table for the SO-101
`derive_dh()` does it in code, using the joint axes our own FK produces:
find the common normal between consecutive axes (`common_normal_feet`), build each frame by rule,
then read off the four numbers from each frame-to-frame transform with `T_to_dh`. The residual in
`T_to_dh` is the safety net: if the frames were built wrongly, the transform wouldn't have the DH
form at all and the residual would be large instead of 1e-16.

| Row | Joint | θ offset [°] | d [m] | a [m] | α [°] |
|---|---|---|---|---|---|
| 1 | shoulder_pan | 0 | 0 | 0.0304 | +90 |
| 2 | shoulder_lift | −76.032 | −0.01828 | 0.1160 | 0 |
| 3 | elbow_flex | +73.825 | 0 | 0.1350 | 0 |
| 4 | wrist_flex | −87.793 | +0.01810 | 0 | +90 |
| 5 | wrist_roll | +178.792 | −0.15923 | 0 | 0 |

Base frame: origin (0.0388, 0, 0.1166), z pointing **down** along the pan axis.
Tool: 7.9 mm from the last DH frame to the tip site.

Reading the table back as physical facts about the robot:

- Row 1: $a_1 = 3.04$ cm is the horizontal offset between the pan axis and the shoulder pitch axis;
  $\alpha_1 = 90°$ is the turn from a vertical axis to a horizontal one.
- Rows 2 and 3: $\alpha = 0$ means "next axis parallel to this one" — our three pitch joints.
  $a_2, a_3$ are the upper arm and forearm lengths, the same 11.6 cm and 13.5 cm as lesson 01.
- Row 4: $a_4 = 0$ says the wrist pitch and roll axes **intersect**. That's a small mechanical
  gift: it makes IK easier, because position and orientation partly decouple at the wrist.
- $d_2 = -1.83$ cm and $d_4 = +1.81$ cm are the lateral offsets. They nearly cancel, which is why
  the tip ends up almost exactly on the arm's plane (lesson 01, experiment A).
- The **θ offsets** exist because the SO-101's zero ("middle of each joint's range") is not the
  zero the DH convention picks. Always check what "zero" means before trusting someone's table.

`test_dh_table_reproduces_fk` confirms the table gives the same pose as our FK to 3e-15 m, so this
is an exact description of the robot, not an approximation.

### Caveats
- **DH tables aren't unique.** Different authors choose different frames and get different tables
  for the same robot.
- **Two conventions exist:** classic (above) and "modified"/Craig's, which reorders the factors to
  $R_x \mathrm{Trans}_x R_z \mathrm{Trans}_z$ and shifts indices. Mixing them silently gives wrong
  answers. Always check which one a source uses.
- **Simulators don't use DH** (MuJoCo, URDF and ROS all use general offset + quaternion) because DH
  can't describe branching structures like two arms on a torso, or bodies whose frames you want
  elsewhere for other reasons.

So: know how to read DH, use general transforms in code.

---

## 6. Exercises

1. Without running code: what is $R_z(90°)$ applied to the point (1, 0, 0)? And $R_y(90°)$ applied
   to (0, 0, 1)?
2. Verify property 1 for `rot_y(0.7)` in Python: check `R.T @ R` and `np.linalg.det(R)`.
3. Compute $R_z(90°)R_x(90°)$ and $R_x(90°)R_z(90°)$ in Python. Are they equal? Describe in words
   what each one does to the vector (0, 0, 1).
4. Using the hand method in section 4, compute the shoulder_lift axis position for q₁ = −45°, then
   check with `Chain.frames`.
5. In the DH table, why is row 4's $a = 0$ good news for inverse kinematics?
6. The tip frame's **x** axis is the approach direction (out of the jaws). What must it equal for a
   top-down grasp? Now try to find joint angles that put the tip at (0.22, 0.08, 0.10) m *and*
   point it that way, by sampling random joint angles inside the limits. Sample 1000, then 100 000,
   and report the best position error each time. (This is a terrible IK algorithm; lesson 03 fixes
   it.)

<details>
<summary>Answers</summary>

1. $R_z(90°)(1,0,0) = (0, 1, 0)$ — x swings onto y. $R_y(90°)(0,0,1) = (1, 0, 0)$ — z swings onto x
   (the cyclic order z → x is why $R_y$'s signs look flipped).
2. `R.T @ R` is the identity to ~1e-16, determinant 1.
3. Not equal. $R_z(90°)R_x(90°)$ sends (0,0,1) to (1, 0, 0); $R_x(90°)R_z(90°)$ sends it to
   (0, −1, 0). Rotations don't commute. Read each product right to left: the rightmost rotation
   acts first.
4. Build $R_1R_z(-45°) = \begin{bmatrix}-0.707&-0.707&0\\-0.707&0.707&0\\0&0&-1\end{bmatrix}$,
   apply it to offset₂ and add offset₁: **(0.0733, 0.0086, 0.1166)**. The height is unchanged, and
   this time y came out positive — negative q₁ swings the arm the other way.
5. Intersecting wrist axes mean the wrist's rotation doesn't move the wrist centre. You can solve
   for the arm's position using the wrist centre first, then handle orientation separately
   (the classic "kinematic decoupling").
6. It must be (0, 0, −1), straight down. Pointing alone is easy — random search lands within
   0.02° in 1000 samples, because one direction constraint on 5 DOF leaves a huge set of solutions.
   Position *and* direction together is a different story: the best of 1000 samples was 49 mm off,
   and 111 000 samples only got to 21 mm with the approach 12° off. Random search dies because
   5 dimensions sampled at even 10 points each is 100 000 poses just to get a coarse grid. Lesson 03
   solves the same problem exactly, in microseconds, with a triangle.
</details>

---

## 7. Takeaways
- A rotation matrix's columns are the child frame's axes in the parent's coordinates; $R^{-1} = R^{\mathsf T}$.
- Homogeneous 4×4 transforms turn "rotate and translate" into one matrix product, so chains of links
  multiply.
- SO-101 FK: fixed offset, fixed rotation, then $R_z(q_i)$, five times, then the tool offset.
- Our implementation agrees with MuJoCo to ~1e-16 m, and the DH table agrees to ~1e-15 m.
- DH compresses each link to 4 numbers by fixing where the frames go. Read it, but don't build with it.

**Next — Lesson 03: Inverse kinematics.** Given a cube at (x, y) with yaw ψ, find the joint angles
that put the gripper above it pointing down. Geometric solution for the pan angle plus a planar 3R
triangle (law of cosines), the reachable-workspace map, and what to do when a target is
unreachable.
