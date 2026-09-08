# Class 4 — Robot Architecture & Motion Retargeting
**Sports Biomechanics Analysis with AI Vision & Robotics**
*Duration: 1 hour | Phase 4 of 7*

---

## Learning Objectives

By the end of this class you will be able to:
- Trace one video frame through the whole pipeline and name the data type at every hop
- Describe exactly what YOLO-pose extracts: 17 `(x, y, confidence)` keypoints in **image pixels**, one person, one frame
- Explain why a joint angle measured from pixels is a **projection** of the true 3-D angle, and which robot DOFs a given camera view can and cannot observe
- Read a Unitree H1 URDF joint block field by field: `origin`, `axis`, `limit`, `parent`/`child`, joint type
- Draw the H1 kinematic tree and state which of its 19 revolute joints we drive
- Define a **motion-retargeting map**: correspondence, range alignment (`scale`, `offset`), sign convention, and how undriven joints are handled
- Explain why `robot_state_publisher` needs the *full* joint vector every frame

---

## 0. Homework Review (5 min)

Quick check from Class 3:
- What were your personal EAR values (eyes open vs closed) and MAR values (closed vs jaw-dropped)? Did the defaults `0.12` / `0.55` fit?
- Your blink-counter: blinks/min at rest vs while concentrating on the workout video?
- Heart-rate validation — which of the five conditions (still / post-exercise / dim light / nodding / still) broke the rPPG measurement, and did that match the Section 5 failure modes?
- Smoothing study: what did the jaw joint do at `alpha = 1.0` vs `0.05` while you talked?

---

## 1. The Full Architecture — From Pixels to Joints (8 min)

Everything we have built so far is one **unidirectional data pipeline**. Class 4 is where you see the whole thing at once.

```
┌────────────┐  BGR frame   ┌────────────┐  (17,3) px    ┌────────────┐  {canonical:rad}  ┌──────────────┐
│  Camera /  │─────────────▶│ PoseDetector│──────────────▶│   mapper   │──────────────────▶│  remap_joints │
│  video     │  H×W×3 uint8 │ (YOLOv8-pose)│  keypoints    │ law of cos │  10 joint angles  │ (per-robot)   │
└────────────┘              └────────────┘               └────────────┘                   └──────┬───────┘
                                                                                                │ {urdf_joint:rad}
                                                                                                ▼
   RViz2  ◀──────  /tf  ◀──────  robot_state_publisher  ◀──────  /joint_states  ◀──────  RobotPublisher
   (render)      TFMessage        (forward kinematics)         sensor_msgs/JointState      (+ /robot_description)
```

### Module responsibilities

| Stage | File | Input | Output |
|-------|------|-------|--------|
| Detect | `src/detector.py` | BGR frame | `(17, 3)` array `[x, y, conf]`, or `None` |
| Map | `src/mapper.py` → `keypoints_to_joints()` | keypoints | `dict` of **10 canonical** joint angles (rad) |
| Retarget | `src/mapper.py` → `remap_joints()` + `src/robot_registry.py` → `JOINT_MAPS` | canonical dict | `dict` keyed by the **target URDF's** joint names |
| Publish | `src/ros_publisher.py` → `RobotPublisher` | retargeted dict | `/joint_states` (all movable joints) + `/robot_description` (latched) |
| Kinematics | `robot_state_publisher` (ROS 2 stock node) | URDF + `/joint_states` | `/tf`, `/tf_static` |
| Render | `rviz2` | `/robot_description` + `/tf` | 3-D view |
| Orchestrate | `src/main.py` | CLI args | runs the loop |

### The ROS 2 graph

```
/pose_mimic_publisher ──/robot_description (latched)──▶ /robot_state_publisher ──/tf,/tf_static──▶ /rviz
        │                                                        ▲
        └────────────────── /joint_states ───────────────────────┘
```

Two things publish `/robot_description` with the **same** URDF text: our `RobotPublisher` (so the topic exists even without a launch script) and `robot_state_publisher` (from its `robot_description` parameter). `robot_state_publisher` is the piece that turns joint **angles** into link **poses** — it walks the URDF tree and composes the transforms (forward kinematics). RViz never does kinematics; it only *looks up* `/tf`.

> Key idea: **`/joint_states` is the mirroring channel.** Everything downstream is deterministic given the URDF. All of our engineering effort goes into producing good numbers on that one topic.

---

## 2. The Motion Signal — What YOLO Actually Extracts (10 min)

### The raw output

`PoseDetector.detect()` returns **one** array of shape `(17, 3)` — the most confident person in the frame (`detector.py` takes the max-confidence detection and drops the rest). Each row is:

```
[ x_pixels , y_pixels , confidence ]     x ∈ [0, W],  y ∈ [0, H],  conf ∈ [0, 1]
```

Image coordinates: **origin top-left, x → right, y → down**. This is *not* a physics frame — y points *down*, and there is no z.

### COCO-17 layout

```
            0 nose
      1 ●───┴───● 2          (eyes)
    3 ●           ● 4        (ears)
        5 ●───────● 6        shoulders
          │       │
        7 ●       ● 8        elbows
          │       │
        9 ●       ● 10       wrists
       11 ●───────● 12       hips
          │       │
       13 ●       ● 14       knees
          │       │
       15 ●       ● 16       ankles
```

We use **12 of the 17** — indices 5–16. The face 5 (0–4) drove Class 3; they are dead weight here.

```python
# src/detector.py
KP = {"left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
      "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
      "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}
```

### Properties — and limitations — of this signal

| Property | Consequence for retargeting |
|----------|----------------------------|
| **2-D projection** of a 3-D pose | one rotational DOF per joint is lost; angles are foreshortened |
| **Pixel units**, not metres | scale depends on subject distance; only *ratios* and *angles* are stable, never lengths |
| **Camera frame**, not world/robot frame | "up" in the image ≠ robot `+z`; a tilted camera tilts every angle |
| **Per-frame, independent** | no temporal model → jitter passes straight through to the joints |
| **`confidence` varies** | occluded/blurred joints drop below `0.3` and must be gated (`mapper._visible`) |
| **Left/right assigned from the image** | when the subject turns their back, YOLO swaps the labels — the robot's limbs cross |
| **One person only** | two athletes in frame → you track whichever scores higher that frame |

### A real sample (workout video, `--robot h1`)

`keypoints_to_joints()` for three consecutive frames, canonical names, radians:

```
frame  L_sh_pitch  L_elbow  R_sh_pitch  R_elbow  L_knee  R_knee
  41      0.32       2.35       0.28      2.35     2.35    2.35     ← arms near torso, legs straight
  63      1.18       0.72       1.02      0.94     2.10    2.31     ← arms raised & bent
  88      0.46       2.35       1.55      0.10     2.35    2.28     ← right arm up & fully folded
```

Read the ranges, not the absolute numbers: `L_elbow` swings `0.72 → 2.35` (folded → straight), `R_sh_pitch` swings `0.28 → 1.55` (arm down → arm out). Those spans are what we align to the robot in Section 5.

---

## 3. Camera Angle & Reference Frames (8 min)

### The projection problem

`mapper._angle_at_b(a, b, c)` computes the angle at vertex `b` of the triangle `a–b–c` **in the image plane**:

```
                     BA · BC
θ = arccos( ───────────────────────── )        θ ∈ [0, π],  unsigned
              ‖BA‖ · ‖BC‖
```

`BA` and `BC` are 2-D pixel vectors. So `θ` is the *true* joint angle only when the limb moves **parallel to the image plane**. Any motion toward or away from the camera is foreshortened, and at the extreme (limb pointing straight at the lens) the angle collapses to noise.

### What a given view can observe

| Camera view | Reads well | Nearly invisible |
|-------------|-----------|------------------|
| **Frontal** (our workout video) | shoulder abduction (arm out to the side), elbow flex *in the frontal plane*, knee flex when the leg swings sideways | shoulder flexion (arm forward), hip flexion, anything sagittal |
| **Side / sagittal** | shoulder & hip flexion, knee flex, elbow flex during a curl | abduction, any left/right symmetry, torso twist |
| **45° / three-quarter** | a bit of everything, cleanly of nothing | — |

The workout video is roughly frontal, so we get **shoulder pitch** and **elbow flexion** believably and treat everything else as unobservable. That is a *deliberate scoping decision you make from the camera geometry*, before writing a single mapping line.

### Two consequences of the unsigned `arccos`

1. **No flex/extend distinction.** `arccos` returns the same value whether the elbow bent forward or backward. For a hinge joint that only bends one way (elbow, knee) this is fine. For the shoulder it means we cannot tell "arm forward" from "arm back".
2. **No side.** The thigh tilted 20° left and 20° right give the same number.

Recovering the sign needs the limb *vector*, not just its length ratio — the signed angle of the thigh relative to a body reference axis (e.g. pelvis-down):

```python
def signed_angle(v, ref):                      # v, ref are 2-D vectors
    cross = ref[0]*v[1] - ref[1]*v[0]          # scalar z of the 2-D cross product
    dot   = ref[0]*v[0] + ref[1]*v[1]
    return math.atan2(cross, dot)              # ∈ (−π, π],  sign tells you which way
```

Our `mapper.py` does **not** do this yet — which is exactly why Section 5's map carries a hand-tuned sign per joint. Lab D4 adds a signed joint.

### The frames you are juggling

| Frame | Units | Axes | Origin |
|-------|-------|------|--------|
| **Image** | pixels | x→right, y→**down**, no z | top-left corner |
| **Robot base** (`pelvis` for H1) | metres | x→forward, y→left, z→up (right-handed) | between the hips |
| **Joint frame** (per URDF `<joint>`) | metres/radians | set by `origin rpy`; rotation about `axis` | set by `origin xyz` |

Retargeting is the bridge from the first row to the third.

---

## 4. The H1 Robot — Reading a URDF (12 min)

### URDF in one paragraph

A URDF is an XML tree of **links** (rigid bodies: `visual`, `collision`, `inertial`) connected by **joints**. Each joint names a `parent` and a `child` link, fixes the child's zero pose with `origin xyz rpy`, and — for a `revolute` joint — allows rotation about `axis` between `limit lower` and `limit upper` (radians). `robot_state_publisher` reads this tree plus `/joint_states` and emits one `/tf` transform per link.

### One joint block, annotated

```xml
<joint name="left_elbow_joint" type="revolute">   <!-- revolute = 1-DOF hinge with limits -->
  <origin xyz="0.0185 0 -0.198" rpy="0 0 0"/>     <!-- child sits 198 mm below parent, at zero angle -->
  <parent link="left_shoulder_yaw_link"/>          <!-- kinematic parent -->
  <child  link="left_elbow_link"/>                 <!-- this joint moves the forearm -->
  <axis xyz="0 1 0"/>                              <!-- rotates about +Y of the joint frame -->
  <limit lower="-1.25" upper="2.61"               <!-- radians: ~ -72° to +150° -->
         effort="18" velocity="20"/>               <!-- torque (N·m) & speed caps — ignored by RViz -->
</joint>
```

Sign follows the **right-hand rule** about `axis`: thumb along `+Y`, fingers curl the positive direction.

### H1 kinematic tree (19 revolute joints)

```
pelvis ─┬─ left_hip_yaw → left_hip_roll → left_hip_pitch → left_knee → left_ankle
        ├─ right_hip_yaw → right_hip_roll → right_hip_pitch → right_knee → right_ankle
        └─ torso ─┬─ left_shoulder_pitch → left_shoulder_roll → left_shoulder_yaw → left_elbow
                  ├─ right_shoulder_pitch → right_shoulder_roll → right_shoulder_yaw → right_elbow
                  └─ (imu, logo, d435 camera, mid360 lidar — fixed joints, sensors)
```

`pelvis` is the **root** (it is nobody's child) — that is why `config/robot_view.rviz` must set **Fixed Frame = `pelvis`** for H1, not `base_link`.

### Every H1 revolute joint

| Joint (×2, left/right unless noted) | `axis` | limits (rad) | plane |
|---|---|---|---|
| `*_hip_yaw_joint` | `0 0 1` | −0.43 … 0.43 | transverse (leg turns out/in) |
| `*_hip_roll_joint` | `1 0 0` | −0.43 … 0.43 | frontal (leg out to side) |
| `*_hip_pitch_joint` | `0 1 0` | −3.14 … 2.53 | sagittal (leg forward/back) |
| `*_knee_joint` | `0 1 0` | −0.26 … 2.05 | sagittal (bend) |
| `*_ankle_joint` | `0 1 0` | −0.87 … 0.52 | sagittal (point/flex foot) |
| `torso_joint` (×1) | `0 0 1` | −2.35 … 2.35 | transverse (waist twist) |
| `*_shoulder_pitch_joint` | `0 1 0` | −2.87 … 2.87 | sagittal (arm forward/back) |
| `*_shoulder_roll_joint` | `1 0 0` | L −0.34…3.11 / R −3.11…0.34 | frontal (arm out to side) |
| `*_shoulder_yaw_joint` | `0 0 1` | L −1.3…4.45 / R −4.45…1.3 | transverse (upper-arm twist) |
| `*_elbow_joint` | `0 1 0` | −1.25 … 2.61 | sagittal (bend) |

Note the **asymmetry**: left and right roll/yaw limits are mirror images, because positive rotation about a shared axis lifts the left arm but lowers the right. Retargeting has to respect that (Section 5, step 3).

### Contrast: the built-in `simple` humanoid

| | `robot/humanoid.urdf` (`simple`) | H1 (`h1`) |
|---|---|---|
| Links | 10, box + cylinder + sphere primitives | 25, STL meshes |
| Revolute joints | 10 | 19 |
| Joint names | `left_elbow`, `left_knee`, … (no suffix) | `left_elbow_joint`, … |
| All axes | `1 0 0` (everything pitches about X) | mixed `1 0 0` / `0 1 0` / `0 0 1` |
| Root link | `base_link` | `pelvis` |
| Meshes | none | `package://h1_description/meshes/*.STL` |
| Needs `robot_state_publisher`? | yes (for `/tf`) | yes |
| Needs mesh path resolution? | no | **yes** — `package://` |

That last row is the practical gotcha we hit: on a RoboStack/conda ROS 2, `ROS_PACKAGE_PATH` does **not** resolve `package://h1_description/...`; RViz uses the ament index. `labs/lab_c_mirror.sh` works around it by building a throwaway ament prefix of symlinks and exporting `AMENT_PREFIX_PATH`.

---

## 5. Motion Retargeting — Defining the Mapping (12 min)

**Retargeting** = transferring motion from a *source* skeleton (human, COCO-17, 2-D) onto a *target* skeleton (H1, 19-DOF, 3-D) that has different proportions, joint definitions, and degrees of freedom. It is never a 1:1 copy. Our map has four decisions.

### Step 1 — Correspondence: which keypoints drive which joint

| Human triplet (COCO indices) | `mapper` measures | Canonical name | → H1 joint |
|---|---|---|---|
| hip → shoulder → elbow  (11/12, 5/6, 7/8) | angle at the shoulder | `*_shoulder_pitch` | `*_shoulder_pitch_joint` |
| shoulder → elbow → wrist  (5/6, 7/8, 9/10) | angle at the elbow | `*_elbow` | `*_elbow_joint` |
| hip → knee → ankle  (11/12, 13/14, 15/16) | angle at the knee | `*_knee` | `*_knee_joint` |

Six joints driven. `mapper.py` also emits `*_hip_pitch` (derived from the knee, unreliable) and leaves `*_shoulder_roll` at `0.0` — we do **not** forward those to H1.

### Step 2 — Range alignment: `target = scale · measured + offset`, then clamp

`mapper` produces **unsigned interior angles** that were then clamped to the *simple* robot's limits, so in practice:

| Canonical measure | ≈ value: limb straight | ≈ value: limb folded / arm down |
|---|---|---|
| `*_elbow`, `*_knee` | `2.35` (π, clamped) | `→ 0` |
| `*_shoulder_pitch` | `→ 1.57` (arm horizontal) | `→ 0` (arm at side) |

H1's zero pose is a straight-limbed stand. So we shift each measure to sit at the H1 joint's zero and let flexion drive it:

```python
# src/robot_registry.py
JOINT_MAPS["h1"] = {
    #  canonical            (h1 joint,                      scale, offset)
    "left_shoulder_pitch":  ("left_shoulder_pitch_joint",  -1.0, 0.0),
    "right_shoulder_pitch": ("right_shoulder_pitch_joint", -1.0, 0.0),
    "left_elbow":           ("left_elbow_joint",           -1.0, 2.35),  # 2.35→0 straight, 0→2.35 folded
    "right_elbow":          ("right_elbow_joint",          -1.0, 2.35),
    "left_knee":            ("left_knee_joint",            -1.0, 2.35),  # clamped to 2.05 by the limit
    "right_knee":           ("right_knee_joint",           -1.0, 2.35),
}
```

```python
def remap_joints(joints, joint_map):
    if not joint_map:
        return joints                                    # 'simple' → names already match
    out = {}
    for canonical, angle in joints.items():
        if canonical in joint_map:
            target, scale, offset = joint_map[canonical]
            out[target] = scale * angle + offset
    return out
```

The final **clamp to the URDF limit** happens in `RobotPublisher.publish_joints()` — e.g. a knee value of `2.35` is clipped to H1's `2.05` upper limit. Never trust the map to stay in range; always clamp against the target URDF.

### Step 3 — Sign / direction

Because the measured angle is **unsigned**, the direction is a free choice we make by eye: launch RViz, run the video, and if a joint moves the wrong way, flip its `scale` sign (`-1.0 ↔ 1.0`) in `JOINT_MAPS`. That is the entirety of the calibration loop right now. A full retargeter would instead compute the *signed* limb angle (Section 3) against a torso reference and drop the guesswork.

### Step 4 — Undriven DOFs

H1 has 19 revolute joints; we drive 6. The other 13 (`hip_yaw`, `hip_roll`, `hip_pitch`, `ankle`, `torso`, `shoulder_roll`, `shoulder_yaw`) are held at **0.0**. So the mirrored H1 cannot twist its waist, step sideways, shrug, or rotate a forearm — by design, because a frontal monocular view cannot observe those motions anyway (Section 3).

### Step 5 — Publish the *full* joint vector every frame

```python
# src/ros_publisher.py  (RobotPublisher.publish_joints, abridged)
names = self._joint_order                    # ALL 19 movable joints, parsed from the URDF
positions = []
for name in names:
    value = float(joint_angles.get(name, 0.0))       # 0.0 for the 13 we don't drive
    lo, hi = self._limits.get(name, (None, None))
    if lo is not None:
        value = max(lo, min(hi, value))              # clamp to URDF limit
    positions.append(value)
msg.name, msg.position = names, positions
```

Why every joint, every frame: **`robot_state_publisher` only emits `/tf` for a joint after it has seen that joint's name in a `JointState` message.** Send only the 6 driven joints and the other 13 links (hips, torso, the entire lower body) get *no transform* — RViz reports "No transform from [pelvis]" and renders nothing. This is the single most common reason "the robot doesn't show up".

### What this map does *not* do

- **Proportions** — H1's arm/leg length ratios ≠ the athlete's; joint *angles* transfer, limb *reach* does not.
- **Root motion** — `pelvis` is pinned at the origin; the robot cannot walk, jump, or lean.
- **Balance / IK** — RViz shows kinematics only. A physical H1 driven by these angles would fall over; it needs a whole-body controller.
- **Self-collision** — nothing checks that the retargeted arms don't pass through the torso.
- **Temporal smoothing** — jitter in the keypoints goes straight to the joints (add EMA + velocity caps, like the Class 3 face pipeline).

---

## 6. Challenges — Why This Is Hard (5 min)

1. **2-D → 3-D is under-determined.** Every joint loses one rotational DOF to projection. The honest fix is a monocular 3-D lift (VideoPose3D, MotionBERT) or a second camera — everything else is a heuristic.
2. **Camera placement decides the curriculum.** A frontal camera can't see sagittal motion; a side camera can't see abduction. Document the rig; scope the map to observable DOFs.
3. **Skeleton mismatch.** COCO-17 has no spine, clavicle, or 3-DOF shoulder; H1 has torso yaw and a 3-DOF shoulder. Some robot DOFs have *no* human source; some human motions have *no* robot DOF.
4. **No dynamics in RViz.** Kinematic mirroring looks great and would collapse on hardware. Balance, contact, and actuator limits are a separate (large) problem.
5. **Manual sign/offset calibration.** We flip `scale` by eye. Production systems capture a **T-pose** reference frame and solve the offsets.
6. **Latency & jitter.** Raw retargeting is twitchy. EMA smoothing and capped joint velocity *improve* perceived quality even though they add lag.
7. **Left/right label swaps** when the athlete turns away from the camera — the robot's arms visibly cross. Gate on torso orientation or hold last-good.

> Professional path: fit a parametric body model (**SMPL / SMPL-X**) to the video, then retarget the model's joints to the robot inside a physics sim (**Isaac Lab**, **MuJoCo**) with an IK/controller in the loop. Humanoid teleop papers to look up: **H2O**, **OmniH2O**, **ExBody**, **PHC**.

---

## 7. Lab Exercises (10 min)

### Lab D1 — Inspect the motion signal (no ROS)

File: `labs/lab_d1_motion_signal.py` — run YOLO + `keypoints_to_joints()` on the workout video, print the 10 canonical angles per frame, and plot `left_elbow` and `right_shoulder_pitch` over time.

```bash
python labs/lab_d1_motion_signal.py --video labs/videos/workout.mp4 --plot
```

Expected: elbow traces that swing between ~0 and ~2.35 as the athlete curls; gaps where `confidence < 0.3`. Which joint is noisiest, and does that match the camera view?

### Lab D2 — Read the H1 URDF

File: `labs/lab_d2_urdf_inspect.py` — parse any robot's URDF and print the kinematic tree, every revolute joint's `axis` + limits, the root link, and a ✓/✗ for whether the current `JOINT_MAPS` drives it.

```bash
python labs/lab_d2_urdf_inspect.py --robot h1
python labs/lab_d2_urdf_inspect.py --robot simple
```

### Lab D3 — Tune the retarget in RViz

```bash
# Terminal 1
./labs/lab_c_mirror.sh h1
# Terminal 2
python src/main.py --robot h1
```

Find a joint that bends the wrong way. Flip its `scale` sign in `src/robot_registry.py` → `JOINT_MAPS["h1"]`, re-run Terminal 2, confirm. Record which joints you flipped.

### Lab D4 — Add a driven joint

Extend `JOINT_MAPS["h1"]` with **one** new joint using a *signed* angle:
- `torso_joint` from the shoulder-line tilt: `atan2(R_shoulder.y − L_shoulder.y, R_shoulder.x − L_shoulder.x)`, or
- `left_hip_pitch_joint` from the signed angle of the thigh vector (hip→knee) vs the pelvis-down vector (Section 3 `signed_angle`).

Add the computation to `mapper.py`, forward it through `remap_joints`, clamp to the URDF limit, and verify direction in RViz.

---

## 8. Take-Home Tasks

1. **Data-type audit.** For one frame, write down the concrete Python type and shape at each of the 7 pipeline stages in Section 1 (e.g. "`np.ndarray (17,3) float32`"). Where could a `None` appear, and what happens downstream?

2. **Observability table.** Watch the workout video. For each of the 6 driven H1 joints, rate 0–3 how well the frontal camera observes that motion, and justify each score from the limb's movement plane.

3. **Range calibration.** Run Lab D1 and record the actual min/max of `left_elbow` and `left_shoulder_pitch` over the whole video. Re-derive the `scale`/`offset` for those two H1 joints from *your measured* range (not the theoretical `0..2.35`) and compare to the values in `JOINT_MAPS`.

4. **Sign recovery.** Implement `signed_angle()` from Section 3 for the elbow (forearm vector vs upper-arm vector). Does the sign flip when the athlete's arm crosses the body? Plot it against the unsigned `_angle_at_b` value.

5. **URDF reading.** In `h1.urdf`, follow the chain `pelvis → left_elbow_link`. List every joint, its `origin xyz`, and its `axis`. What is the total translational offset from `pelvis` to `left_elbow_link` at the zero pose?

6. *(Stretch)* Write a `JOINT_MAPS` entry for **`g1`** (Unitree G1, 23-DOF). Run `python labs/lab_d2_urdf_inspect.py --robot g1` to get the joint names, map the same 6 joints, launch `./labs/lab_c_mirror.sh g1`, and mirror the video onto G1.

---

## Resources

| Resource | URL |
|----------|-----|
| Unitree `unitree_ros` (H1/G1 URDFs + meshes) | https://github.com/unitreerobotics/unitree_ros |
| URDF XML spec | https://wiki.ros.org/urdf/XML |
| URDF tutorial (building a visual robot) | https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html |
| `robot_state_publisher` | https://github.com/ros/robot_state_publisher |
| `tf2` design & conventions | https://docs.ros.org/en/humble/Concepts/Intermediate/About-Tf2.html |
| REP-103 — units & coordinate conventions | https://www.ros.org/reps/rep-0103.html |
| REP-105 — coordinate frames for mobile platforms | https://www.ros.org/reps/rep-0105.html |
| `sensor_msgs/JointState` | https://docs.ros2.org/latest/api/sensor_msgs/msg/JointState.html |
| COCO keypoint format | https://cocodataset.org/#keypoints-2020 |
| YOLOv8 pose docs | https://docs.ultralytics.com/tasks/pose/ |
| VideoPose3D (monocular 3-D lift) | https://github.com/facebookresearch/VideoPose3D |
| MotionBERT (3-D pose) | https://github.com/Walter0807/MotionBERT |
| SMPL / SMPL-X body model | https://smpl-x.is.tue.mpg.de/ |
| H2O — human-to-humanoid teleop | https://human2humanoid.com/ |
| OmniH2O | https://omni.human2humanoid.com/ |
| ExBody — expressive whole-body control | https://expressive-humanoid.github.io/ |
| Isaac Lab (retargeting + RL in sim) | https://isaac-sim.github.io/IsaacLab/ |

---

## Class Schedule Overview

| Class | Topic |
|-------|-------|
| 1 | System Architecture, Setup & Toolchains |
| 2 | Keypoint Tracking, YOLO & CNN Deep Dive |
| 3 | Facial Expression Recognition & Robot Face Mapping |
| **4** | **Robot Architecture & Motion Retargeting** ← you are here |
| 5 | ROS 2 Publisher & Joint State Messages |
| 6 | RViz2 Visualization & URDF Tuning |
| 7 | Sports Analysis — Tennis Serve / Golf Swing Scoring |
