# Class 3 — Facial Expression Recognition & Robot Face Mapping
**Sports Biomechanics Analysis with AI Vision & Robotics**
*Duration: 1 hour | Phase 3 of 7*

---

## Learning Objectives

By the end of this class you will be able to:
- Explain why YOLO's 5 sparse face keypoints are not enough for expression recognition
- Detect 68 dense facial landmarks with dlib, using YOLO's face keypoints as a free face locator
- Compute geometric expression features: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), eyebrow raise
- Compare geometric (dlib) vs deep-learning (DeepFace / FER) expression recognition and choose the right tool
- Publish expression intensities as `sensor_msgs/JointState` to drive a robot face in ROS 2
- Apply smoothing and hysteresis so the robot face doesn't flicker
- Explain how remote photoplethysmography (rPPG) measures heart rate from ordinary face video

---

## 0. Homework Review (5 min)

Quick check from Class 2:
- Which 3 keypoints had the lowest average confidence in `lab_a2_confidence_analysis.py`?
- How much slower was `yolov8x-pose` than `yolov8n-pose` per frame?
- What was the SPARC score for the right wrist — and where did jerk spike?

---

## 1. From Body Pose to Faces — Why 5 Keypoints Aren't Enough (5 min)

YOLO-pose gives us exactly **5 face keypoints**: nose (0), eyes (1, 2), ears (3, 4).

```
       1   2      ← eyes
        \ /
         0        ← nose
       /   \
      3     4     ← ears
```

That is enough to know **where the face is** and roughly **where it's pointing** — but expressions live in what these 5 points can't see:

| Expression cue | Needed landmarks | In YOLO's 5? |
|----------------|------------------|--------------|
| Smile / frown | Mouth corners, lips | ❌ |
| Surprise | Eyebrow position, jaw drop | ❌ |
| Blink / squint | Eyelid contours | ❌ |
| Head pose | Eyes + nose + ears | ✅ (coarse) |

**The trick**: we don't throw YOLO away. Its 5 keypoints give us a **free, fast face region** — we crop that region and hand it to a *dense* landmark detector. This is a classic **coarse-to-fine pipeline**:

```
┌──────────┐   ┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌─────────┐
│  Webcam  │──▶│ YOLO Pose     │──▶│ Face crop from  │──▶│ dlib 68-landmark │──▶│ Feature │──▶ ROS 2
│          │   │ (17 kps)      │   │ eye/nose/ear kps│   │ shape predictor  │   │ extract │    face robot
└──────────┘   └───────────────┘   └─────────────────┘   └──────────────────┘   └─────────┘
                body → Class 2/4          this class            this class        this class
```

Bonus: because YOLO already localized the face, we can **skip dlib's own face detector** (its slowest stage) entirely.

---

## 2. Dense Facial Landmarks with dlib (15 min)

### What is dlib?

dlib is a C++ machine learning library with Python bindings. Its facial landmark detector implements **Kazemi & Sullivan's "One Millisecond Face Alignment" (2014)** — an ensemble of regression trees that predicts 68 landmarks in ~1 ms per face. No GPU, no deep network.

> dlib: http://dlib.net | Paper: https://ieeexplore.ieee.org/document/6909637

### Install

```bash
# Option A — conda (easiest, prebuilt binaries)
mamba install -c conda-forge dlib -y

# Option B — pip (compiles from source; needs cmake)
pip install cmake
pip install dlib

# Download the 68-landmark model (~95 MB, one time)
curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat labs/models/
```

### The 68-landmark map (memorize the ranges)

```
 0–16   Jaw line (17 pts)                    17-21   22-26
17–21   Right eyebrow                      ~~~~~   ~~~~~
22–26   Left eyebrow                      36-41    42-47
27–35   Nose bridge + nostrils             (eye)  (eye)
36–41   Right eye (6 pts → EAR!)              27
42–47   Left eye  (6 pts → EAR!)              |
48–59   Outer mouth (12 pts)                31-35
60–67   Inner mouth (8 pts → MAR!)        48 ─────── 54
                                             \ 60-67 /
                                              ~~~~~~
                                              (mouth)
```

### YOLO crop → dlib landmarks (the core integration)

```python
import cv2
import dlib
import numpy as np
from ultralytics import YOLO

pose_model = YOLO("yolov8n-pose.pt")
predictor  = dlib.shape_predictor("labs/models/shape_predictor_68_face_landmarks.dat")

def face_rect_from_yolo(kps, frame_shape):
    """Build a face bounding box from YOLO's 5 face keypoints."""
    face_pts = kps[:5]                          # nose, eyes, ears
    valid = face_pts[face_pts[:, 2] > 0.3]      # keep confident points
    if len(valid) < 3:
        return None
    cx, cy = valid[:, 0].mean(), valid[:, 1].mean()
    # Face size ≈ 2.2x the spread of the keypoints (ears set the width)
    size = 2.2 * max(valid[:, 0].ptp(), 40)
    x1, y1 = int(cx - size/2), int(cy - size/2)
    x2, y2 = int(cx + size/2), int(cy + size/2 + 0.3*size)  # extend down to chin
    h, w = frame_shape[:2]
    return dlib.rectangle(max(x1,0), max(y1,0), min(x2,w-1), min(y2,h-1))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = pose_model(frame, conf=0.5, verbose=False)
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        kps = results[0].keypoints.data[0].cpu().numpy()
        rect = face_rect_from_yolo(kps, frame.shape)
        if rect is not None:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = predictor(gray, rect)               # ← 68 landmarks, ~1 ms
            for i in range(68):
                p = shape.part(i)
                cv2.circle(frame, (p.x, p.y), 1, (0, 255, 0), -1)
    cv2.imshow("landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

Note: `shape_predictor` is **not** a detector — it *refines* a given rectangle into landmarks. Feeding it YOLO's rectangle is exactly how it's meant to be used.

---

## 3. Recognizing Expressions — Geometry vs Deep Learning (10 min)

### Approach A — Geometric features from landmarks (dlib, recommended)

Compute **ratios** from landmark distances. Ratios are scale-invariant — they work whether the face is near or far.

**Eye Aspect Ratio (EAR)** — Soukupová & Čech, 2016. Detects blinks/squints:

```
        p2  p3
      /        \
   p1 ─────────── p4        EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)
      \        /
        p6  p5               open eye ≈ 0.30   closed ≈ 0.05
```

**Mouth Aspect Ratio (MAR)** — same idea on inner-mouth points 60–67. Jaw drop (surprise) ≈ MAR > 0.6; closed mouth ≈ 0.0–0.1.

**Smile ratio** — mouth width (48↔54) / jaw width (2↔14). Smiling stretches the mouth corners outward.

**Eyebrow raise** — vertical distance from eyebrow centroid (17–26) to eye centroid, normalized by face height.

```python
import numpy as np

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def to_np(shape):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

def eye_aspect_ratio(pts, idx):          # idx = range(36,42) or range(42,48)
    p = pts[list(idx)]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]))

def mouth_aspect_ratio(pts):
    p = pts[60:68]                        # inner mouth
    return (dist(p[1], p[7]) + dist(p[2], p[6]) + dist(p[3], p[5])) / (2.0 * dist(p[0], p[4]))

def smile_ratio(pts):
    return dist(pts[48], pts[54]) / dist(pts[2], pts[14])

def brow_raise(pts):
    brow = pts[17:27].mean(axis=0)
    eyes = pts[36:48].mean(axis=0)
    face_h = dist(pts[27], pts[8])        # nose bridge → chin
    return (eyes[1] - brow[1]) / face_h
```

A simple **rule-based classifier** on top:

```python
def classify(ear, mar, smile, brow, baseline):
    """baseline = per-user neutral values captured at startup."""
    if mar > 0.55 and brow > baseline["brow"] * 1.15:
        return "surprise"
    if smile > baseline["smile"] * 1.10 and mar < 0.35:
        return "happy"
    if ear < 0.12:
        return "eyes_closed"
    if brow < baseline["brow"] * 0.85:
        return "frown"
    return "neutral"
```

> The **baseline calibration** step (capture 30 neutral frames at startup, average the features) is what makes this robust across different faces. Everyone's resting eyebrow height is different.

### Approach B — Deep learning classifiers (DeepFace, FER) — and why we skip them

Libraries like **DeepFace** and **FER** run a CNN trained on FER-2013-style datasets and output **7 discrete emotion probabilities** (happy, sad, angry, surprise, fear, disgust, neutral) from a single call like `DeepFace.analyze(frame, actions=["emotion"])`. Tempting — but we won't use them in this course:

- They pull in **TensorFlow (~2 GB)**, and on **macOS (especially Apple Silicon)** the install routinely breaks: you need the separate `tensorflow-macos` / `tensorflow-metal` wheels, and DeepFace's pinned dependencies (mtcnn, retina-face) often conflict with them. If you already fought this battle and lost — that's expected, not your fault.
- Even when they run, inference costs 100–300 ms/frame on CPU — our real-time budget is gone.
- Most importantly, discrete labels are the **wrong output type** for driving a robot (see the table below).

### Which one for this project? (the deep-learning vs dlib decision)

| Criterion | dlib geometric | DeepFace | FER (`pip install fer`) |
|-----------|---------------|----------|------------------------|
| Speed (CPU) | **~1 ms/face** | 100–300 ms/face | 50–150 ms/face |
| Dependencies | small C++ lib | TensorFlow (~2 GB) | TensorFlow + mtcnn |
| Output | **continuous ratios** | 7 discrete labels | 7 discrete labels |
| Interpretable | ✅ you see every number | ❌ black box | ❌ black box |
| Robot actuation | **direct** — ratios → joint angles | needs label → pose lookup | needs label → pose lookup |
| Accuracy on subtle expressions | medium (rules) | medium (FER-2013 ceiling ≈ 65-75%) | medium |
| Per-user calibration | easy | not possible | not possible |

**For driving a robot face, dlib wins for one decisive reason**: robots need **continuous intensities**, not labels. A servo eyebrow needs "raise eyebrow 0.63 of max", not "surprise". Geometric ratios *are already* actuation signals. A discrete label like "happy" forces you to play a canned animation — the robot stops mirroring *you*.

> Interested in the "proper" middle ground? Look up **FACS Action Units** (Ekman) — the anatomical vocabulary of facial movement (AU1 = inner brow raiser, AU12 = lip corner puller…). Libraries like **OpenFace 2.0** and **py-feat** output AU intensities directly — the professional version of what our ratios approximate.

---

## 4. Mapping Expressions to a Robot Face in ROS 2 (10 min)

### The robot side — face joints in URDF

An expressive robot face (InMoov head, custom servo face, or the display face on a Unitree G1) is just **more joints**. A minimal servo face:

```xml
<!-- robot/face.urdf (excerpt) -->
<joint name="jaw_joint" type="revolute">
  <parent link="head"/> <child link="jaw"/>
  <axis xyz="1 0 0"/>
  <limit lower="0.0" upper="0.5" effort="1" velocity="2"/>   <!-- rad -->
</joint>
<joint name="left_brow_joint" type="revolute">
  <parent link="head"/> <child link="left_brow"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.3" upper="0.3" effort="1" velocity="2"/>
</joint>
<!-- right_brow_joint, left_eyelid_joint, right_eyelid_joint,
     left_lip_corner_joint, right_lip_corner_joint ... -->
```

### The mapping table — features → joints

This is the heart of the class. Each geometric feature maps **linearly** to a joint, after normalizing against the calibrated baseline:

| dlib feature | Robot joint | Mapping |
|--------------|-------------|---------|
| MAR (0 → 0.7) | `jaw_joint` (0 → 0.5 rad) | `jaw = clamp(MAR / 0.7) * 0.5` |
| brow_raise vs baseline | `left/right_brow_joint` | `brow = clamp((raise/base − 1) * 3) * 0.3` |
| EAR (0.05 → 0.30) | `left/right_eyelid_joint` | `lid = 1 − clamp((EAR−0.05)/0.25)` |
| smile_ratio vs baseline | `lip_corner_joints` | `smile = clamp((ratio/base − 1) * 5) * 0.2` |

### The ROS 2 node

Same pattern as our body pipeline — a publisher on `/face/joint_states`:

```python
# src/face_publisher.py (excerpt)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

FACE_JOINTS = ["jaw_joint", "left_brow_joint", "right_brow_joint",
               "left_eyelid_joint", "right_eyelid_joint",
               "left_lip_corner_joint", "right_lip_corner_joint"]

class FacePublisher(Node):
    def __init__(self):
        super().__init__("face_expression_publisher")
        self.joint_pub = self.create_publisher(JointState, "/face/joint_states", 10)
        self.label_pub = self.create_publisher(String, "/face/expression", 10)
        self.smoothed  = {j: 0.0 for j in FACE_JOINTS}
        self.alpha     = 0.3          # EMA smoothing factor

    def publish_face(self, targets: dict, label: str):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for j in FACE_JOINTS:
            # Exponential moving average — the anti-flicker filter
            self.smoothed[j] = (1 - self.alpha) * self.smoothed[j] \
                             + self.alpha * targets.get(j, 0.0)
            msg.name.append(j)
            msg.position.append(self.smoothed[j])
        self.joint_pub.publish(msg)

        lbl = String(); lbl.data = label
        self.label_pub.publish(lbl)
```

Two topics, two audiences:
- `/face/joint_states` (`sensor_msgs/JointState`) — **continuous**, for the servo controller / RViz2. This is the mirroring channel.
- `/face/expression` (`std_msgs/String`) — **discrete label**, for high-level behavior ("athlete looks strained → suggest rest").

### Verify in RViz2 / CLI

```bash
ros2 topic echo /face/joint_states --once
ros2 topic hz /face/joint_states        # should match your camera FPS
ros2 topic echo /face/expression
```

In RViz2 with the face URDF loaded, raise your eyebrows — the robot's brow links should rotate in sync.

---

## 5. Bonus — Vital Signs from the Same Face Pixels (rPPG) (5 min)

Here is something that surprises everyone: the same webcam frames we're using for expressions also contain your **heart rate**. No extra sensor.

### How it works — remote photoplethysmography (rPPG)

Every heartbeat pushes a pulse of blood through the capillaries in your facial skin. Hemoglobin absorbs light (green wavelengths most strongly), so with each beat the skin gets **imperceptibly darker, then lighter** — a change of well under 1% in pixel intensity. Your eye can't see it; a camera averaging thousands of pixels can. This is the same principle as the pulse oximeter clipped on your finger at the doctor's office (that one is *contact* PPG; ours is *remote*), and the same signal your smartwatch reads with its green LEDs.

The classic pipeline (Verkruysse et al., 2008):

```
┌─────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌──────────────┐   ┌─────────┐
│ Face video  │──▶│ Skin ROI         │──▶│ Spatial average│──▶│ Detrend +    │──▶│ FFT peak │──▶ BPM
│ (30 fps)    │   │ (forehead/cheeks)│   │ per frame → 1D │   │ bandpass     │   │ 0.7–4 Hz │
└─────────────┘   └──────────────────┘   │ green signal   │   │ (42–240 BPM) │   └─────────┘
                                         └────────────────┘   └──────────────┘
```

1. **ROI selection** — pick stable, well-perfused skin: the **forehead** (landmarks 19–24 give its lower edge) and **cheeks**. Avoid eyes, mouth, hair — they move with expressions and would corrupt the signal. Our dlib landmarks give us this ROI *for free*, tracked every frame.
2. **Spatial averaging** — mean green-channel value over the ROI, one number per frame. Averaging thousands of pixels cancels sensor noise and leaves the tiny pulse signal.
3. **Temporal filtering** — collect ~10 s (300 frames), remove the slow trend (lighting drift), band-pass to 0.7–4 Hz — the physiologically plausible heart-rate band.
4. **Frequency analysis** — FFT the filtered signal; the strongest peak *is* the pulse. Peak at 1.2 Hz → **72 BPM**.

Minimal working core (full version in Lab C4):

```python
import numpy as np

# forehead_means: list of ROI green-channel means, one per frame, ≥10 s long
sig = np.array(forehead_means)
sig = sig - np.convolve(sig, np.ones(31)/31, mode="same")   # detrend
freqs = np.fft.rfftfreq(len(sig), d=1/fps)
power = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))**2
band  = (freqs > 0.7) & (freqs < 4.0)                        # 42–240 BPM
bpm   = freqs[band][np.argmax(power[band])] * 60
print(f"Heart rate: {bpm:.0f} BPM")
```

### Beyond the green channel

- **CHROM / POS** — combine R, G, B channels into a projection that cancels lighting changes and motion; the standard classical methods (POS, Wang et al. 2017, is the usual baseline).
- **Deep rPPG** (PhysNet, PhysFormer) — CNNs/transformers trained end-to-end on video → pulse; better under motion, needs a GPU.
- **Respiration rate** falls out of the same signal (a slower 0.1–0.5 Hz modulation), and **Eulerian Video Magnification** (MIT, 2012) is the famous demo that amplifies these invisible color changes so you can *see* the pulse.
- Ready-made toolkits: **pyVHR** and **rPPG-Toolbox** implement all of the above.

### Why it's hard (and why it's perfect for sports)

Motion is the killer: an athlete mid-swing shifts the ROI and swamps the 1% pulse signal with 100% brightness changes — real systems gate measurements to still moments (our head-pose gate from Task 3 helps here). Lighting flicker (50/60 Hz mains), video compression, and varying skin tones (less contrast for the camera on darker skin — a documented fairness issue) all degrade accuracy. But for our coach, even a resting-between-sets heart rate is gold: **effort and recovery data with zero wearables** — published straight onto a `/face/heart_rate` topic (`std_msgs/Float32`) next to our expression topics.

---

## 6. Challenges — Why Faces Are Harder Than Elbows (5 min)

1. **Head pose** — dlib landmarks degrade beyond ~±30° yaw. A tennis player mid-serve is rarely facing the camera. Mitigation: gate on YOLO ear visibility (both ears low-confidence ⇒ profile view ⇒ hold last expression).
2. **Scale** — at full-court distance the face is ~40 px wide; landmark jitter swamps the signal. Expression work needs the athlete within a few meters, or a second zoomed camera.
3. **Flicker** — raw per-frame classification oscillates `neutral↔happy` at the decision boundary. Fixes: EMA on joint values (done above) + **hysteresis** on labels (only switch after N=5 consecutive frames agree).
4. **Individual variation** — resting faces differ enormously; without per-user baseline calibration your "frown" threshold will insult half your users.
5. **Latency budget** — the face pipeline shares the frame loop with body pose. dlib adds ~1 ms (fine); DeepFace adds ~200 ms (breaks real time — another reason for our library choice).
6. **The uncanny valley** — a robot face mirroring you at 30 Hz with zero lag feels creepy; slight smoothing and capped joint velocity actually *improves* perceived quality. Robotics is not only signal processing.
7. **Ethics** — emotion recognition is scientifically contested (expressions ≠ inner feelings) and regulated in some jurisdictions (e.g., the EU AI Act restricts emotion inference in workplaces/schools). We frame our system as **expression mirroring**, not emotion detection.

---

## 7. Lab Exercises (10 min)

### Lab C1 — 68 landmarks on your face (no ROS needed)

File: `labs/lab_c1_dlib_landmarks.py` — the YOLO-crop → dlib code from Section 2, plus landmark group coloring.

```bash
python labs/lab_c1_dlib_landmarks.py            # webcam
python labs/lab_c1_dlib_landmarks.py --video labs/videos/workout.mp4
```

Expected: green dots on eyes/brows, red on mouth, blue on jaw — tracking your face live. Check the console FPS: dlib should cost you < 2 ms per frame.

### Lab C2 — Feature dashboard & rule-based classifier

File: `labs/lab_c2_expression_features.py` — computes EAR, MAR, smile ratio, brow raise every frame; 30-frame neutral calibration at startup; prints the classified expression.

```bash
python labs/lab_c2_expression_features.py
```

```
Calibrating... hold a neutral face (30 frames)   ✓ baseline saved
EAR 0.28 | MAR 0.08 | smile 0.42 | brow 0.19  →  neutral
EAR 0.27 | MAR 0.63 | smile 0.44 | brow 0.24  →  surprise
```

Try: smile, jaw drop, blink, eyebrow raise. Which feature is noisiest?

### Lab C3 — Drive the robot face (Linux / ROS 2 required)

File: `labs/lab_c3_face_ros.py` — Lab C2 + the `FacePublisher` node from Section 4.

```bash
# Terminal 1
python labs/lab_c3_face_ros.py
# Terminal 2
ros2 topic hz /face/joint_states
ros2 topic echo /face/expression
```

### Lab C4 — Heart rate from your webcam (no ROS needed)

File: `labs/lab_c4_rppg_heartrate.py` — forehead ROI from dlib landmarks, green-channel signal, FFT peak → BPM (Section 5 pipeline). Sit **still** in steady light for the 15-second capture.

```bash
python labs/lab_c4_rppg_heartrate.py
```

```
Capturing 15 s... keep still, face the camera
Signal quality: OK (ROI stable, 448 frames @ 29.9 fps)
Heart rate: 68 BPM   (peak 1.13 Hz, SNR 4.2)
```

Verify against your smartwatch or a 15-second manual pulse count × 4.

---

## 8. Take-Home Tasks

1. **Threshold tuning**: Run Lab C2 and record your personal EAR when eyes are open vs closed, and MAR closed vs jaw-dropped. Are the defaults (0.12, 0.55) right for you? Adjust and report your values.

2. **Blink counter**: Extend Lab C2 to count blinks per minute (EAR below threshold for 2–5 consecutive frames = one blink). Athletes' blink rate drops under concentration — verify on the workout video or on yourself.

3. **Head-pose gate**: Using only YOLO's 5 face keypoints, estimate yaw (hint: compare nose-to-left-ear vs nose-to-right-ear horizontal distance). Suppress expression output when |yaw| looks larger than ~30°. How often does the gate trigger on the workout video?

4. **Heart-rate validation**: Run Lab C4 five times: (a) sitting still, (b) right after 20 jumping jacks, (c) in dim light, (d) while slowly nodding, (e) still again. Record the BPM and compare each against a smartwatch or manual pulse count. Which conditions break the measurement, and does that match the failure modes from Section 5? Write 5–10 sentences.

5. **Smoothing study**: In Lab C3, try `alpha = 1.0` (no smoothing), `0.3`, and `0.05`. Echo `/face/joint_states` and describe the trade-off: what does the jaw joint do in each case when you talk?

6. *(Stretch)* Add a `mirror` mode to the face URDF: extend `robot/humanoid.urdf` with the jaw + brow joints from Section 4, launch `robot_state_publisher` with it, and see your expressions on the RViz2 model in real time.

---

## Resources

| Resource | URL |
|----------|-----|
| dlib | http://dlib.net |
| Kazemi & Sullivan — 1 ms Face Alignment | https://ieeexplore.ieee.org/document/6909637 |
| 68-landmark annotation scheme (iBUG 300-W) | https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/ |
| EAR blink paper (Soukupová & Čech) | https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf |
| DeepFace (skipped — heavy TF dependency) | https://github.com/serengil/deepface |
| FER library | https://github.com/JustinShenk/fer |
| rPPG origin paper (Verkruysse 2008) | https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-26-21434 |
| POS algorithm (Wang 2017) | https://ieeexplore.ieee.org/document/7565547 |
| rPPG-Toolbox | https://github.com/ubicomplab/rPPG-Toolbox |
| pyVHR | https://github.com/phuselab/pyVHR |
| Eulerian Video Magnification (MIT) | https://people.csail.mit.edu/mrub/vidmag/ |
| OpenFace 2.0 (Action Units) | https://github.com/TadasBaltrusaitis/OpenFace |
| py-feat (Python AU toolkit) | https://py-feat.org |
| FACS overview | https://www.paulekman.com/facial-action-coding-system/ |
| InMoov head (open-source face robot) | https://inmoov.fr/head/ |
| ROS 2 JointState msg | https://docs.ros2.org/latest/api/sensor_msgs/msg/JointState.html |
| EU AI Act & emotion recognition | https://artificialintelligenceact.eu/article/5/ |

---

## Class Schedule Overview

| Class | Topic |
|-------|-------|
| 1 | System Architecture, Setup & Toolchains |
| 2 | Keypoint Tracking, YOLO & CNN Deep Dive |
| **3** | **Facial Expression Recognition & Robot Face Mapping** ← you are here |
| 4 | Kinematic Math — Angles from Keypoints |
| 5 | ROS 2 Publisher & Joint State Messages |
| 6 | RViz2 Visualization & URDF Tuning |
| 7 | Sports Analysis — Tennis Serve / Golf Swing Scoring |
