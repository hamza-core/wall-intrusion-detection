# Beginner's Guide — How This Model Is Trained

This guide explains, in simple English, how the wall-intrusion model was built
and how you can retrain it yourself. **No deep AI knowledge is needed to follow
along.**

---

## 1. The big picture

The system watches a video feed and answers one question:

> **"Is somebody climbing over the wall in this video?"**

It answers with two numbers for each person in the scene:

- **A probability** between `0.0` and `1.0` (0 = normal, 1 = crossing the wall)
- **A decision** when the probability passes a chosen **threshold** (for example 0.5)
  → `Normal` or `Wall Crossing`

It does **not** look at raw video pixels. Instead, it works in 3 stages:

```
 STAGE 1: EXTRACT FEATURES        STAGE 2: TRAIN           STAGE 3: USE
 (extract_every_5_frames.py)     (train_every_5_frames.py) (predict.py / web app)

 video frames ──► YOLOv11n-pose ──► 104 numbers     ──►  BiLSTM  ──► probability
 every 5th frame  (person skeletons) per person        (many training   0.98 → alarm!
                                                   ──►  examples)      0.02 → normal
```

Think of it like this:

- **Stage 1 (extraction)** turns each video into a neat table of numbers:
  for every person, for every sampled frame — where their body parts are
  and how far they are from the wall boundaries.
- **Stage 2 (training)** shows thousands of these tables to a neural network
  (a BiLSTM). It slowly learns the *pattern of numbers* that happens when
  someone climbs over a wall.
- **Stage 3 (prediction)** uses the trained model on new videos — either the
  little `predict.py` tool or the web dashboard.

---

## 2. What each script does

| Script | What it is | When you run it |
|---|---|---|
| `extract_every_5_frames.py` | Watches every video in `dataset/train`, samples **every 5th frame**, detects people with YOLO, and saves the numbers as files | **Step 1** — once, before training |
| `extract_wall_region_features.py` | A helper library that calculates the **48 wall-region features** (distances to the wall boundaries). It is imported by the script above | (Runs automatically — no need to call it) |
| `train_every_5_frames.py` | Reads the saved numbers and trains the BiLSTM model. Saves `models/multi_person_region_lstm_every_5_frames.pth` | **Step 2** — the main training run |
| `predict.py` | Tests the trained model on any single video from the command line | **Step 3** — quick testing |
| The web app (`backend/` + `frontend/`) | Full dashboard with live camera mode and file uploads | Everyday use |

---

## 3. Before you start — folder layout

Training expects these things on disk:

```
your-project/
├── yolo11n-pose.pt                      ← the pose detector (already included)
├── dataset/
│   └── train/
│       ├── Normal Class/   *.mp4        ← videos where nobody crosses
│       └── Wall crossing/  *.mp4        ← videos where someone crosses
├── artifacts/
│   ├── wall_region_annotations/train/   ← JSON files: the wall boundaries
│   │   └── <class>/<video-name>.json
│   ├── every_5_frames/                  ← (created by Step 1)
│   ├── models_every_5_frames/           ← (created by Step 2: the .pth file)
│   └── experiments_every_5_frames/      ← (created by Step 2: results JSONs)
├── extract_every_5_frames.py
└── train_every_5_frames.py
```

Two kinds of **labels** are used:

- **Folder name** = the ground-truth class of a video.
  `Normal Class` → label `0`, `Wall crossing` → label `1`.
- **Annotation JSON** = the drawn wall line (top and bottom boundary points).
  A video *with* an annotation also contributes wall-region features;
  a video *without* one still trains, just on pose features only.

---

## 4. The 104 numbers (features) — the heart of the system

For every person, in every sampled frame, the script builds exactly **104 numbers**:

| Position | Count | What they mean |
|---|---|---|
| `0–54` | **55** | **Pose features** — normalized positions of the 17 body keypoints as detected by YOLOv11n-pose (nose, shoulders, wrists, hips, knees…), plus per-person confidence. Normalized 0–1 relative to the image, so camera size doesn't matter. |
| `55–102` | **48** | **Wall-region features** — how close each body part (nose, wrists, shoulders, hips…) is to the wall's *top* and *bottom* boundary lines, inside vs outside the safe zone. These tell the model "the person is touching the wall" vs "standing far away". |
| `103` | **1** | **Annotation flag** — `1` means this video has a wall annotation (so the 48 features are real), `0` means pose-only. |

Other key shape facts:

- **5 person slots** — up to 5 people are tracked per frame
  (`MAX_PEOPLE = 5`). Empty slots are filled with zeros.
- **Every 5th frame** (`FRAME_STRIDE = 5`) — real videos are 30 fps;
  sampling 1 in 5 frames is 4× faster and still captures the motion.
- **Variable length** — a 10-second video gives ~60 sampled frames,
  a 2-minute one gives ~720. The trainer handles any length.

---

## 5. Step 1 — Extract the features

```bash
python extract_every_5_frames.py              # process all videos
python extract_every_5_frames.py --limit 5    # just 5 videos, to test fast
python extract_every_5_frames.py --video my_clip.mp4    # one specific video
```

What happens inside:

1. It looks at `dataset/train/<class>/*.mp4` (only videos that have an
   annotation JSON — that annotation is what defines the wall).
2. Every 5th frame is resized to 640 px and sent to **YOLOv11n-pose**
   (`yolo11n-pose.pt`, runs locally on your Mac GPU or CPU — no internet).
3. Detections are matched across frames into **track slots** (person A stays
   slot 0, person B stays slot 1…), so the model can follow a person over time.
4. It saves, for each video:
   - `pose_sequences/...npz` — skeleton features `(frames, 5, 55)`
   - `wall_region_features/...npz` — combined pose + wall features `(frames, 5, 103)`

**Wait, what's an `.npz` file?** A compressed NumPy archive — simply a file
with several arrays inside. You'll never open it by hand; the trainer reads it.

Output you'll see: a line per video like `✔ 1/240 dataset/.../1.mp4 (62 frames)`.
Approximate time: a few seconds per one-minute video on an Apple-silicon Mac.

---

## 6. Step 2 — Train the model

```bash
python train_every_5_frames.py
```

That's all. It does the whole run itself:

1. **Splits** videos into *training* and *validation* sets
   (the split is loaded from `artifacts/experiment_split_every_5_frames.json`,
   so retraining uses exactly the same split → results are comparable).
2. **Shows example batches** to the network, 4 videos at a time (`BATCH_SIZE = 4`).
3. After each pass over the data (**epoch**), it checks score on the
   validation set — videos the model has **never seen**.
4. It keeps the epoch with the best validation loss
   (**early stopping**: if validation stops improving for 12 epochs, it stops
   early — no wasted time).
5. Saves:
   - `artifacts/models_every_5_frames/multi_person_region_lstm_every_5_frames.pth`
     ← **the trained model** (this is the file the web app uses)
   - `..._history.json` — loss per epoch (you can plot this)
   - `..._results.json` — the final scores (see next section)

### How the model thinks (short version)

- Each person's frame-by-frame numbers are fed through a **BiLSTM**
  (a network that reads a sequence forwards *and* backwards → it understands
  motion in both directions: "where did the person come from, where are they
  going").
- From each person's whole timeline it boils the motion down to
  **two snapshots**: the *peak* moment (most suspicious instant) and the
  *last valid frame* (how the motion ended).
- Of all tracked people, it takes the **most suspicious one** and makes the
  final call. So one climber among many bystanders is still detected.

### Numbers you can safely play with

Found at the top of `train_every_5_frames.py`:

| Setting | Default | Meaning if you change it |
|---|---|---|
| `MAX_EPOCHS = 60` | raise → allow longer learning (early stopping still protects you) |
| `PATIENCE = 12` | lower → stops more aggressively |
| `BATCH_SIZE = 4` | raise → fewer update steps per epoch, more stable (needs more memory) |
| `LEARNING_RATE = 0.001` | 0.0005 → gentler, more careful learning |

### Numbers you must NOT change

- `FEATURE_SIZE = 104`, `POSE_FEATURE_SIZE = 55`, `REGION_FEATURE_SIZE = 48`
- `MAX_PEOPLE = 5`, `FRAME_STRIDE = 5` (changing it requires re-extraction
  and breaks compatibility with the model shipped in `models/`)

---

## 7. Step 3 — Test on a single video

```bash
python predict.py \
    --video dataset/test/Normal\ class/1.mp4 \
    --annotation artifacts/wall_region_annotations/test/Normal\ class/1.json
```

Optional:

```bash
--threshold 0.2   # stricter: alarm even at 20% probability
--out result.json # also save the result as JSON
```

Typical output:

```
Video:          1.mp4
Probability:    0.009757
Verdict:        NORMAL  (threshold 0.50)
```

Try the same on a *Wall crossing* test video — the probability should be high
(usually > 0.9).

---

## 8. Reading the results (metric glossary)

`..._results.json` contains these — what they mean:

| Metric | Plain meaning | Good value for this task |
|---|---|---|
| **accuracy** | % of videos judged correctly | ~1.00 |
| **precision** | Of all "INTRUSION!" alarms, how many were real | as high as possible |
| **recall** | Of all real intrusions, how many were caught | as high as possible |
| **F1** | Balance between precision and recall | ~1.00 |
| **ROC-AUC** | How well the model ranks real vs normal (1.0 = perfect) | ~1.00 |
| **confusion matrix** | 4 numbers: correct normals, false alarms, missed intrusions, caught intrusions | aim for a perfect diagonal |

The trainer also reports results at **two thresholds**:

- `0.50` — the general-purpose default
- *optimal threshold* — picked automatically (0.20–0.80) to maximize F1 on
  the validation set. This matches the threshold you can set in the web UI.

---

## 9. Common errors, explained

| Error | Why it happens | Fix |
|---|---|---|
| `Split file not found` | Step 1 never ran (or ran in the wrong folder) | Run `extract_every_5_frames.py` from the project root first |
| `Missing pose file` | A video in the split has no extracted `.npz` | Re-run Step 1 (maybe with more videos, without `--limit`) |
| `Pose/region track IDs differ` | The pose and wall files disagree — usually from older artifacts | Delete `artifacts/every_5_frames/` and re-run Step 1 |
| `Pose label mismatch` | A `dataset` folder name and a saved label disagree | Check the video is in the right class folder and re-extract it |
| Very slow / "mps" warnings | No Apple GPU acceleration available → falls back to CPU | That's fine, just slower. Errors are not expected |
| Results look worse than expected | You changed a "do not change" number, or a dataset/annotation was edited | Restore defaults, then re-run Step 1 and Step 2 |

Golden rule when something breaks: **delete the folder
`artifacts/every_5_frames/`, redo Step 1, then redo Step 2.**

---

## 10. Mini glossary of AI terms used here

| Term | In plain English |
|---|---|
| **feature** | One measured number about the scene (e.g. "left wrist is 0.12 m from the wall line") |
| **epoch** | One full pass of the model over all training videos |
| **validation set** | Videos kept aside — the model never trains on them; they check if it *really learned* |
| **loss** | "How wrong" the model currently is — during training it should go down |
| **BiLSTM** | A network that reads a sequence of frames forwards *and* backwards — good at spotting motion patterns |
| **pooling** | Boiling a long sequence of values down to one summary (here: the *peak* moment + the *last* frame) |
| **threshold** | The cut-off probability for shouting "intrusion!" — lower = more sensitive (more false alarms) |
| **early stopping** | Stop when validation stops improving — avoids over-tuning memorization |
| **checkpoint (`.pth`)** | The saved trained model — a file with all learned weights |
| **batch** | A small group of videos processed together (here: 4) |
| **track ID** | The slot number of one person across frames (person 0, person 1, …) |
| **seed = 42** | a fixed starting point for randomness → re-runs give the same result |

---

## 11. Quick command cheat-sheet

```bash
# 0. Check GPU (optional)
python -c "import torch; print(torch.device('cuda' if torch.backends.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')))"

# 1. Extract features (test on 3 videos first, then all)
python extract_every_5_frames.py --limit 3
python extract_every_5_frames.py

# 2. Train
python train_every_5_frames.py

# 3. Test one video
python predict.py --video <video.mp4> --annotation <annotation.json>

# 4. Use the web app (preferred for everyday use)
cd backend && uvicorn app.main:app --reload
cd frontend && npm run dev
```
