# Wall Intrusion Detection — Multi-Person Region LSTM (Every-5-Frames)

Self-contained, offline surveillance-AI system: detects **wall intrusion
activity in multi-person CCTV videos** using YOLO11 pose estimation,
wall-region geometry and a variable-length BiLSTM — including a
**live-CCTV mode** that raises an alarm the moment a person crosses the
monitored wall while the feed plays.

Everything needed to run the exact trained checkpoint
`multi_person_region_lstm_every_5_frames.pth` lives in this
repository — no internet access and no external services are required
at inference time (the YOLO weights are local).

## Contents

| File | Origin | Purpose |
|---|---|---|
| `models/multi_person_region_lstm_every_5_frames.pth` | copied from `artifacts/models_every_5_frames/` | Trained BiLSTM classifier (raw `state_dict`) |
| `yolo11n-pose.pt` | copied from project root | Local YOLO11 pose detector (never downloads) |
| `extract_every_5_frames.py` | copied unchanged | Person detection, box correction, 55-d normalized pose features, IoU track association, frame sampling every 5 frames |
| `extract_wall_region_features.py` | copied unchanged | 48-d wall-region features from annotated top/bottom boundaries |
| `train_every_5_frames.py` | copied unchanged | Source of truth for the `MultiPersonIntrusionLSTM_V2` architecture (imported by `predict.py` and the backend; its training code only runs if executed directly) |
| `predict.py` | **new** | End-to-end CLI inference: video + wall annotation → prediction |
| `backend/` | copied from `development/backend/`, adapted | FastAPI API serving this checkpoint (port 8010) |
| `frontend/` | copied from `development/frontend/`, adapted | React/Vite UI wired to the standalone backend (port 5174) |
| `requirements.txt` | new | Python dependencies for `predict.py` |

### Backend / frontend adaptations (copies only)

- `backend/app/services/ml_inference.py` — rebuilt for this checkpoint:
  variable-length every-5-frame sampling, 104-feature input, packed
  sequences with `lengths`, architecture imported from
  `train_every_5_frames.py`. Velocity gate exists but is **disabled by
  default** (`VELOCITY_GATE_ENABLED=false`) since it was tuned for the
  30-frame pipeline; this keeps the API numerically identical to the
  model's own evaluation script. Annotated previews returned to the
  client are capped evenly (`MAX_PREVIEW_FRAMES`; set to `0` to
  return every sampled frame) while the model still sees every
  sampled frame. Each prediction also returns `annotated_video`: an
  H.264 MP4 (as a base64 data URI) replaying the whole sampled
  timeline with the wall boundaries, person bounding boxes and pose
  skeletons drawn, playing at real speed (writer fps = video fps / 5).
- `backend/app/services/live_service.py` — **new**: live CCTV session
  service (see "Live CCTV monitoring" below).
- `backend/app/config.py` — checkpoint path, `feature_size=104`,
  `frame_stride=5`, CORS for port 5174.
- `frontend/vite.config.js` — dev server on **5174**, proxying `/api`
  to the standalone backend on **8010** (no clash with the original
  development stack on 5173/8000).
- `frontend/src/App.jsx` — subtitle notes the every-5-frames variant;
  adds the annotated-video replay beside the frame previews, a
  "Total Video Frames" metric, and a "Live CCTV Monitoring" card
  (streaming overlay, alarm banner, beep, alarm log). Everything else
  is UI-compatible with the original.

## Input signature

At training time the model consumes sequences of shape
`(frames, 5 people, 104 features)` where the 104 features are:

- `55` normalized pose features (from YOLO11n-pose)
- `48` wall-region features (distances / relative positions of body
  groups and the bounding box to the annotated wall boundaries)
- `1` wall-annotation-availability flag = 1.0

Sequences are **variable length**: one sampled frame every 5 video
frames. The BiLSTM is shared per person (packed sequences), then peak
temporal pooling + last-valid-frame pooling, masked max pooling across
people, and a small classifier head.

## Usage

### CLI (single file)

```bash
python predict.py \
    --video path/to/video.mp4 \
    --annotation path/to/annotation.json \
    --threshold 0.5 \
    --out result.json
```

### Full web app (backend + frontend)

The package also contains isolated copies of the FastAPI backend and
React frontend, adapted to serve this checkpoint. They use
**different ports** than the original development package so both can
run simultaneously:

```bash
# Terminal 1 — backend (port 8010)
cd backend
uvicorn app.main:app --port 8010

# Terminal 2 — frontend (port 5174, proxies /api to 8010)
cd frontend
npm install   # if node_modules is not present
npm run dev
```

Open http://localhost:5174 — upload a video, mark the wall top/bottom
boundaries on the captured frame, set the threshold, and click
Analyze. The API contract is identical to the original backend
(`POST /api/predict`, `GET /api/health`); the response additionally
reports `frame_stride`, `total_video_frames`, `video_duration_seconds`,
an `annotated_video` data URI, and `model_version = "every_5_frames"`.

### Batch analysis results

- **Frame previews** — every sampled frame (`MAX_PREVIEW_FRAMES=0`)
  with wall lines, person boxes, pose skeletons and keypoints drawn.
- **Annotated Detection Replay** — the same annotations rendered as a
  playable H.264 MP4 beside the frame grid (1 sample = 5 video frames,
  i.e. it plays at real speed).
- **Total Video Frames** — the model only sees every 5th frame, but
  the result reports the full count of the source video.

### Live CCTV monitoring

The "3. Live CCTV Monitoring" card turns the same video into a
simulated camera stream:

1. Click **Start Live Monitoring** — the video plays in real time
   (muted) like a CCTV feed.
2. The frontend captures the current frame ~3×/second and POSTs it to
   `/api/live/frame`.
3. Each frame runs YOLO detection + track association + wall-region
   features, and the BiLSTM is evaluated over a trailing window
   (last 60 sampled frames) on **every** ingest.
4. The moment the window probability crosses the threshold, an alarm
   fires: red banner over the feed, beep, red box colour, and an
   entry in the alarm log with the frame number + timestamp when the
   person actually crossed.
5. Alarm release uses hysteresis (`threshold - 0.10`) to avoid
   flapping; when the video ends the session stops automatically and
   a summary (frames streamed, alarm list, max probability) is shown.

Backend endpoints (all `POST`, multipart form):

| Endpoint | Fields | Purpose |
|---|---|---|
| `/api/live/start` | `boundaries` (same JSON as `/api/predict`), `threshold` | Opens a session, returns `session_id` |
| `/api/live/frame` | `session_id`, `frame` (JPEG), `frame_index`, `timestamp_seconds` | Ingest one stream frame; returns normalized detections, current `latest_probability`, `alarm_active`, `alarm_raised` and the alarm list |
| `/api/live/stop` | `session_id` | Closes the session, returns the full summary |

`alarm_raised` is `true` only on the single ingest where a new alarm
starts — the frontend triggers the beep/banner on it, while
`alarm_active` controls the red overlay while the hysteresis window is
still open.

### Annotation JSON schema

Identical to what the web frontend produces (normalized coordinates,
`[0, 1]`, relative to the video frame):

```json
{
    "frame_width": 320,
    "frame_height": 240,
    "wall_top_points":    [[0.80, 0.44], [0.38, 0.83]],
    "wall_bottom_points": [[0.79, 0.58], [0.69, 0.71]]
}
```

## Requirements

```
python -m pip install -r requirements.txt        # package root
python -m pip install -r backend/requirements.txt  # for the web API
```

torch, ultralytics (YOLO), opencv-python, numpy, plus FastAPI/pydantic
for the backend. No internet access is required at inference time;
the YOLO weights are local.

## Notes

- Untouched 30-frame model variants (`artifacts/models/multi_person*
  .pth`, backend `development/`) are **not** compatible with this
  checkpoint: this variant uses variable-length sequences and a
  104-d feature vector (103 + annotation flag), and its forward
  signature is `(x, track_ids, lengths)`.
- The checkpoint is a bare `state_dict`; `predict.py` owns the
  architecture definition, loaded verbatim from the training script.
