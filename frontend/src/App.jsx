import React, { useState, useEffect, useRef } from 'react';

export default function App() {
  // API Health status
  const [apiStatus, setApiStatus] = useState({ online: false, device: '', checked: false });

  // Video State
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [videoDims, setVideoDims] = useState({ width: 0, height: 0 });
  const [videoFps, setVideoFps] = useState(30);
  const videoRef = useRef(null);

  // Annotation State
  // Points are stored in NORMALIZED coordinates [0, 1] relative to video frame
  const [topPoints, setTopPoints] = useState([]);
  const [bottomPoints, setBottomPoints] = useState([]);
  const [activeBoundary, setActiveBoundary] = useState('top'); // 'top' or 'bottom'
  const [capturedFrameImg, setCapturedFrameImg] = useState(null);
  const canvasRef = useRef(null);

  // Analysis / Prediction State
  const [threshold, setThreshold] = useState(0.50);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisTimer, setAnalysisTimer] = useState(0);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // ---------------------------------------------------------------------------
  // Live CCTV state
  // ---------------------------------------------------------------------------
  const [liveState, setLiveState] = useState({
    active: false,
    starting: false,
    sessionId: null,
    latestProbability: null,
    alarmActive: false,
    alarms: [],        // {frame_number, timestamp_seconds, probability}
    framesStreamed: 0,
    dropped: 0,        // frames skipped while a request was in flight
    error: null,
    summary: null,     // set on stop
  });
  const liveOverlayRef = useRef({ people: [] }); // latest normalized detections
  const liveRequestInFlight = useRef(false);
  const liveSessionRef = useRef(null);
  const liveCanvasRef = useRef(null);
  const drawLoopRef = useRef(null);
  const captureTimerRef = useRef(null);
  const audioCtxRef = useRef(null);

  // ---------------------------------------------------------------------------
  // Check API Health on load
  // ---------------------------------------------------------------------------
  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setApiStatus({
          online: data.status === 'healthy' && data.yolo_loaded && data.lstm_loaded,
          device: data.device || 'cpu',
          checked: true,
        });
      } else {
        setApiStatus({ online: false, device: '', checked: true });
      }
    } catch {
      setApiStatus({ online: false, device: '', checked: true });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Timer during analysis
  useEffect(() => {
    let timerId;
    if (isAnalyzing) {
      setAnalysisTimer(0);
      timerId = setInterval(() => {
        setAnalysisTimer((t) => t + 1);
      }, 1000);
    }
    return () => clearInterval(timerId);
  }, [isAnalyzing]);

  // ---------------------------------------------------------------------------
  // Video Handling
  // ---------------------------------------------------------------------------
  const handleVideoSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }

    const url = URL.createObjectURL(file);
    setVideoFile(file);
    setVideoUrl(url);
    setTopPoints([]);
    setBottomPoints([]);
    setCapturedFrameImg(null);
    setPredictionResult(null);
    setErrorMessage(null);
  };

  const handleClearVideo = () => {
    if (liveState.active || liveSessionRef.current) handleStopLive(true);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoFile(null);
    setVideoUrl(null);
    setVideoDims({ width: 0, height: 0 });
    setTopPoints([]);
    setBottomPoints([]);
    setCapturedFrameImg(null);
    setPredictionResult(null);
    setErrorMessage(null);
    setLiveState({
      active: false, starting: false, sessionId: null, latestProbability: null,
      alarmActive: false, alarms: [], framesStreamed: 0, dropped: 0, error: null, summary: null,
    });
  };

  const handleVideoMetadataLoaded = () => {
    if (videoRef.current) {
      const w = videoRef.current.videoWidth || 640;
      const h = videoRef.current.videoHeight || 480;
      setVideoDims({ width: w, height: h });
      // Capture the initial frame automatically
      setTimeout(captureCurrentFrame, 200);
    }
  };

  // ---------------------------------------------------------------------------
  // Frame Capture for Annotation Canvas
  // ---------------------------------------------------------------------------
  const captureCurrentFrame = () => {
    if (!videoRef.current) return;
    const v = videoRef.current;
    if (v.videoWidth === 0 || v.videoHeight === 0) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = v.videoWidth;
    tempCanvas.height = v.videoHeight;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight);

    const img = new Image();
    img.src = tempCanvas.toDataURL('image/jpeg');
    img.onload = () => {
      setCapturedFrameImg(img);
    };
  };

  // ---------------------------------------------------------------------------
  // Canvas Redraw
  // ---------------------------------------------------------------------------
  const redrawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;

    ctx.clearRect(0, 0, width, height);

    // 1. Draw captured video frame background
    if (capturedFrameImg) {
      ctx.drawImage(capturedFrameImg, 0, 0, width, height);
    } else {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#64748b';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Select video & click "Capture Frame" to annotate', width / 2, height / 2);
    }

    // 2. Draw wall region shading if both boundaries have >= 2 points
    if (topPoints.length >= 2 && bottomPoints.length >= 2) {
      ctx.beginPath();
      // Forward along top points
      ctx.moveTo(topPoints[0].x * width, topPoints[0].y * height);
      for (let i = 1; i < topPoints.length; i++) {
        ctx.lineTo(topPoints[i].x * width, topPoints[i].y * height);
      }
      // Backward along bottom points to form a polygon
      for (let i = bottomPoints.length - 1; i >= 0; i--) {
        ctx.lineTo(bottomPoints[i].x * width, bottomPoints[i].y * height);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
      ctx.fill();
    }

    // 3. Helper to draw boundary points and lines
    const drawBoundary = (points, color, label) => {
      if (points.length === 0) return;

      // Connecting lines
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.moveTo(points[0].x * width, points[0].y * height);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x * width, points[i].y * height);
      }
      ctx.stroke();

      // Points and labels
      points.forEach((p, idx) => {
        const px = p.x * width;
        const py = p.y * height;

        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${label}${idx + 1}`, px, py - 10);
      });
    };

    drawBoundary(topPoints, '#38bdf8', 'T');
    drawBoundary(bottomPoints, '#f97316', 'B');
  };

  useEffect(() => {
    redrawCanvas();
  }, [capturedFrameImg, topPoints, bottomPoints, videoDims]);

  // ---------------------------------------------------------------------------
  // Canvas Click: Add Point
  // ---------------------------------------------------------------------------
  const handleCanvasClick = (e) => {
    if (!capturedFrameImg || liveState.active || isAnalyzing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Convert to normalized coordinates [0, 1]
    const normX = Math.min(Math.max(clickX / rect.width, 0), 1);
    const normY = Math.min(Math.max(clickY / rect.height, 0), 1);

    const newPoint = {
      x: Number(normX.toFixed(6)),
      y: Number(normY.toFixed(6)),
    };

    if (activeBoundary === 'top') {
      setTopPoints((prev) => [...prev, newPoint]);
    } else {
      setBottomPoints((prev) => [...prev, newPoint]);
    }
  };

  const handleUndoPoint = () => {
    if (activeBoundary === 'top') {
      setTopPoints((prev) => prev.slice(0, -1));
    } else {
      setBottomPoints((prev) => prev.slice(0, -1));
    }
  };

  const handleClearCurrentBoundary = () => {
    if (activeBoundary === 'top') {
      setTopPoints([]);
    } else {
      setBottomPoints([]);
    }
  };

  const handleClearAllBoundaries = () => {
    setTopPoints([]);
    setBottomPoints([]);
  };

  // ---------------------------------------------------------------------------
  // Submit Video and Annotation for Prediction
  // ---------------------------------------------------------------------------
  const handleAnalyze = async () => {
    if (!videoFile) {
      setErrorMessage('Please select a video file.');
      return;
    }
    if (topPoints.length < 2) {
      setErrorMessage('Wall Top boundary requires at least 2 points.');
      return;
    }
    if (bottomPoints.length < 2) {
      setErrorMessage('Wall Bottom boundary requires at least 2 points.');
      return;
    }

    setErrorMessage(null);
    setPredictionResult(null);
    setIsAnalyzing(true);

    try {
      const boundaryPayload = {
        wall_top_points: topPoints,
        wall_bottom_points: bottomPoints,
        frame_width: videoDims.width || 640,
        frame_height: videoDims.height || 480,
      };

      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('boundaries', JSON.stringify(boundaryPayload));
      formData.append('threshold', threshold.toString());

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.error || 'Prediction request failed.');
      }

      setPredictionResult(data);
    } catch (err) {
      setErrorMessage(err.message || 'An unexpected error occurred during prediction.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const isReadyToAnalyze =
    videoFile &&
    topPoints.length >= 2 &&
    bottomPoints.length >= 2 &&
    !isAnalyzing &&
    !liveState.active;

  // ---------------------------------------------------------------------------
  // Live CCTV Mode
  // The video plays in real time like a camera feed; frames are captured
  // ~3x/second, POSTed to /api/live/frame, and YOLO detections returned
  // are drawn as an overlay. Alarm fires when the streaming BiLSTM window
  // probability crosses the threshold (person actually crossing).
  // ---------------------------------------------------------------------------
  const LIVE_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8],
    [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16],
  ];

  const playAlarmBeep = () => {
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;
      const now = ctx.currentTime;
      [0, 0.35, 0.7].forEach((offset) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'square';
        osc.frequency.value = 880;
        gain.gain.setValueAtTime(0.0001, now + offset);
        gain.gain.exponentialRampToValueAtTime(0.25, now + offset + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.25);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now + offset);
        osc.stop(now + offset + 0.3);
      });
    } catch {
      /* audio may be blocked until a user gesture — banner still shows */
    }
  };

  const drawLiveFrame = () => {
    const canvas = liveCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.videoWidth === 0) return;

    if (canvas.width !== videoDims.width || canvas.height !== videoDims.height) {
      canvas.width = videoDims.width || 640;
      canvas.height = videoDims.height || 480;
    }

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Wall boundaries
    if (liveState.alarmActive) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.18)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    const drawLine = (pts, color, label) => {
      if (pts.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.moveTo(pts[0].x * canvas.width, pts[0].y * canvas.height);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x * canvas.width, pts[i].y * canvas.height);
      }
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = 'bold 20px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(label, pts[0].x * canvas.width + 8, Math.max(24, pts[0].y * canvas.height - 10));
    };
    drawLine(topPoints, '#f9a825', 'WALL TOP');
    drawLine(bottomPoints, '#fb8c00', 'WALL BOTTOM');

    // Detections from the live backend
    const people = liveOverlayRef.current.people || [];
    for (let i = 0; i < people.length; i++) {
      const person = people[i];
      const color = liveState.alarmActive ? '#ef4444' : '#22c55e';

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(
        person.box[0] * canvas.width,
        person.box[1] * canvas.height,
        (person.box[2] - person.box[0]) * canvas.width,
        (person.box[3] - person.box[1]) * canvas.height,
      );

      ctx.fillStyle = color;
      ctx.font = 'bold 18px sans-serif';
      ctx.fillText(
        `P${person.track_id} ${person.confidence.toFixed(2)}`,
        person.box[0] * canvas.width + 4,
        Math.max(20, person.box[1] * canvas.height - 6),
      );

      // Skeleton
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2;
      LIVE_SKELETON.forEach(([a, b]) => {
        const ka = person.keypoints[a];
        const kb = person.keypoints[b];
        if (ka[2] >= 0.25 && kb[2] >= 0.25) {
          ctx.beginPath();
          ctx.moveTo(ka[0] * canvas.width, ka[1] * canvas.height);
          ctx.lineTo(kb[0] * canvas.width, kb[1] * canvas.height);
          ctx.stroke();
        }
      });

      person.keypoints.forEach(([x, y, c]) => {
        if (c >= 0.25) {
          ctx.beginPath();
          ctx.fillStyle = '#fb923c';
          ctx.arc(x * canvas.width, y * canvas.height, 5, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    }

    // Probability bar
    if (liveState.latestProbability != null) {
      const barW = canvas.width * 0.28;
      const barH = 26;
      const barX = 20, barY = canvas.height - barH - 20;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
      ctx.fillRect(barX, barY, barW, barH);
      ctx.fillStyle = liveState.alarmActive ? '#ef4444' : '#38bdf8';
      ctx.fillRect(barX, barY, barW * Math.min(1, liveState.latestProbability), barH);
      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(
        `P(cross) ${(liveState.latestProbability * 100).toFixed(1)}%  T=${liveState.alarms.length}`,
        barX + 6, barY + 18,
      );
    }

    drawLoopRef.current = requestAnimationFrame(drawLiveFrame);
  };

  const feedLiveFrame = async () => {
    const video = videoRef.current;
    if (!video || video.paused || video.ended) return;
    if (liveRequestInFlight.current) {
      setLiveState((s) => ({ ...s, dropped: s.dropped + 1 }));
      return;
    }

    liveRequestInFlight.current = true;
    try {
      // Capture the CURRENT frame at (near) native resolution.
      const maxW = Math.min(video.videoWidth, 1280);
      const scale = maxW / video.videoWidth;
      const c = document.createElement('canvas');
      c.width = Math.round(video.videoWidth * scale);
      c.height = Math.round(video.videoHeight * scale);
      c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);

      const blob = await new Promise((resolve) =>
        c.toBlob((b) => resolve(b), 'image/jpeg', 0.8),
      );
      if (!blob || !liveSessionRef.current) return;

      const fd = new FormData();
      fd.append('session_id', liveSessionRef.current);
      fd.append('frame', blob, 'frame.jpg');
      fd.append('frame_index', String(Math.round(video.currentTime * videoFps)));
      fd.append('timestamp_seconds', String(video.currentTime));

      const res = await fetch('/api/live/frame', { method: 'POST', body: fd });
      if (!res.ok) return;
      const data = await res.json();

      liveOverlayRef.current = { people: data.people || [] };
      setLiveState((s) => ({
        ...s,
        latestProbability: data.latest_probability,
        alarmActive: data.alarm_active,
        alarms: data.alarms || s.alarms,
        framesStreamed: data.frames_streamed,
      }));

      if (data.alarm_raised) {
        playAlarmBeep();
      }
    } catch {
      /* transient network/parse errors — next tick retries */
    } finally {
      liveRequestInFlight.current = false;
    }
  };

  const handleStartLive = async () => {
    if (!videoFile || topPoints.length < 2 || bottomPoints.length < 2 || liveState.starting) return;

    setLiveState((s) => ({ ...s, starting: true, error: null, summary: null, alarms: [], latestProbability: null, alarmActive: false, framesStreamed: 0, dropped: 0 }));
    try {
      const fd = new FormData();
      fd.append('boundaries', JSON.stringify({
        wall_top_points: topPoints,
        wall_bottom_points: bottomPoints,
        frame_width: videoDims.width || 640,
        frame_height: videoDims.height || 480,
      }));
      fd.append('threshold', threshold.toString());

      const res = await fetch('/api/live/start', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Could not start live session.');
      liveSessionRef.current = data.session_id;
      setLiveState((s) => ({ ...s, starting: false, active: true, sessionId: data.session_id }));

      // Play from the start like a camera feed.
      const v = videoRef.current;
      if (v) {
        v.muted = true;
        v.currentTime = 0;
        await v.play();
      }

      drawLoopRef.current = requestAnimationFrame(drawLiveFrame);
      captureTimerRef.current = setInterval(feedLiveFrame, 300);
    } catch (err) {
      setLiveState((s) => ({ ...s, starting: false, active: false, error: err.message }));
    }
  };

  const handleStopLive = async (reset = false) => {
    clearInterval(captureTimerRef.current);
    if (drawLoopRef.current) cancelAnimationFrame(drawLoopRef.current);
    drawLoopRef.current = null;
    captureTimerRef.current = null;
    liveRequestInFlight.current = false;
    liveOverlayRef.current = { people: [] };

    let summary = null;
    if (!reset && liveSessionRef.current) {
      try {
        const fd = new FormData();
        fd.append('session_id', liveSessionRef.current);
        const res = await fetch('/api/live/stop', { method: 'POST', body: fd });
        if (res.ok) {
          const data = await res.json();
          summary = {
            frames_streamed: data.frames_streamed,
            alarm_count: data.alarm_count,
            alarms: data.alarms || [],
            max_probability: data.max_probability,
            last_probability: data.last_probability,
            max_people_in_frame: data.max_people_in_frame,
          };
        }
      } catch { /* best effort */ }
    }
    liveSessionRef.current = null;

    const v = videoRef.current;
    if (v && !reset) v.pause();

    setLiveState((s) => ({
      ...s, active: false, starting: false, alarmActive: false, sessionId: null, summary,
    }));
  };

  useEffect(() => {
    return () => {
      clearInterval(captureTimerRef.current);
      if (drawLoopRef.current) cancelAnimationFrame(drawLoopRef.current);
      liveSessionRef.current = null;
    };
  }, []);

  return (
    <div className="container">
      {/* Header */}
      <header className="app-header">
        <div className="app-title-group">
          <h1>
            <span>🛡️</span> Wall Intrusion Detection System
          </h1>
          <p>
            Offline spatial-temporal vision pipeline with YOLO pose tracking &amp; multi-person region LSTM (every-5-frames variant)
          </p>
        </div>

        <div>
          {apiStatus.online ? (
            <span className="badge badge-online">
              <span className="pulse-dot"></span>
              API Online ({apiStatus.device})
            </span>
          ) : (
            <span className="badge badge-offline">
              <span className="pulse-dot"></span>
              {apiStatus.checked ? 'API Offline' : 'Connecting...'}
            </span>
          )}
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left Column: Video & Annotation */}
        <div className="workspace-column">
    
{/* 1. Camera Feed & Live Monitoring */}
<div className="card">
  <div className="card-title">
    <span>1. Camera Feed &amp; Live Monitoring</span>

    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      {liveState.active && (
        <span className="badge badge-online">
          <span className="pulse-dot"></span> LIVE — streaming
        </span>
      )}
      {videoFile && (
        <button
          type="button"
          className="btn btn-danger-outline"
          style={{ padding: '4px 10px', fontSize: '0.8rem' }}
          onClick={handleClearVideo}
          disabled={isAnalyzing}
        >
          Remove Video
        </button>
      )}
    </div>
  </div>

  {!videoFile ? (
    <label
      htmlFor="video-input"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        minHeight: '340px',
        boxSizing: 'border-box',
        padding: '30px 20px',
        border: '2px dashed #475569',
        borderRadius: '12px',
        background: 'rgba(15, 23, 42, 0.35)',
        cursor: 'pointer',
        textAlign: 'center',
        gap: '10px',
        transition: 'all 0.2s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#38bdf8';
        e.currentTarget.style.background = 'rgba(56, 189, 248, 0.06)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#475569';
        e.currentTarget.style.background = 'rgba(15, 23, 42, 0.35)';
      }}
    >
      <input
        id="video-input"
        type="file"
        accept="video/mp4,video/avi,video/quicktime,video/x-matroska,.mp4,.avi,.mov,.mkv"
        style={{ display: 'none' }}
        onChange={handleVideoSelect}
        disabled={isAnalyzing}
      />

      {/* Upload Icon */}
      <div
        style={{
          width: '58px',
          height: '58px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '14px',
          background: 'rgba(56, 189, 248, 0.12)',
          fontSize: '30px',
        }}
      >
        📹
      </div>

      {/* Upload Title */}
      <div
        style={{
          fontSize: '1rem',
          fontWeight: 650,
          color: '#f8fafc',
        }}
      >
        Click to select surveillance video
      </div>

      {/* Supported Formats */}
      <div
        style={{
          fontSize: '0.85rem',
          color: '#94a3b8',
        }}
      >
        Supports MP4, AVI, MOV, MKV
      </div>

      <div
        style={{
          fontSize: '0.75rem',
          color: '#64748b',
        }}
      >
        Select a video from your device to begin analysis
      </div>
    </label>
  ) : (
    <div>
      {/* CCTV-style stage: natural aspect ratio (no distortion), large view */}
      <div className="stage-wrap">
        <div
          className="monitor-stage"
          style={{
            aspectRatio: videoDims.width
              ? `${videoDims.width} / ${videoDims.height}`
              : '16 / 9',
            maxWidth: videoDims.width
              ? `calc(76vh * ${(videoDims.width / Math.max(1, videoDims.height)).toFixed(4)})`
              : '100%',
          }}
        >
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            onLoadedMetadata={handleVideoMetadataLoaded}
            onEnded={() => { if (liveState.active) handleStopLive(); }}
          />

          {liveState.active && (
            <div className="stage-live-chip">
              <span className="pulse-dot" style={{ color: '#fff' }}></span> LIVE
            </div>
          )}

          {/* Live overlay canvas — only mounted while monitoring */}
          {liveState.active && (
            <>
              <canvas
                ref={liveCanvasRef}
                className="stage-overlay"
              />
              {liveState.alarmActive && (
                <div className="stage-alarm-banner">
                  ⚠️ ALERT: PERSON CROSSING WALL
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Video Information */}
      <div className="file-info-bar">
        <div>
          <strong style={{ color: '#f8fafc' }}>
            {videoFile.name}
          </strong>

          <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Native Resolution: {videoDims.width} × {videoDims.height} px
          </div>
        </div>

        <button
          type="button"
          className="btn btn-secondary"
          onClick={captureCurrentFrame}
          disabled={isAnalyzing || liveState.active}
          title="Capture the current video frame for wall annotation"
        >
          📸 Grab Frame
        </button>
      </div>

      {/* Live CCTV controls — directly under the feed (no scrolling needed) */}
      <div className="live-toolbar">
        <button
          type="button"
          className="btn btn-analyze"
          style={{ minWidth: 190 }}
          onClick={handleStartLive}
          disabled={
            !videoFile ||
            topPoints.length < 2 ||
            bottomPoints.length < 2 ||
            liveState.active ||
            liveState.starting
          }
        >
          {liveState.starting ? 'Starting…' : liveState.active ? '● Streaming…' : '▶ Start Live Monitoring'}
        </button>
        <button
          type="button"
          className="btn btn-danger-outline"
          onClick={() => handleStopLive()}
          disabled={!liveState.active}
        >
          ■ Stop Monitoring
        </button>

        {/* P(crossing) meter with threshold marker */}
        <div className="live-meter">
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8' }}>
            <span>P(crossing) — threshold {threshold.toFixed(2)}</span>
            <strong style={{ color: liveState.alarmActive ? '#ef4444' : '#38bdf8' }}>
              {liveState.latestProbability != null
                ? `${(liveState.latestProbability * 100).toFixed(1)}%`
                : '—'}
            </strong>
          </div>
          <div className="live-meter-bar">
            <div
              className={`live-meter-fill${liveState.alarmActive ? ' alarm' : ''}`}
              style={{ width: `${Math.min(100, (liveState.latestProbability ?? 0) * 100)}%` }}
            />
            <div className="live-meter-threshold" style={{ left: `${threshold * 100}%` }} />
          </div>
        </div>
      </div>

      {liveState.error && (
        <div className="alert alert-danger" style={{ marginTop: 10 }}>
          <span>⚠️</span>
          <div>{liveState.error}</div>
        </div>
      )}

      {/* Live stats — always visible in this card */}
      <div className="metric-row">
        <div className="metric-box">
          <div className="metric-label">Frames Streamed</div>
          <div className="metric-value">{liveState.active ? liveState.framesStreamed : '—'}</div>
          <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>
            {liveState.active ? `dropped ${liveState.dropped}` : 'start monitoring to stream'}
          </div>
        </div>
        <div className="metric-box">
          <div className="metric-label">P(Crossing)</div>
          <div className="metric-value" style={{ color: liveState.alarmActive ? '#ef4444' : undefined }}>
            {liveState.latestProbability != null
              ? `${(liveState.latestProbability * 100).toFixed(1)}%`
              : '—'}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>window probability</div>
        </div>
        <div className="metric-box">
          <div className="metric-label">Alarms</div>
          <div className="metric-value" style={{ color: liveState.alarms.length > 0 ? '#ef4444' : undefined }}>
            {liveState.alarms.length}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>during stream</div>
        </div>
      </div>

      {/* Alarm log during stream / summary after stop */}
      {liveState.alarms.length > 0 && (
        <div className="alarm-log">
          {liveState.alarms.map((a, i) => (
            <div key={i} className="alarm-log-item">
              <span>🚨 Alarm #{i + 1} at <strong>{Number(a.timestamp_seconds).toFixed(2)}s</strong></span>
              <span>P = <strong>{(a.probability * 100).toFixed(1)}%</strong></span>
            </div>
          ))}
          {liveState.summary && (
            <div
              className="alarm-log-item"
              style={{ borderTop: '1px solid rgba(239, 68, 68, 0.3)', paddingTop: 6 }}
            >
              <span>
                Summary: {liveState.summary.frames_streamed} frames · peak{' '}
                {liveState.summary.max_people_in_frame} people
              </span>
              <span>
                max P ={' '}
                <strong>
                  {liveState.summary.max_probability != null
                    ? `${(liveState.summary.max_probability * 100).toFixed(1)}%`
                    : '—'}
                </strong>
              </span>
            </div>
          )}
        </div>
      )}

      {videoFile && (topPoints.length < 2 || bottomPoints.length < 2) && (
        <p style={{ fontSize: '0.78rem', color: '#fb923c', margin: '10px 0 0' }}>
          ⚠️ Mark both wall boundaries (Step 2 below) to enable live monitoring and analysis.
        </p>
      )}
    </div>
  )}
</div>


          {/* 2. Wall Boundary Annotation Card */}
          <div className="card">
            <div className="card-title">
              <span>2. Wall Boundary Marking</span>
              <span style={{ fontSize: '0.85rem', color: '#94a3b8', fontWeight: 'normal' }}>
                Top: {topPoints.length} pts | Bottom: {bottomPoints.length} pts
              </span>
            </div>

            <div className="annotation-controls">
              <div className="boundary-toggle-group">
                <button
                  type="button"
                  className={`boundary-tab tab-top ${activeBoundary === 'top' ? 'active' : ''}`}
                  onClick={() => setActiveBoundary('top')}
                >
                  🔵 Wall Top Boundary ({topPoints.length})
                </button>

                <button
                  type="button"
                  className={`boundary-tab tab-bottom ${activeBoundary === 'bottom' ? 'active' : ''}`}
                  onClick={() => setActiveBoundary('bottom')}
                >
                  🔴 Wall Bottom Boundary ({bottomPoints.length})
                </button>
              </div>

              <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: '5px 10px', fontSize: '0.8rem' }}
                  onClick={handleUndoPoint}
                  disabled={
                    (activeBoundary === 'top' ? topPoints.length : bottomPoints.length) === 0 ||
                    isAnalyzing
                  }
                >
                  ↩ Undo
                </button>

                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: '5px 10px', fontSize: '0.8rem' }}
                  onClick={handleClearCurrentBoundary}
                  disabled={
                    (activeBoundary === 'top' ? topPoints.length : bottomPoints.length) === 0 ||
                    isAnalyzing
                  }
                >
                  Clear Active
                </button>

                <button
                  type="button"
                  className="btn btn-danger-outline"
                  style={{ padding: '5px 10px', fontSize: '0.8rem' }}
                  onClick={handleClearAllBoundaries}
                  disabled={topPoints.length === 0 && bottomPoints.length === 0 || isAnalyzing}
                >
                  Clear All
                </button>
              </div>
            </div>

            {/* Canvas Area */}
            <div className="canvas-wrapper">
              <canvas
                ref={canvasRef}
                className="annotation-canvas"
                width={videoDims.width || 640}
                height={videoDims.height || 480}
                onClick={handleCanvasClick}
              />
            </div>

            <div className="canvas-hint">
              <span>
                💡 Click on the frame to place points along the wall boundary line.
              </span>
              <span>
                Minimum 2 points required per boundary.
              </span>
            </div>
          </div>

        </div>

        {/* Right Column: Actions & Results */}
        <div className="sidebar-column">
          {/* Action Card */}
          <div className="card">
            <div className="card-title">
              <span>3. Model Inference</span>
            </div>

            {/* Threshold / Sensitivity Slider */}
            <div style={{ marginBottom: 18, background: 'rgba(15, 23, 42, 0.4)', padding: '12px 14px', borderRadius: 8, border: '1px solid #334155' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f1f5f9' }}>
                  Detection Threshold:
                </span>
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#38bdf8', background: 'rgba(56, 189, 248, 0.1)', padding: '2px 8px', borderRadius: 4 }}>
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.80"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                disabled={isAnalyzing}
                style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer', margin: '4px 0' }}
              />
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
                <span>🛡️ High Sensitivity (0.20)</span>
                <span>⚖️ Balanced (0.50)</span>
                <span>🎯 Strict (0.80)</span>
              </div>
            </div>

            {errorMessage && (
              <div className="alert alert-danger">
                <span>⚠️</span>
                <div>{errorMessage}</div>
              </div>
            )}

            <button
              type="button"
              className="btn btn-analyze"
              onClick={handleAnalyze}
              disabled={!isReadyToAnalyze}
            >
              {isAnalyzing ? (
                <>
                  <div className="spinner"></div>
                  Analyzing Video ({analysisTimer}s)...
                </>
              ) : (
                <>🚀 Analyze Video</>
              )}
            </button>

            {!videoFile && (
              <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: 10, textAlign: 'center' }}>
                Upload a video to begin.
              </p>
            )}

            {videoFile && (topPoints.length < 2 || bottomPoints.length < 2) && (
              <p style={{ fontSize: '0.8rem', color: '#fb923c', marginTop: 10, textAlign: 'center' }}>
                Please mark both Top (≥2 pts) and Bottom (≥2 pts) wall boundaries.
              </p>
            )}
          </div>

          {/* Results Card */}
          <div className="card">
            <div className="card-title">
              <span>Prediction Result</span>
              {predictionResult && (
                <span style={{ fontSize: '0.8rem', color: '#38bdf8', fontWeight: 600 }}>
                  Model: {predictionResult.model_version?.toUpperCase() || 'V2'} (Threshold: {predictionResult.threshold_used?.toFixed(2) || '0.50'})
                </span>
              )}
            </div>

            {isAnalyzing && (
              <div style={{ textAlign: 'center', padding: '36px 12px' }}>
                <div className="spinner" style={{ margin: '0 auto 16px', width: 36, height: 36 }}></div>
                <div style={{ fontWeight: 600, color: '#f8fafc', marginBottom: 4 }}>
                  Processing spatial-temporal pipeline...
                </div>
                <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                  Extracting YOLO poses, wall geometry & running LSTM inference
                </div>
              </div>
            )}

            {!isAnalyzing && !predictionResult && (
              <div style={{ textAlign: 'center', padding: '36px 16px', color: '#64748b' }}>
                <div style={{ fontSize: '2.5rem', marginBottom: 8 }}>📊</div>
                <div>No analysis performed yet.</div>
                <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
                  Results will display here once inference completes.
                </div>
              </div>
            )}

            {!isAnalyzing && predictionResult && (
              <div>
                <div
                  className={`result-banner ${
                    predictionResult.prediction === 'Wall Crossing' ? 'crossing' : 'normal'
                  }`}
                >
                  <div style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Predicted Classification
                  </div>
                  <h2>{predictionResult.prediction}</h2>
                  <div style={{ fontSize: '0.8rem', opacity: 0.9 }}>
                    {predictionResult.prediction === 'Wall Crossing'
                      ? '⚠️ Wall intrusion activity detected'
                      : '✅ Normal activity within monitored scene'}
                  </div>
                </div>

                {predictionResult.warning && (
                  <div className="alert alert-warning">
                    <span>ℹ️</span>
                    <div>{predictionResult.warning}</div>
                  </div>
                )}

                <div className="metrics-list">
                  <div className="metric-box">
                    <div className="metric-label">Model Probability</div>
                    <div className="metric-value">
                      {(predictionResult.model_score * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>
                      Raw sigmoid output
                    </div>
                  </div>

                  <div className="metric-box">
                    <div className="metric-label">Processing Time</div>
                    <div className="metric-value">
                      {predictionResult.processing_time_seconds.toFixed(2)}s
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>
                      Local inference
                    </div>
                  </div>

                  <div className="metric-box">
                    <div className="metric-label">Max People Detected</div>
                    <div className="metric-value">
                      {predictionResult.people_detected}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>
                      In single frame
                    </div>
                  </div>

                  <div className="metric-box">
                    <div className="metric-label">Sampled Frames</div>
                    <div className="metric-value">
                      {predictionResult.frames_processed}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>
                      Every {predictionResult.frame_stride || 5}th frame
                    </div>
                  </div>

                  <div className="metric-box">
                    <div className="metric-label">Total Video Frames</div>
                    <div className="metric-value">
                      {predictionResult.total_video_frames ?? '—'}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: 2 }}>
                      Full video length
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Annotated YOLO Preview Gallery */}
      {predictionResult?.preview_frames?.length > 0 && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-title">
            <span>Annotated Detection Replay</span>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8', fontWeight: 'normal' }}>
              {predictionResult.preview_frames.length} sampled frames
              {predictionResult.total_video_frames ? ` of ${predictionResult.total_video_frames} total` : ''}
            </span>
          </div>

          <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: 16 }}>
            Full annotated video (wall line, person boxes, pose keypoints) replaying the sampled timeline — beside every processed frame below.
          </p>

          {/* Side-by-side: annotated video on the left, frame grid on the right */}
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 20,
            alignItems: 'flex-start',
          }}>
            {predictionResult.annotated_video && (
              <div style={{
                flex: '1 1 320px',
                minWidth: 280,
                background: 'rgba(15, 23, 42, 0.55)',
                border: '1px solid #334155',
                borderRadius: 10,
                overflow: 'hidden',
                position: 'sticky',
                top: 16,
              }}>
                <video
                  src={predictionResult.annotated_video}
                  controls
                  style={{ display: 'block', width: '100%', height: 'auto', background: '#000' }}
                />
                <div style={{
                  padding: '10px 12px',
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 14,
                  fontSize: '0.78rem',
                  color: '#cbd5e1',
                  borderTop: '1px solid #334155',
                }}>
                  <span><span style={{ color: '#f9a825' }}>▬</span> Wall top</span>
                  <span><span style={{ color: '#fb8c00' }}>▬</span> Wall bottom</span>
                  <span><span style={{ color: '#22c55e' }}>■</span> Person box</span>
                  <span><span style={{ color: '#facc15' }}>■</span> Skeleton</span>
                </div>
              </div>
            )}

            <div style={{
              flex: '2 1 420px',
              minWidth: 280,
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
              gap: 16,
              maxHeight: 750,
              overflowY: 'auto',
              paddingRight: 6
            }}>
              {predictionResult.preview_frames.map((frame) => (
                <div
                  key={frame.frame_number}
                  style={{
                    background: 'rgba(15, 23, 42, 0.55)',
                    border: '1px solid #334155',
                    borderRadius: 10,
                    overflow: 'hidden'
                  }}
                >
                  <img
                    src={frame.image}
                    alt={`Annotated sampled frame ${frame.frame_number}`}
                    loading="lazy"
                    style={{
                      display: 'block',
                      width: '100%',
                      height: 'auto',
                      background: '#000'
                    }}
                  />

                  <div style={{
                    padding: '10px 12px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: 8,
                    flexWrap: 'wrap',
                    fontSize: '0.8rem'
                  }}>
                    <span style={{ color: '#f1f5f9', fontWeight: 600 }}>
                      Sample {frame.frame_number}
                    </span>
                    <span style={{ color: '#94a3b8' }}>
                      {Number(frame.timestamp_seconds).toFixed(2)}s
                    </span>
                    <span style={{ color: '#38bdf8' }}>
                      {frame.people_detected} {frame.people_detected === 1 ? 'person' : 'people'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 18,
            marginTop: 16,
            paddingTop: 12,
            borderTop: '1px solid #334155',
            fontSize: '0.8rem',
            color: '#cbd5e1'
          }}>
            <span><span style={{ color: '#22c55e' }}>■</span> Person bounding box</span>
            <span><span style={{ color: '#facc15' }}>■</span> Pose skeleton</span>
            <span><span style={{ color: '#fb923c' }}>■</span> Keypoints</span>
            <span><span style={{ color: '#38bdf8' }}>⏱</span> Annotated video plays at real speed (1 sample = 5 video frames)</span>
          </div>
        </div>
      )}

    </div>
  );
}
