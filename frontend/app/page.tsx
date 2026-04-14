"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type PredictionItem = {
  class: string;
  confidence: number;
  recommended_bin: string;
  tip: string;
};

type PredictionResponse = {
  predicted_class: string;
  confidence: number;
  recommended_bin: string;
  tip: string;
  top_predictions: PredictionItem[];
};

type ErrorResponse = {
  error?: string;
  details?: string;
};

type SelectedImage = {
  blob: Blob;
  name: string;
};

const ACCEPTED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const APP_NAME = process.env.NEXT_PUBLIC_APP_NAME || "EcoSort AI";

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value * 100));
}

function getBinStyle(bin: string): { color: string; bg: string } {
  const lower = bin.toLowerCase();
  if (lower.includes("organic") || lower.includes("bio") || lower.includes("compost") || lower.includes("food")) {
    return { color: "#4ade80", bg: "rgba(74,222,128,0.12)" };
  }
  if (lower.includes("recycl") || lower.includes("plastic") || lower.includes("metal") || lower.includes("gelb") || lower.includes("yellow")) {
    return { color: "#fbbf24", bg: "rgba(251,191,36,0.12)" };
  }
  if (lower.includes("paper") || lower.includes("cardboard") || lower.includes("papier")) {
    return { color: "#60a5fa", bg: "rgba(96,165,250,0.12)" };
  }
  if (lower.includes("glass") || lower.includes("glas")) {
    return { color: "#22d3ee", bg: "rgba(34,211,238,0.12)" };
  }
  if (lower.includes("hazard") || lower.includes("battery") || lower.includes("electronic") || lower.includes("sonder")) {
    return { color: "#fb923c", bg: "rgba(251,146,60,0.12)" };
  }
  return { color: "#94a3b8", bg: "rgba(148,163,184,0.1)" };
}

export default function HomePage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const hasImage = Boolean(selectedImage && previewUrl);

  const clearResultState = useCallback(() => {
    setPrediction(null);
    setErrorMessage(null);
  }, []);

  const revokePreviewUrl = useCallback((url: string | null) => {
    if (url?.startsWith("blob:")) {
      URL.revokeObjectURL(url);
    }
  }, []);

  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsVideoReady(false);
    setIsCameraOn(false);
  }, []);

  const setNewImage = useCallback(
    (blob: Blob, name: string, previewOverride?: string) => {
      clearResultState();
      setSelectedImage({ blob, name });
      setPreviewUrl((currentUrl) => {
        revokePreviewUrl(currentUrl);
        return previewOverride ?? URL.createObjectURL(blob);
      });
    },
    [clearResultState, revokePreviewUrl]
  );

  const onFilePicked = useCallback(
    (file: File | null) => {
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setErrorMessage("Please upload a JPG, PNG, or WEBP image.");
        return;
      }
      setNewImage(file, file.name || `upload-${Date.now()}.jpg`);
    },
    [setNewImage]
  );

  const startCamera = useCallback(async () => {
    clearResultState();
    setIsVideoReady(false);
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage("This browser does not support camera access.");
      return;
    }
    try {
      cleanupStream();
      const candidates: MediaStreamConstraints[] = [
        { video: { facingMode: { exact: "environment" } }, audio: false },
        { video: { facingMode: { ideal: "environment" } }, audio: false },
        { video: true, audio: false },
      ];
      let mediaStream: MediaStream | null = null;
      let lastError: unknown = null;
      for (const constraints of candidates) {
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch (err) {
          lastError = err;
        }
      }
      if (!mediaStream) {
        throw lastError || new Error("Unable to access camera.");
      }
      streamRef.current = mediaStream;
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.muted = true;
        videoRef.current.setAttribute("playsinline", "true");
        videoRef.current.setAttribute("webkit-playsinline", "true");
        const playPromise = videoRef.current.play();
        if (playPromise && typeof playPromise.catch === "function") {
          await playPromise.catch(() => undefined);
        }
      }
      setIsCameraOn(true);
    } catch (error) {
      let message = "Camera could not be started.";
      if (error instanceof Error) {
        if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
          message = "No camera found on this device.";
        } else if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
          message = "Camera access denied. Please allow camera permissions in your browser settings.";
        } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
          message = "Camera is already in use by another application.";
        } else if (error.name === "OverconstrainedError") {
          message = "Camera does not meet the required constraints.";
        } else if (error.name === "SecurityError") {
          message = "Camera access requires a secure (HTTPS) connection.";
        } else {
          message = error.message;
        }
      }
      setErrorMessage(message);
    }
  }, [cleanupStream, clearResultState]);

  const stopCamera = useCallback(() => {
    cleanupStream();
  }, [cleanupStream]);

  const waitForVideoFrame = useCallback(async (video: HTMLVideoElement, timeoutMs = 3000): Promise<boolean> => {
    const startedAt = Date.now();
    if (video.paused) {
      try {
        await video.play();
      } catch {
        // iOS Safari can reject autoplay until stream stabilizes; polling continues.
      }
    }
    while (Date.now() - startedAt < timeoutMs) {
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        return true;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 120));
    }
    return false;
  }, []);

  const takeSnapshot = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) {
      setErrorMessage("Camera not ready.");
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ready = (video.videoWidth > 0 && video.videoHeight > 0) || (await waitForVideoFrame(video, 7000));
    if (!ready || !video.videoWidth || !video.videoHeight) {
      setErrorMessage("Video feed is still warming up. Please wait a moment and try again.");
      return;
    }
    setIsVideoReady(true);
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setErrorMessage("Canvas context unavailable.");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.92);
    });
    if (!blob) {
      setErrorMessage("Could not capture snapshot.");
      return;
    }
    setNewImage(blob, `snapshot-${Date.now()}.jpg`, dataUrl);
    cleanupStream();
  }, [cleanupStream, setNewImage, waitForVideoFrame]);

  const analyzeImage = useCallback(async () => {
    if (!selectedImage) {
      setErrorMessage("Please select or capture an image first.");
      return;
    }
    clearResultState();
    setIsLoading(true);
    try {
      const form = new FormData();
      form.append("file", selectedImage.blob, selectedImage.name);
      const response = await fetch("/api/predict", {
        method: "POST",
        body: form,
      });
      const payload = (await response.json()) as PredictionResponse | ErrorResponse;
      if (!response.ok) {
        const err = payload as ErrorResponse;
        const msg = err.details ? `${err.error || "Error"}: ${err.details}` : err.error;
        setErrorMessage(msg || "Prediction failed.");
        return;
      }
      setPrediction(payload as PredictionResponse);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Network error";
      setErrorMessage(`Request failed: ${message}`);
    } finally {
      setIsLoading(false);
    }
  }, [clearResultState, selectedImage]);

  const resetAll = useCallback(() => {
    cleanupStream();
    setSelectedImage(null);
    setPrediction(null);
    setErrorMessage(null);
    setPreviewUrl((currentUrl) => {
      revokePreviewUrl(currentUrl);
      return null;
    });
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [cleanupStream, revokePreviewUrl]);

  useEffect(() => {
    if (!isCameraOn || isVideoReady) return;
    const timer = window.setInterval(() => {
      const video = videoRef.current;
      if (!video) return;
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        setIsVideoReady(true);
      }
    }, 200);
    return () => window.clearInterval(timer);
  }, [isCameraOn, isVideoReady]);

  useEffect(() => {
    return () => {
      cleanupStream();
      revokePreviewUrl(previewUrl);
    };
  }, [cleanupStream, previewUrl, revokePreviewUrl]);

  const topPredictions = useMemo(() => prediction?.top_predictions ?? [], [prediction]);

  return (
    <main className="page-shell">
      <div className="background-orb orb-one" />
      <div className="background-orb orb-two" />
      <div className="background-orb orb-three" />

      {/* ── Hero ── */}
      <header className="hero glass-panel">
        <div className="hero-badge">
          <span className="hero-badge-dot" />
          Computer Vision · Waste Intelligence
        </div>
        <h1>
          <span className="hero-leaf">🌿</span> {APP_NAME}
        </h1>
        <p className="subtitle">
          Snap a photo or upload an image — our AI identifies the waste type and
          tells you exactly which bin to use. Sort smarter, live greener.
        </p>
        <div className="hero-stats">
          <div className="stat-chip">⚡ Real-time AI</div>
          <div className="stat-chip">♻️ 10+ Categories</div>
          <div className="stat-chip">📱 Camera Ready</div>
          <div className="stat-chip">🌍 Go Green</div>
        </div>
      </header>

      {/* ── Content grid ── */}
      <section className="content-grid">

        {/* Left: capture */}
        <article className="glass-panel control-card">
          <div className="card-header">
            <span className="card-icon">📷</span>
            <div>
              <h2>Capture Image</h2>
              <p className="muted">Live camera or file upload</p>
            </div>
          </div>

          <div className={`camera-stage${isCameraOn ? " is-active" : ""}`}>
            {isCameraOn ? (
              <video
                ref={videoRef}
                className="camera-feed"
                playsInline
                muted
                autoPlay
                onLoadedMetadata={() => setIsVideoReady(true)}
                onCanPlay={() => setIsVideoReady(true)}
                onPlaying={() => setIsVideoReady(true)}
              />
            ) : previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="Selected item" className="camera-feed" />
            ) : (
              <div className="camera-placeholder">
                <div className="placeholder-icon">
                  <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                </div>
                <p>No image selected</p>
                <span className="placeholder-hint">Use camera or upload below</span>
              </div>
            )}
            <canvas ref={canvasRef} className="hidden-canvas" />
          </div>

          <div className="button-row">
            {!isCameraOn ? (
              <button type="button" className="btn btn-cam" onClick={startCamera}>
                📸 Start Camera
              </button>
            ) : (
              <>
                <button
                  type="button"
                  className="btn btn-cam"
                  onClick={takeSnapshot}
                  disabled={!isVideoReady}
                >
                  {isVideoReady ? "📷 Take Snapshot" : "📷 Take Snapshot (warming up)"}
                </button>
                <button type="button" className="btn btn-ghost" onClick={stopCamera}>
                  ✕ Stop Camera
                </button>
              </>
            )}
          </div>

          <label className="upload-zone">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp"
              onChange={(event) => onFilePicked(event.target.files?.[0] || null)}
              className="upload-input"
            />
            <span className="upload-label">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17,8 12,3 7,8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload Image
            </span>
            <span className="upload-hint">JPG · PNG · WEBP</span>
          </label>

          <div className="button-row">
            <button
              type="button"
              className="btn btn-primary"
              onClick={analyzeImage}
              disabled={!hasImage || isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner" />
                  Analyzing...
                </>
              ) : (
                <>🔍 Analyze Image</>
              )}
            </button>
            <button type="button" className="btn btn-ghost" onClick={resetAll}>
              ↺ Reset
            </button>
          </div>

          {errorMessage && (
            <div className="status error">
              <span>⚠️</span>
              <p>{errorMessage}</p>
            </div>
          )}
        </article>

        {/* Right: results */}
        <article className="glass-panel result-card">
          <div className="card-header">
            <span className="card-icon">🌱</span>
            <div>
              <h2>Classification Result</h2>
              <p className="muted">AI-powered waste detection</p>
            </div>
          </div>

          {prediction ? (
            <>
              <div className="primary-result">
                <span className="label">Detected Waste Type</span>
                <strong className="pred-class">{prediction.predicted_class}</strong>
                <div className="confidence-row">
                  <div className="confidence-bar-wrap">
                    <div
                      className="confidence-bar-fill"
                      style={{ width: `${clampPercent(prediction.confidence)}%` }}
                    />
                  </div>
                  <span className="confidence-pct">
                    {clampPercent(prediction.confidence).toFixed(1)}%
                  </span>
                </div>
              </div>

              {(() => {
                const binStyle = getBinStyle(prediction.recommended_bin);
                return (
                  <div
                    className="recommendation"
                    style={{
                      borderColor: `${binStyle.color}38`,
                      background: binStyle.bg,
                    }}
                  >
                    <div className="bin-header">
                      <span className="label">Recommended Bin</span>
                      <span
                        className="bin-dot"
                        style={{ background: binStyle.color, color: binStyle.color }}
                      />
                    </div>
                    <strong className="bin-name" style={{ color: binStyle.color }}>
                      {prediction.recommended_bin}
                    </strong>
                    <p className="tip-text">💡 {prediction.tip}</p>
                  </div>
                );
              })()}

              <div className="top-list">
                <h3>Top 3 Predictions</h3>
                {topPredictions.map((item) => {
                  const percent = clampPercent(item.confidence);
                  return (
                    <div key={item.class} className="pred-item">
                      <div className="pred-line">
                        <span>{item.class}</span>
                        <span className="pred-pct">{percent.toFixed(1)}%</span>
                      </div>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${percent}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="result-empty">
              <div className="empty-icon">♻️</div>
              <p className="empty-title">No analysis yet</p>
              <p className="muted">
                Upload an image or take a snapshot,
                <br />
                then hit Analyze Image.
              </p>
            </div>
          )}
        </article>
      </section>

      <footer className="page-footer">
        <p>EcoSort AI · Powered by Computer Vision · Sort smarter, live greener 🌍</p>
      </footer>
    </main>
  );
}
