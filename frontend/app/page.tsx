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
const MAX_UPLOAD_BYTES = 3_500_000;
const MAX_IMAGE_EDGE = 1920;

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

async function optimizeImageForUpload(file: File): Promise<SelectedImage> {
  const objectUrl = URL.createObjectURL(file);
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Could not read the selected image."));
      img.src = objectUrl;
    });

    const originalWidth = image.naturalWidth || image.width;
    const originalHeight = image.naturalHeight || image.height;
    const maxEdge = Math.max(originalWidth, originalHeight);
    const scale = maxEdge > MAX_IMAGE_EDGE ? MAX_IMAGE_EDGE / maxEdge : 1;
    const width = Math.max(1, Math.round(originalWidth * scale));
    const height = Math.max(1, Math.round(originalHeight * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Canvas context unavailable.");
    }
    context.drawImage(image, 0, 0, width, height);

    const jpegBlob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.88);
    });
    if (!jpegBlob) {
      throw new Error("Could not process image.");
    }

    const baseName = (file.name || `upload-${Date.now()}`).replace(/\.[^/.]+$/, "");
    return { blob: jpegBlob, name: `${baseName}.jpg` };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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
    async (file: File | null) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        setErrorMessage("Please upload an image file.");
        return;
      }

      try {
        const shouldOptimize = !ACCEPTED_TYPES.includes(file.type) || file.size > MAX_UPLOAD_BYTES;
        if (shouldOptimize) {
          const optimized = await optimizeImageForUpload(file);
          setNewImage(optimized.blob, optimized.name);
          return;
        }
        setNewImage(file, file.name || `upload-${Date.now()}.jpg`);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not process selected image.";
        setErrorMessage(message);
      }
    },
    [setNewImage]
  );

  const analyzeImage = useCallback(async () => {
    if (!selectedImage) {
      setErrorMessage("Please upload an image first.");
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
        headers: {
          Accept: "application/json",
        },
      });
      const contentType = response.headers.get("content-type") || "";
      const rawBody = await response.text();
      let payload: PredictionResponse | ErrorResponse | null = null;
      if (rawBody) {
        if (contentType.includes("application/json")) {
          try {
            payload = JSON.parse(rawBody) as PredictionResponse | ErrorResponse;
          } catch {
            payload = { error: "Invalid JSON response from server." };
          }
        } else {
          try {
            payload = JSON.parse(rawBody) as PredictionResponse | ErrorResponse;
          } catch {
            payload = { error: rawBody.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim() || "Unexpected non-JSON response from server." };
          }
        }
      }
      if (!response.ok) {
        const err = payload as ErrorResponse | null;
        const fallback = `Prediction failed (${response.status}).`;
        const msg = err?.details ? `${err.error || "Error"}: ${err.details}` : err?.error;
        setErrorMessage(msg || fallback);
        return;
      }
      if (!payload || typeof (payload as PredictionResponse).predicted_class !== "string") {
        setErrorMessage("Server returned an unexpected response format.");
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
    setSelectedImage(null);
    setPrediction(null);
    setErrorMessage(null);
    setPreviewUrl((currentUrl) => {
      revokePreviewUrl(currentUrl);
      return null;
    });
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [revokePreviewUrl]);

  useEffect(() => {
    return () => {
      revokePreviewUrl(previewUrl);
    };
  }, [previewUrl, revokePreviewUrl]);

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
          Upload an image from your gallery — our AI identifies the waste type and
          tells you exactly which bin to use. Sort smarter, live greener.
        </p>
        <div className="hero-stats">
          <div className="stat-chip">⚡ Real-time AI</div>
          <div className="stat-chip">♻️ 10+ Categories</div>
          <div className="stat-chip">📱 Mobile Upload</div>
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
              <h2>Upload Image</h2>
              <p className="muted">Choose a photo from your device gallery</p>
            </div>
          </div>

          <div className="camera-stage">
            {previewUrl ? (
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
                <span className="placeholder-hint">Upload from gallery below</span>
              </div>
            )}
          </div>

          <label className="upload-zone">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
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
            <span className="upload-hint">JPG · PNG · WEBP · HEIC (auto-optimized)</span>
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
                Upload an image from your gallery,
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
