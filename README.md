# 🌿 EcoSort AI

> **Snap it. Sort it. Save the planet.**
> AI-powered waste classification that tells you exactly which bin to use — in real time.

EcoSort AI is a full-stack computer vision application built for waste management. Point your camera at any piece of trash, and the model tells you whether it's **battery, glass, metal, organic, paper, or plastic** — with a confidence score and bin recommendation.

**93% accuracy.** Six waste categories. Runs in the browser.

---

## What's inside

```
ecosort-ai/
├── 🧠 backend/          FastAPI inference server (TensorFlow + EfficientNetB0)
├── 💻 frontend/         Next.js 15 web app with live camera + upload
├── 📊 model/            Trained model weights + metrics + confusion matrix
├── 📓 notebooks/        Full training documentation (Jupyter)
├── ⚙️  scripts/          Training script (transfer learning pipeline)
└── ☸️  k8s/             Kubernetes deployment manifests
```

---

## How it works

```
Browser
  │  (photo or camera snapshot)
  ▼
Next.js frontend  ──POST /api/predict──▶  FastAPI backend
                                               │
                                          EfficientNetB0
                                          (224×224 RGB)
                                               │
                                          Top-3 predictions
                                          + bin guidance
                                               │
  Browser  ◀────────── JSON response ──────────┘
```

The model is a fine-tuned **EfficientNetB0** pre-trained on ImageNet. Images are resized to 224×224, normalized via the EfficientNet preprocessor, and classified into one of six categories. The backend wraps this in a FastAPI server; the Next.js frontend proxies requests through its own `/api/predict` route so secrets stay server-side.

---

## Model performance

| Class    | Precision | Recall | F1     |
|----------|-----------|--------|--------|
| Organic  | 99.1%     | 100%   | 99.6%  |
| Battery  | 98.3%     | 97.4%  | 97.9%  |
| Paper    | 97.3%     | 94.0%  | 95.7%  |
| Metal    | 86.4%     | 96.6%  | 91.1%  |
| Glass    | 90.5%     | 88.6%  | 89.6%  |
| Plastic  | 86.5%     | 82.1%  | 84.2%  |

**Overall test accuracy: 93.02% · Test loss: 0.2067**

Plastic and glass are the trickiest — they often look alike under bad lighting. The notebook in `notebooks/` has a full error analysis.

---

## Quickstart

### Prerequisites

- Python 3.10+
- Node.js 20+
- The trained model at `model/ecosort.keras` (or train your own — see below)

---

### 1 — Backend

```bash
cd backend
pip install -r requirements.txt

# copy and fill in paths if needed
cp .env.example .env

uvicorn main:app --reload --port 8000
```

Check it's alive:
```bash
curl http://127.0.0.1:8000/health
```

Test a prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@your_image.jpg"
```

---

### 2 — Frontend

```bash
cd frontend
cp .env.local.example .env.local   # set ECOSORT_API_BASE_URL if needed
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Use your camera or upload a photo — results appear instantly.

---

### 3 — Train your own model

```bash
# Expects data already split:
# data/split/{train,val,test}/<class_name>/

python scripts/train.py
```

Outputs saved to `model/`: weights, labels, metrics, confusion matrix, training history.

---

## Kubernetes (Minikube)

The backend ships with a production-ready Kubernetes manifest. Comes with readiness/liveness probes, resource limits, and a ClusterIP service.

```bash
# Point Docker at Minikube's daemon
eval "$(minikube -p minikube docker-env)"

# Build image inside Minikube
docker build -f backend/Dockerfile -t ecosort-backend:dev .

# Deploy
kubectl apply -f k8s/backend.yaml
kubectl -n ecosort rollout status deploy/ecosort-backend

# Forward port for local testing
kubectl -n ecosort port-forward svc/ecosort-backend 8000:8000
curl http://127.0.0.1:8000/health
```

Resource allocation per pod: `500m–2 CPU` / `1–4 Gi` memory.

---

## API reference

### `GET /health`
Returns model status and loaded class labels.

```json
{
  "status": "ok",
  "model": "ecosort.keras",
  "classes": ["battery", "glass", "metal", "organic", "paper", "plastic"]
}
```

### `POST /predict`
Upload an image, get waste classification back.

**Request:** `multipart/form-data` with a `file` field + optional `?top_k=3`

**Response:**
```json
{
  "predicted_class": "plastic",
  "confidence": 0.94,
  "recommended_bin": "Plastic Recycling Bin",
  "tip": "Rinse the container before disposing.",
  "top_predictions": [
    { "class": "plastic",  "confidence": 0.94 },
    { "class": "metal",    "confidence": 0.04 },
    { "class": "glass",    "confidence": 0.02 }
  ]
}
```

---

## Waste categories & bins

| Category | Bin                      |
|----------|--------------------------|
| 🔋 Battery  | Battery / E-Waste Collection |
| 🍶 Glass    | Glass Recycling Bin      |
| 🥫 Metal    | Metal Recycling Bin      |
| 🍌 Organic  | Compost / Organic Bin    |
| 📄 Paper    | Paper Recycling Bin      |
| 🧴 Plastic  | Plastic Recycling Bin    |

---

## Tech stack

| Layer     | Tech                                      |
|-----------|-------------------------------------------|
| Model     | TensorFlow / Keras · EfficientNetB0       |
| Backend   | FastAPI · Uvicorn · Pillow · NumPy        |
| Frontend  | Next.js 15 · React 19 · TypeScript        |
| Deploy    | Docker · Kubernetes (Minikube)            |
| Fonts     | Sora · IBM Plex Sans                      |

---

## Project layout (detailed)

```
backend/
  main.py              FastAPI app — /predict, /health endpoints
  requirements.txt     Python dependencies
  Dockerfile           Production Docker image (python:3.11-slim)

frontend/
  app/
    page.tsx           Main UI — camera, upload, results
    layout.tsx         Root layout + fonts
    globals.css        Dark bio-theme glassmorphism styles
    api/predict/
      route.ts         Next.js proxy → FastAPI
  .env.local.example   Environment variable template
  next.config.ts       Security headers, production config

model/
  ecosort.keras            Production model weights
  ecosort_best.keras       Best checkpoint from training
  labels.json              Class index mapping
  metrics.json             Test accuracy + loss
  classification_report.json  Per-class precision/recall/F1
  confusion_matrix.csv     Full confusion matrix
  history.csv              Training history per epoch

scripts/
  train.py             Transfer learning pipeline (EfficientNetB0)
                       — early stopping, LR scheduling, full reporting

notebooks/
  EcoSort_AI_Trainingsdokumentation_DE.ipynb
                       Full training documentation in German
                       — data analysis, error analysis, edge cases

k8s/
  backend.yaml         Deployment + Service + probes + resource limits
```

---

## Limitations

- **Plastic vs. glass**: visually similar materials, especially in poor lighting — these are the model's weakest categories (84% and 90% F1).
- **Single-item assumption**: model works best with one dominant object in frame. Mixed waste confuses it.
- **6 categories only**: e-waste (beyond batteries), clothing, and construction materials are not covered.

See `notebooks/EcoSort_AI_Trainingsdokumentation_DE.ipynb` for the full error analysis.

---

*Sort smarter. Waste less. 🌍*
