# EcoSort Frontend (Next.js)

Minimalistisches Next.js-Frontend fuer EcoSort AI mit:
- Kamera-Live-Preview + Snapshot
- Manueller Bild-Upload
- Vorhersage ueber Next.js-Proxy (`/api/predict`)
- Modernes Glassmorphism-UI

## Voraussetzungen
- Node.js 20+
- Laufender FastAPI-Backend-Server auf `http://127.0.0.1:8000`

## Setup
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Frontend startet standardmaessig auf `http://localhost:3000`.

## Environment
In `.env.local`:
```env
ECOSORT_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_APP_NAME=EcoSort AI
```

## API-Fluss
Browser -> `POST /api/predict` (Next.js) -> `POST /predict?top_k=3` (FastAPI)

Damit bleibt die Backend-URL im Frontend gekapselt und CORS-Probleme werden reduziert.
