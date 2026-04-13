# 🌿 EcoSort AI — Präsentation

> **Abfallklassifikation mit künstlicher Intelligenz**
> Foto aufnehmen → KI erkennt Mülltyp → richtiger Behälter wird angezeigt

---

## Aufteilung

| Person | Thema |
|--------|-------|
| **Person 1** | Projektüberblick, Problemstellung, System-Architektur |
| **Person 2** | Datensatz, Modellaufbau, Trainingsprozess |
| **Person 3** | Ergebnisse, Fehleranalyse, Deployment & Live-Demo |

---
---

# 👤 Person 1 — Projektüberblick & Architektur

---

## Was ist EcoSort AI?

- **Ziel:** Ein Bild von Müll aufnehmen → KI erkennt automatisch, in welchen Behälter er gehört
- **Problem:** Falsch sortierter Müll kostet Österreich jährlich Millionen € in Nachsortierung
- **Lösung:** Computer-Vision-Modell, das 6 Müllkategorien unterscheidet

### Die 6 Kategorien

| Kategorie | Behälter |
|-----------|----------|
| 🔋 Battery | Batterie / E-Müll Sammelstelle |
| 🍶 Glass | Glascontainer |
| 🥫 Metal | Metallrecycling |
| 🍌 Organic | Biotonne / Kompost |
| 📄 Paper | Altpapier |
| 🧴 Plastic | Gelber Sack / Plastik |

---

## Wie funktioniert das System?

```
Nutzer (Browser / Handy)
        │
        │  Foto (Kamera oder Upload)
        ▼
  Next.js Frontend
  (React · TypeScript)
        │
        │  POST /api/predict
        ▼
  FastAPI Backend
  (Python · TensorFlow)
        │
        │  Bild vorverarbeiten
        ▼
  EfficientNetB0 Modell
        │
        │  Top-3 Vorhersagen + Konfidenz
        ▼
  Ergebnis im Browser
  (Klasse · Behälter · Tipp)
```

---

## Tech-Stack auf einen Blick

| Schicht | Technologie |
|---------|-------------|
| **KI / Modell** | TensorFlow · Keras · EfficientNetB0 |
| **Backend** | FastAPI · Python 3.11 · Uvicorn |
| **Frontend** | Next.js 15 · React 19 · TypeScript |
| **Container** | Docker |
| **Deployment** | Kubernetes (LeoCloud) |
| **CI/CD** | GitHub Actions |

---

## Live-Architektur auf LeoCloud

```
Internet
    │
    ▼
Ingress  (https://if200166.cloud.htl-leonding.ac.at)
    │
    ├──▶  ecosort-frontend  (Pod · Next.js · Port 3000)
    │           │
    │           │  intern: http://ecosort-backend:8000
    │           ▼
    └──▶  ecosort-backend   (Pod · FastAPI · Port 8000)
                │
                └── Modell liegt im Container-Image
```

Kubernetes übernimmt: automatischer Neustart bei Absturz, Health-Checks, Rolling Updates ohne Downtime.

---
---

# 👤 Person 2 — Datensatz, Modell & Training

---

## Der Datensatz

- **Gesamtbilder:** 4.650 Fotos
- **Klassen:** 6 (gleichmäßig verteilt → ~775 Bilder pro Klasse)
- **Aufteilung:**

```
data/split/
├── train/   →  70%  ≈ 3.255 Bilder  (Modell lernt darauf)
├── val/     →  15%  ≈  698 Bilder   (Hyperparameter-Tuning)
└── test/    →  15%  ≈  702 Bilder   (finale Bewertung)
```

### Warum ausgeglichene Klassen wichtig sind

> Wenn eine Klasse 10× mehr Bilder hat als eine andere, lernt das Modell einfach immer die häufige Klasse vorherzusagen — und erreicht trotzdem hohe Accuracy. Bei uns: **jede Klasse gleich groß** → Accuracy ist wirklich aussagekräftig.

---

## Bildvorverarbeitung

Bevor ein Bild ins Modell kommt, passieren diese Schritte:

```
Rohbild (beliebige Größe)
    │
    ▼
Resize → 224 × 224 Pixel
    │
    ▼
RGB sicherstellen (kein RGBA / Graustufe)
    │
    ▼
EfficientNet-Preprocessing
(Pixelwerte normalisiert nach ImageNet-Standard)
    │
    ▼
Modell-Input: (224, 224, 3) Float-Tensor
```

---

## Data Augmentation — künstlich mehr Daten erzeugen

Nur während des Trainings werden Bilder zufällig verändert. Das Modell sieht jedes Mal eine leicht andere Version → wird robuster.

```python
data_augmentation = Sequential([
    RandomFlip("horizontal"),      # Spiegeln
    RandomRotation(0.08),          # ± 4.6° drehen
    RandomZoom(0.1),               # ± 10% zoomen
    RandomContrast(0.1),           # Kontrast ± 10%
])
```

> **Ziel:** Das Modell soll auch bei schlechter Beleuchtung, leicht schrägem Winkel oder verschiedenen Hintergründen funktionieren.

---

## Das Modell — EfficientNetB0

### Was ist Transfer Learning?

```
ImageNet-Training
(1,2 Mio. Bilder · 1000 Klassen · Monate Rechenzeit)
        │
        ▼
  Vortrainierte Gewichte
  (das Modell "kennt" bereits Kanten, Texturen, Formen)
        │
        ▼
  Wir übernehmen diese Gewichte
  und trainieren nur den Kopf neu
        │
        ▼
  EcoSort-Modell
  (6 Müll-Klassen · unsere 4.650 Bilder · Stunden)
```

### Modellarchitektur

```
Input (224×224×3)
    │
    ▼  [gefroren in Phase 1]
EfficientNetB0 Backbone
(Merkmale extrahieren: Kanten, Texturen, Formen)
    │
    ▼
GlobalAveragePooling2D
(räumliche Dimensionen → 1D-Vektor)
    │
    ▼
Dropout (0.2)
(verhindert Overfitting — 20% der Neuronen zufällig deaktiviert)
    │
    ▼
Dense (6 Neuronen · Softmax)
(Wahrscheinlichkeit für jede der 6 Klassen)
    │
    ▼
Output: [0.03, 0.02, 0.01, 0.91, 0.02, 0.01]
         bat  glass metal organic paper plastic
                           ↑ Modell sagt: Organic (91%)
```

---

## Training in 2 Phasen

### Phase 1 — Kopf trainieren (Head Training)

```
EfficientNetB0-Backbone:  GEFROREN  ❄️
Neuer Klassifikationskopf: TRAINIERBAR ✅

Epochen:        10
Lernrate:       0.001  (schnelles Lernen)
Optimizer:      Adam
Loss-Funktion:  Sparse Categorical Crossentropy
```

> Der Backbone bleibt eingefroren — wir lernen nur, wie man die bereits extrahierten Merkmale in 6 Klassen einteilt. Das geht schnell.

---

### Phase 2 — Feinabstimmung (Fine-Tuning)

```
EfficientNetB0-Backbone:  TOP 20% AUFGETAUT  🔥
Klassifikationskopf:      TRAINIERBAR ✅

Epochen:        15
Lernrate:       0.00001  (sehr langsam — keine großen Gewichtssprünge)
Optimizer:      Adam
```

> Jetzt passen sich die oberen Schichten des Backbones leicht an unsere Müll-Bilder an. Lernrate sehr niedrig, damit die vortrainierten Gewichte nicht "vergessen" werden.

---

## Callbacks — Automatische Steuerung des Trainings

```python
# 1. Bestes Modell speichern
ModelCheckpoint(monitor="val_accuracy", save_best_only=True)

# 2. Frühzeitig stoppen wenn kein Fortschritt mehr
EarlyStopping(monitor="val_loss", patience=5)
# → Training stoppt automatisch nach 5 Epochen ohne Verbesserung

# 3. Lernrate automatisch reduzieren
ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)
# → Falls val_loss 2 Epochen stagniert → Lernrate × 0.2
```

> **Ohne Callbacks:** Modell könnte overfittten, Rechenzeit wird verschwendet.
> **Mit Callbacks:** Automatisch das beste Modell wird gespeichert, Training stoppt zum richtigen Zeitpunkt.

---
---

# 👤 Person 3 — Ergebnisse, Fehler & Deployment

---

## Testergebnisse

Das Modell wurde auf **702 völlig neue Bilder** (Test-Set) getestet — Bilder, die das Modell **nie** gesehen hat.

```
Test Accuracy:  93,02 %
Test Loss:      0,2067
```

### Ergebnisse pro Klasse

| Klasse | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| 🍌 Organic | 99,2 % | **100 %** | **99,6 %** |
| 🔋 Battery | 98,3 % | 97,4 % | 97,9 % |
| 📄 Paper | 97,3 % | 94,0 % | 95,7 % |
| 🥫 Metal | 86,3 % | 96,6 % | 91,1 % |
| 🍶 Glass | 91,2 % | 88,0 % | 89,6 % |
| 🧴 Plastic | 86,5 % | 82,1 % | 84,2 % |

> **Precision:** Von allem, was das Modell als Plastik bezeichnet — wie viel ist wirklich Plastik?
> **Recall:** Von allem echten Plastik im Test-Set — wie viel hat das Modell gefunden?

---

## Was bedeuten diese Zahlen?

### Beste Klasse: Organic (F1 99,6 %)
Bioabfall hat sehr charakteristische visuelle Merkmale (Farbe, Textur von Lebensmitteln) → leicht zu erkennen.

### Schwächste Klasse: Plastic (F1 84,2 %)
Plastik verwechselt das Modell am häufigsten mit **Metal** oder **Glass** — weil:
- Ähnliche glatte Oberflächen
- Transparente Flaschen sehen je nach Licht wie Glas aus
- Metallic-beschichtetes Plastik (z. B. Chipstüten) sieht aus wie Metall

---

## Fehleranalyse — Wo macht das Modell Fehler?

### Häufigste Verwechslungen (aus der Confusion Matrix)

```
Plastic  →  fälschlicherweise als  Metal    erkannt
Glass    →  fälschlicherweise als  Plastic  erkannt
Metal    →  fälschlicherweise als  Glass    erkannt
```

### Robustheitstests

| Test | Ergebnis |
|------|----------|
| Leichte Rotation (±15°) | ✅ Robust |
| Helligkeitsänderung | ✅ Robust |
| Blur / Unschärfe | ✅ Meist robust |
| Sehr niedrige Auflösung (32×32) | ⚠️ Unsicher |
| Fremdes Bild (kein Müll) | ❌ Trotzdem eine Klasse zugewiesen |

> **Wichtig:** Softmax erzwingt immer eine Klasse — auch bei einem Bild einer Katze gibt das Modell eine Müll-Kategorie aus. Für Produktion: Konfidenz-Schwellwert nötig (z. B. unter 60% → "Bitte näherfotografieren").

---

## CI/CD — Automatisches Deployment

Jedes Mal wenn Code auf `main` gepusht wird:

```
Push auf main
    │
    ▼
GitHub Actions startet
    │
    ├── Backend-Image bauen & pushen  →  ghcr.io/kyawit/ecosortai-backend
    └── Frontend-Image bauen & pushen →  ghcr.io/kyawit/ecosortai-frontend
    │
    ▼
LeoCloud: neue Pods starten mit neuem Image
    │
    ├── Rollout Backend  (max. 5 Minuten)
    └── Rollout Frontend (max. 2 Minuten)
    │
    ▼
Smoke-Tests (automatisch)
    ├── GET /health  →  200 OK?
    └── GET /        →  200 OK?
    │
    ▼
✅ Fertig — App live unter https://if200166.cloud.htl-leonding.ac.at
❌ Fehler  — Automatischer Rollback auf letzte funktionierende Version
```

---

## Live-Demo

1. Browser öffnen → `https://if200166.cloud.htl-leonding.ac.at`
2. Kamera starten oder Bild hochladen
3. „Analyze Image" klicken
4. Ergebnis: Klasse + Konfidenz + empfohlener Behälter + Tipp

---

## Fazit & Ausblick

### Was wurde erreicht?
- ✅ 93 % Genauigkeit auf 6 Müllkategorien
- ✅ Vollständige Web-App mit Kamera-Support
- ✅ Automatisches Deployment auf Kubernetes
- ✅ CI/CD Pipeline mit Auto-Rollback

### Was könnte noch verbessert werden?
- **Mehr Klassen** — Kleidung, Bauschutt, E-Müll generell
- **Konfidenz-Schwelle** — Bei < 60% Unsicherheit Hinweis anzeigen
- **Mehr Trainingsdaten** — besonders für Plastik & Glas
- **Mehrsprachigkeit** — App bisher nur Englisch

---

*EcoSort AI · HTL Leonding · DSAI*
