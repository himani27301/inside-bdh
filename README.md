# Inside BDH — See Sparsity Happen

<p align="center">
  <b>Change the input. Keep the checkpoint fixed. Watch BDH's sparse internal state change.</b>
</p>

<p align="center">
  <a href="https://inside-bdh-1.onrender.com"><b>🌐 Live Demo</b></a>
  &nbsp;•&nbsp;
  <a href="https://inside-bdh.onrender.com"><b>⚙️ Backend API</b></a>
  &nbsp;•&nbsp;
  <a href="https://github.com/himani27301/inside-bdh"><b>💻 Repository</b></a>
</p>

---

## ✨ Project Preview

![Inside BDH UI](assets/hero.png)

**Inside BDH** is an interactive educational artifact for exploring one specific mechanism inside a Dragon Hatchling (BDH) model:

> **BDH converts a signed latent projection into an input-dependent, non-negative sparse state; changing the input changes which latent units survive.**

Instead of only reading about sparsity, the learner changes the input and inspects the model's **real intermediate states** layer-by-layer.

---

## 🎯 What problem are we solving?

Sparse neural computation is usually explained through equations, papers, or static diagrams. That makes it hard to build intuition for what actually changes inside a model when the input changes.

Inside BDH turns that idea into an experiment:

1. choose two sequences
2. keep the same trained checkpoint
3. run both inputs
4. compare the internal sparse states
5. inspect which latent units survive
6. see exactly what ReLU clips to zero

The important idea is that the **model stays fixed while the input changes**.

---

## 🧪 What can the learner inspect?

### Same checkpoint, different input
The comparison keeps the model fixed and changes only the input.

### Six BDH layers
For each layer, the UI compares:
- X sparse activity
- Y sparse activity
- XY interaction activity
- maximum activation
- strongest latent units
- head-level activity
- overlap between top units

### Before ReLU → After ReLU
The artifact exposes sampled values before and after ReLU.

```text
pre-ReLU:  -0.82
post-ReLU:  0
```

```text
pre-ReLU:   1.24
post-ReLU:  1.24
```

Negative values are clipped to zero while positive values survive.

### ReLU consistency check
The UI compares expected zeroing behaviour with the observed post-ReLU zero rate for the sampled values.

### ReLU challenge
A short prediction activity asks the learner to predict the output of ReLU before revealing the result.

---

## 📊 Example Output

![Inside BDH results](assets/results.png)

The output is generated from a real forward pass through the deployed toy checkpoint rather than hard-coded example values.

---

## 🏗️ Architecture

![Inside BDH architecture](assets/architecture.svg)

```text
React / Vite frontend
        ↓
FastAPI backend
        ↓
PyTorch BDH checkpoint
        ↓
Intermediate tensor extraction
        ↓
Sparse-state / ReLU / head visualizations
```

---

## 🔴 What is live?

| Component | Status |
|---|---|
| User text input | **Live** |
| BDH forward pass | **Live** |
| Layer measurements | **Live** |
| Sparse activity | **Live** |
| Strongest-unit extraction | **Live** |
| Head activity | **Live** |
| Pre-ReLU / post-ReLU values | **Live** |
| UI animation and layout | Presentation layer |
| Published BDH claims | Referenced research |

---

## 🤖 Model used in this artifact

For the educational demo we use a **small toy BDH checkpoint** trained on **Tiny Shakespeare**.

```text
Layers:       6
Embedding:    256
Heads:        4
Vocabulary:   256 byte values
Dataset:      Tiny Shakespeare
```

This checkpoint exists so that the internal mechanism can be explored interactively.

> It is **not** a reproduction of Pathway's large-scale published BDH results. The interface deliberately separates observations from our toy checkpoint from claims reported in published work.

---

## 🧰 Tech stack

### Frontend
- React
- Vite
- responsive CSS
- animated interaction and visualization

### Backend
- FastAPI
- Uvicorn
- Python

### Model
- PyTorch
- BDH implementation
- trained toy checkpoint

### Deployment
- Render Static Site — frontend
- Render Web Service — backend
- GitHub — source repository

---

## 🚀 Live links

**Frontend:** https://inside-bdh-1.onrender.com

**Backend:** https://inside-bdh.onrender.com

> The hosted backend may take a few seconds to wake after inactivity.

---

## 🧭 Suggested demo

**Input A**
```text
The king walked into the castle.
```

**Input B**
```text
A A A A A A A A A A A A
```

Then:

1. click **Compare Internal States**
2. switch between Layers 1–6
3. compare sparse activity
4. inspect strongest units
5. inspect head activity
6. inspect **Before ReLU → After ReLU**
7. try the Noise preset
8. answer the ReLU challenge

---

## 📁 Project structure

```text
inside-bdh/
│
├── bdh-main/
│   ├── api.py
│   ├── bdh.py
│   ├── train.py
│   ├── analyze.py
│   └── requirements.txt
│
├── inside-bdh/
│   ├── src/
│   │   ├── App.jsx
│   │   └── index.css
│   ├── public/
│   └── package.json
│
├── assets/
│   ├── hero.png
│   ├── results.png
│   └── architecture.svg
│
├── README.md
├── SOURCES_AND_LICENSES.md
└── AI_DISCLOSURE.md
```

---

## 🖥️ Run locally

### 1. Clone

```bash
git clone https://github.com/himani27301/inside-bdh.git
cd inside-bdh
```

### 2. Backend

```bash
cd bdh-main
pip install -r requirements.txt
uvicorn api:app --reload
```

Backend:
```text
http://127.0.0.1:8000
```

### 3. Frontend

Open another terminal:

```bash
cd inside-bdh
npm install
npm run dev
```

Frontend:
```text
http://localhost:5173
```

For local development, point the frontend API URL to:
```text
http://127.0.0.1:8000/analyze
```

---

## 🧠 Educational design

The artifact is deliberately centered around **one testable claim** rather than trying to visualize every mechanism in BDH.

```text
Change input
    ↓
Run same checkpoint
    ↓
Inspect sparse state
    ↓
Compare layers / units / heads
    ↓
Inspect ReLU transformation
    ↓
Test understanding
```

---

## ⚠️ Limitations

- the checkpoint is intentionally small
- it is trained on a toy corpus
- strongest-unit overlap does not by itself imply semantic interpretability
- only selected internal measurements are visualized
- observations from this checkpoint should not be generalized to every BDH system
- published BDH claims and our toy-checkpoint observations are kept separate

---

## 📚 Credits, reuse and disclosure

The BDH implementation used by this project is based on Pathway's public BDH codebase.

See:
- `SOURCES_AND_LICENSES.md`
- `AI_DISCLOSURE.md`

for reused code, external resources, licenses, and tooling disclosure.

---

## 💡 Core takeaway

**Don't just tell learners that sparsity happens. Let them change the input and watch the sparse internal state change.**
