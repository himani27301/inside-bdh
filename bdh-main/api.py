from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import bdh

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "bdh_trained.pt"

app = FastAPI(title="Inside BDH API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bdh.BDH(bdh.BDHConfig()).to(device)

if not CHECKPOINT.exists():
    raise RuntimeError(
        f"Missing {CHECKPOINT.name}. Put your trained checkpoint beside api.py."
    )

model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
model.eval()


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "status": "ok",
        "project": "Inside BDH",
        "device": str(device),
        "checkpoint": CHECKPOINT.name,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    encoded = list(text.encode("utf-8"))
    if len(encoded) > 256:
        raise HTTPException(
            status_code=400,
            detail="Keep the demo input under 256 UTF-8 bytes for fast feedback.",
        )

    data = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        _, _, states = model(data, return_states=True)

    return {
        "input": text,
        "input_length": len(encoded),
        "model_label": "Independent toy BDH checkpoint trained on Tiny Shakespeare",
        "live": True,
        "layers": states,
    }
