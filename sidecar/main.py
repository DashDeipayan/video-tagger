"""
CLIP inference sidecar.

Deliberately dumb: it maps pixels to vectors and text to vectors, and knows
nothing about tags, videos or the vocabulary. All domain logic lives in the Go
services. That keeps this process a pure, stateless function of its input --
which is exactly what lets it be replicated behind a load balancer on a GPU
pool, scaled independently of the API tier, or swapped for a larger model
without any Go code changing.

Endpoints:
    GET  /health        -> readiness + which device/model is loaded
    POST /embed/images  {"paths": [...]} -> {"embeddings": [[...]], "dim": n}
    POST /embed/text    {"texts": [...]} -> {"embeddings": [[...]], "dim": n}

All returned vectors are L2-normalised, so cosine similarity is a plain dot
product on the Go side.
"""

import logging
import os

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sidecar")


def pick_device() -> str:
    """Prefer Apple Silicon's GPU, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()
log.info("loading %s onto %s", MODEL_ID, DEVICE)
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)
log.info("model ready on %s", DEVICE)

app = FastAPI(title="CLIP embedding sidecar")


class ImageReq(BaseModel):
    paths: list[str]


class TextReq(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/embed/images")
@torch.inference_mode()
def embed_images(req: ImageReq):
    """Embed a batch of frames.

    The whole batch goes through the model in one forward pass: batching is what
    keeps a GPU saturated, and is the single biggest lever on inference cost per
    frame at scale.
    """
    if not req.paths:
        return {"embeddings": [], "dim": 0}
    try:
        images = [Image.open(p).convert("RGB") for p in req.paths]
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"cannot read frame: {e}") from e

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    # transformers>=5 returns an output object; pooler_output is the
    # already-projected embedding (applying visual_projection again fails
    # on a shape mismatch, which is how we confirmed it).
    feats = model.get_image_features(**inputs).pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return {"embeddings": feats.cpu().tolist(), "dim": feats.shape[-1]}


@app.post("/embed/text")
@torch.inference_mode()
def embed_text(req: TextReq):
    """Embed prompt strings into the same space as the images.

    This endpoint serves two callers: the tagger embeds the fixed vocabulary
    once at startup, and the search service embeds arbitrary user phrases at
    query time. Same function, and that symmetry is precisely what makes
    open-vocabulary search possible without re-processing any video.
    """
    if not req.texts:
        return {"embeddings": [], "dim": 0}
    inputs = processor(
        text=req.texts, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    feats = model.get_text_features(**inputs).pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return {"embeddings": feats.cpu().tolist(), "dim": feats.shape[-1]}
