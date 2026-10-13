#!/usr/bin/env python3
"""Generate an editable .excalidraw HLD for the video tagging system."""
import json, random

random.seed(7)
els = []
bound = {}           # element id -> list of bound element refs
_n = [0]

def nid(p="e"):
    _n[0] += 1
    return "%s%03d" % (p, _n[0])

def base(i, t, x, y, w, h, stroke="#1e1e1e", bg="transparent", **kw):
    e = {
        "id": i, "type": t, "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": stroke, "backgroundColor": bg, "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": {"type": 3},
        "seed": random.randint(1, 2**31), "version": 1,
        "versionNonce": random.randint(1, 2**31), "isDeleted": False,
        "boundElements": [], "updated": 1, "link": None, "locked": False,
    }
    e.update(kw)
    return e

def box(x, y, w, h, label, sub="", stroke="#1e1e1e", bg="transparent",
        font=1, size=20, dashed=False):
    """Rounded rect with centred bound text."""
    bid, tid = nid("box"), nid("txt")
    r = base(bid, "rectangle", x, y, w, h, stroke, bg)
    if dashed:
        r["strokeStyle"] = "dashed"
        r["roundness"] = {"type": 3}
    text = label + ("\n" + sub if sub else "")
    lines = text.count("\n") + 1
    lh = size * 1.25
    t = base(tid, "text", x + 8, y + (h - lines * lh) / 2, w - 16, lines * lh,
             stroke, "transparent",
             text=text, fontSize=size, fontFamily=font, textAlign="center",
             verticalAlign="middle", containerId=bid, originalText=text,
             lineHeight=1.25, roundness=None)
    r["boundElements"] = [{"type": "text", "id": tid}]
    els.extend([r, t])
    return bid

def label(x, y, text, size=14, font=3, color="#5c5f66", align="left", w=None):
    tid = nid("lbl")
    lines = text.count("\n") + 1
    lh = size * 1.25
    width = w if w else max(len(l) for l in text.split("\n")) * size * 0.62
    els.append(base(tid, "text", x, y, width, lines * lh, color, "transparent",
                    text=text, fontSize=size, fontFamily=font, textAlign=align,
                    verticalAlign="top", containerId=None, originalText=text,
                    lineHeight=1.25, roundness=None))
    return tid

def arrow(a, b, color="#1e1e1e", dashed=False, start_arrow=None,
          end_arrow="arrow", bend=None):
    """Bound arrow between two element ids."""
    aid = nid("arr")
    ra = next(e for e in els if e["id"] == a)
    rb = next(e for e in els if e["id"] == b)
    ax, ay = ra["x"] + ra["width"] / 2, ra["y"] + ra["height"] / 2
    bx, by = rb["x"] + rb["width"] / 2, rb["y"] + rb["height"] / 2
    pts = [[0, 0], [bx - ax, by - ay]] if bend is None else \
          [[0, 0], bend, [bx - ax, by - ay]]
    e = base(aid, "arrow", ax, ay, abs(bx - ax), abs(by - ay), color,
             "transparent",
             points=pts, startBinding={"elementId": a, "focus": 0, "gap": 6},
             endBinding={"elementId": b, "focus": 0, "gap": 6},
             startArrowhead=start_arrow, endArrowhead=end_arrow,
             roundness={"type": 2}, elbowed=False)
    if dashed:
        e["strokeStyle"] = "dashed"
    els.append(e)
    ra["boundElements"].append({"id": aid, "type": "arrow"})
    rb["boundElements"].append({"id": aid, "type": "arrow"})
    return aid


# ───────── palette ─────────
BLUE   = ("#1971c2", "#a5d8ff")
GREEN  = ("#2f9e44", "#b2f2bb")
ORANGE = ("#e8590c", "#ffd8a8")
RED    = ("#e03131", "#ffc9c9")
VIOLET = ("#6741d9", "#d0bfff")
GRAY   = ("#495057", "#e9ecef")
TEAL   = ("#0c8599", "#99e9f2")

# ───────── title ─────────
label(60, 30, "Video Tagging & Search  —  High-Level Design", size=30, font=1,
      color="#1e1e1e")
label(60, 72, "scale target: 1,000,000+ hours of video per month  =  ~1,389x realtime, sustained",
      size=16, font=3, color="#868e96")

# ───────── client ─────────
client = box(60, 250, 150, 90, "Client", "edge / upload", *BLUE, size=18)

# ───────── main pipeline row ─────────
ingest  = box(280, 250, 190, 90, "INGEST", ":8080", *BLUE, size=18)
decoder = box(600, 250, 190, 90, "DECODER", "CPU · ffmpeg", *GREEN, size=18)
tagger  = box(920, 250, 190, 90, "TAGGER", "scoring", *ORANGE, size=18)
search  = box(1240, 250, 190, 90, "SEARCH", ":8081", *BLUE, size=18)

arrow(client, ingest)
arrow(ingest, decoder, color="#c2255c")
arrow(decoder, tagger, color="#c2255c")
arrow(tagger, search, color="#c2255c")

label(214, 358, "POST\n→ 202", size=12)
label(486, 196, "video\n.ingested", size=12, color="#c2255c")
label(798, 196, "video.frames\n.extracted", size=12, color="#c2255c")
label(1124, 196, "video\n.tagged", size=12, color="#c2255c")

# ───────── supporting services row ─────────
blob = box(280, 520, 200, 100, "BLOB STORE", "local FS → S3\nlifecycle tiering", *GRAY, size=16)
arrow(ingest, blob, color="#495057")

sidecar = box(890, 520, 250, 100, "CLIP SIDECAR", ":8501 · GPU\npixels→vec · text→vec", *RED, size=16)
arrow(tagger, sidecar, color="#e03131", start_arrow="arrow")

taxo = box(1200, 520, 200, 100, "TAXONOMY", ":8082\nversioned tag DAG", *VIOLET, size=16)
arrow(taxo, tagger, color="#6741d9", dashed=True)
arrow(taxo, search, color="#6741d9", dashed=True)
label(1200, 634, "snapshot + cache;\npoll /version; re-embed,\nno restart", size=12, color="#5f3dc4")

index = box(1460, 720, 240, 100, "INDEX", "tag index (GIN)\n+ vectors (ANN)", *TEAL, size=16)
arrow(search, index, color="#0c8599")

# ───────── deepstream overlay ─────────
box(572, 212, 592, 440, "", "", "#f08c00", "transparent", dashed=True)
label(576, 176, "DeepStream SDK would replace this whole block", size=15, font=1,
      color="#e8590c")
label(576, 668,
      "NVDEC hardware decode → nvstreammux batching →\n"
      "TensorRT inference, all of it GPU-resident.\n"
      "\n"
      "Kills the frames-to-disk round trip: ~1.8B JPEG\n"
      "writes + reads per month, gone.\n"
      "\n"
      "Publishes to Kafka via nvmsgbroker, so the SAME\n"
      "video.tagged contract holds — ingest, search and\n"
      "taxonomy never learn anything changed.\n"
      "\n"
      "Cost: NVIDIA-only; decode+inference re-couple.",
      size=13, color="#e8590c")

# ───────── legend ─────────
label(60, 420, "─── solid pink  =  Kafka topic (named on the arrow)\n"
                "─── dashed violet =  cached snapshot, polled\n"
                "─── solid grey/teal =  storage read/write",
      size=12, color="#868e96")

# ───────── scale callout ─────────
box(60, 860, 640, 200, "", "", "#868e96", "#f8f9fa")
label(84, 880, "THE NUMBER THAT DRIVES EVERYTHING", size=14, font=3, color="#495057")
label(84, 910,
      "1,000,000 hr/mo ÷ 30 ÷ 24  =  1,389 hr of video per wall-clock hour\n"
      "every frame @30fps         =  41,667 inferences/sec   X  impossible\n"
      "1 frame every 2 seconds    =     694 inferences/sec   ✓  ~1 GPU\n"
      "storage 450 MB/hr          =  450 TB/mo → 5.4 PB/yr  ← the real cost",
      size=14, color="#212529")

box(740, 860, 760, 200, "", "", "#2f9e44", "#ebfbee")
label(764, 880, "THE INSIGHT", size=14, font=3, color="#2b8a3e")
label(764, 910,
      "DECODE ONCE  ·  EMBED ONCE  ·  TAG MANY TIMES\n\n"
      "A tag is a sentence, not a trained output head. Frame vectors are\n"
      "3.5 MB/hr against 450 MB/hr of video (~128x smaller) — so re-tagging\n"
      "the entire back catalogue is a dot-product pass, never a re-decode.",
      size=14, color="#212529")

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": els,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}

out = "/Users/deipayandash/instawork-interview/docs/architecture.excalidraw"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
print("wrote %s  (%d elements)" % (out, len(els)))
