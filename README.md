# Video Tagging & Search

A prototype system that ingests video, tags it with a real zero-shot vision model, and makes it searchable by any combination of tags — designed against a requirement of **1,000,000+ hours of video per month**.

See **[DESIGN.md](DESIGN.md)** for the scale analysis, the bottleneck ranking, and the taxonomy design (hierarchical DAG, cascaded scoring, dynamic vs. closed vocabulary).

---

## Quick start

```bash
make up      # builds and starts kafka, the CLIP sidecar, and all five Go services
make demo    # registers every videos/*.mp4, waits for tagging, runs example searches
```

Then open **http://localhost:8080**.

> **Supply your own footage.** The clips this was developed against are real
> first-person video of identifiable people and are not committed. Drop any
> `.mp4` files into `videos/` before `make demo`, or drag-and-drop them into the
> UI. See [videos/README.md](videos/README.md) for what the default taxonomy is
> tuned for.

`make status` shows what is running; `make down` stops the services; `make clean` also wipes the catalog.

### Prerequisites

| | |
|---|---|
| Go | 1.23+ |
| ffmpeg | 8+ (uses `-fps_mode`, which replaced `-vsync`) |
| Kafka | 3.9 in KRaft mode — `brew install kafka`. No ZooKeeper, no Docker |
| Python | 3.12 for the CLIP sidecar — `make sidecar` builds the venv on first run |

The sidecar downloads CLIP weights (~600 MB) on first start and runs on Apple Silicon's GPU (MPS), CUDA, or CPU automatically.

---

## Architecture

Five Go services and one Python sidecar, communicating over Kafka. Each is independently deployable and scalable.

```
ingest :8080 ──▶ video.ingested ──▶ decoder ──▶ video.frames.extracted ──▶ tagger ──▶ video.tagged ──▶ search :8081
                                    (CPU)                                   (GPU) │                     │
                                                                   CLIP sidecar :8501                   │
                                                                                  ▲                     │
                                                            taxonomy :8082 ───────┴─────────────────────┘
                                                            (versioned tag DAG, fetched as a snapshot)
```

| Component | Role | Scales on |
|---|---|---|
| `cmd/ingest` | Accepts uploads/registrations, persists the blob, publishes. Serves the UI | Request volume |
| `cmd/decoder` | ffmpeg frame sampling | CPU cores |
| `cmd/tagger` | Embeds frames, walks the taxonomy, scores | GPU |
| `cmd/search` | Consumes `video.tagged`, owns its index projection, serves queries | Read volume |
| `cmd/taxonomy` | Owns the tag DAG, its version, and how it grows | Rarely — it is read-mostly and tiny |
| `sidecar/` | Stateless CLIP embedder: pixels→vectors, text→vectors | GPU |

The taxonomy is consumed as a **cached versioned snapshot**, not a per-lookup RPC: the graph is a few KB and is read for every frame scored. Consumers poll `/version` and re-embed in place when it changes — no restart.

### Layout

```
cmd/            five service entry points
internal/
  bus/          Kafka transport + in-process fallback, behind one interface
  store/        Store interface + SQLite implementation (tag index + vectors)
  tagging/      Embedder interface, cascaded zero-shot tagger, CLIP client, mock
  taxonomy/     the tag DAG: graph, seed, versioned store, client, ancestor expansion
  frames/       ffmpeg sampling and probing
  model/        domain types and Kafka event contracts
  config/       environment-driven settings
sidecar/        Python CLIP inference service
scripts/        demo.py (scripted walkthrough), tree.py (DAG renderer)
docs/           architecture.excalidraw — editable HLD diagram
web/            single-page demo UI
videos/         sample corpus (not committed — supply your own)
```

---

## API

### Ingest — `:8080`

```bash
# upload bytes
curl -F file=@clip.mp4 localhost:8080/api/videos

# register a file already on disk (the path that matters at scale)
curl -X POST localhost:8080/api/videos/register \
     -H 'Content-Type: application/json' \
     -d '{"path":"videos/3.mp4"}'

curl localhost:8080/api/videos/{id}     # status + tags
curl localhost:8080/api/stats           # corpus summary
```

Ingest returns **202 Accepted** immediately — processing is asynchronous. Poll `/api/videos/{id}` to watch `PENDING → DECODING → TAGGING → TAGGED`.

### Search — `:8081`

**Closed vocabulary** — exact tags against the inverted index, with hierarchy:

```bash
curl 'localhost:8081/api/search?tags=kitchen,dishwashing&mode=all'        # AND
curl 'localhost:8081/api/search?tags=bedroom,construction_site&mode=any'  # OR

# Roll-up: matches videos tagged with any DESCENDANT of `cleaning`
curl 'localhost:8081/api/search?tags=cleaning'

# Exact: only videos whose evidence STOPPED at `cleaning`
curl 'localhost:8081/api/search?tags=cleaning&exact=true'

# Path-scoped, and depth-limited roll-up
curl 'localhost:8081/api/search?tags=indoor.kitchen'
curl 'localhost:8081/api/search?tags=cleaning&max_depth=1'
```

**Open vocabulary** — any phrase, embedded at query time, matched against stored frame vectors:

```bash
curl --get localhost:8081/api/search \
     --data-urlencode 'q=scrubbing a metal grate under running water'
```

Results carry `match_ts`, the timestamp of the best-matching frame, so the UI deep-links to the moment that caused the hit.

| Param | Meaning |
|---|---|
| `tags` | comma-separated canonical tags (closed vocabulary) |
| `q` | free-text phrase (open vocabulary) |
| `mode` | `all` (AND, default) or `any` (OR) |
| `exact` | `true` restricts to terminal matches, disabling roll-up |
| `max_depth` | how many levels below the matched term its evidence may sit |
| `min_confidence` | floor on tag confidence |
| `min_score` | relevance floor for `q` queries (default 0.26; pass 0 to disable) |
| `limit` | max results (default 50) |

Semantic results are filtered by a relevance floor so a query that matches
nothing returns nothing. See [DESIGN.md](DESIGN.md) for why that number is a
presentation choice rather than a calibrated threshold.

### Taxonomy — `:8082`

```bash
curl localhost:8082/api/taxonomy/tree | python3 scripts/tree.py   # or: make taxonomy
curl 'localhost:8082/api/taxonomy/ancestors?tag=dishwashing'

# Ask where a new term belongs — embedding finds its nearest subtree
curl -X POST localhost:8082/api/taxonomy/place \
     -H 'Content-Type: application/json' \
     -d '{"name":"grout_scrubbing","kind":"activity",
          "prompts":["a first-person view of someone scrubbing tile grout"]}'

# Add it for real (bumps the version; consumers re-embed on their next poll)
#   ...same body with "commit": true
curl -X DELETE localhost:8082/api/taxonomy/terms/grout_scrubbing
```

Pass `kind` to restrict candidate parents to one facet. Without it, placement can file an activity under a place — embedding proximity captures what a term is *about*, not what kind of thing it *is*. The response carries a `margin` and `needs_review` flag for near-ties.

---

## Results on the sample corpus

Four egocentric work videos, 9.1 minutes total. Top-scoring leaf per facet — **8/8** on environment and activity:

| Video | Ground truth | Environment | Activity | Object |
|---|---|---|---|---|
| 1.mp4 | bed-making, bedroom | `bedroom` 1.00 | `bed_making` 0.99 | `bedsheet` 1.00 |
| 2.mp4 | sorting, factory line | `factory_floor` 1.00 | `sorting` 0.90 | `conveyor` 0.99 |
| 3.mp4 | dishwashing, kitchen | `kitchen` 0.98 | `dishwashing` 0.97 | `sink` 0.95 |
| 4.mp4 | plastering, construction | `construction_site` 0.88 | `plastering` 0.84 | `cement` 0.99 |

**Hierarchical roll-up** — `?tags=cleaning` returns all four videos, every one matched through a *descendant*; none was tagged `cleaning` directly:

| Video | via | `exact=true` |
|---|---|---|
| 3.mp4 | `dishwashing` 0.99 | — |
| 2.mp4 | `dishwashing` 0.62 | — |
| 1.mp4 | `wiping` 0.59 | — |
| 4.mp4 | `scrubbing` 0.32 | — |

`?tags=cleaning&exact=true` correctly returns **0** — no video's evidence stopped at `cleaning`. `?tags=surface_finishing` returns 4.mp4, reached through `plastering`'s *second* parent in the DAG.

**Open vocabulary** — none of these phrases exist in the taxonomy:

| Query | Top hit |
|---|---|
| "scrubbing a metal grate under running water" | 3.mp4 @22s |
| "someone spreading wet concrete overhead" | 4.mp4 @24s |
| "folding fabric onto a mattress" | 1.mp4 @56s |
| "workers inspecting items on an assembly line" | 2.mp4 @46s |

---

## Configuration

All services read the same environment variables (`internal/config`):

| Variable | Default | Notes |
|---|---|---|
| `KAFKA_BROKERS` | `localhost:9092` | empty or `none` selects the in-process bus |
| `DB_PATH` | `./data/catalog.db` | |
| `BLOB_DIR` | `./data/blobs` | resolved to absolute — paths cross service boundaries |
| `FRAME_DIR` | `./data/frames` | likewise |
| `SIDECAR_URL` | `http://127.0.0.1:8501` | falls back to the mock embedder if unreachable |
| `TAXONOMY_URL` | `http://127.0.0.1:8082` | fetched once and cached; `/version` is polled |
| `TAXONOMY_PATH` | `./data/taxonomy.json` | delete to reset to the curated seed |
| `SAMPLE_EVERY_SEC` | `2.0` | frame sampling interval |
| `MAX_FRAMES` | `24` | per-video cap, bounds work for long videos |
| `KEYFRAMES_ONLY` | `false` | `true` decodes I-frames only (~250× cheaper on this corpus) |

---

## Design notes

Three decisions carry most of the weight. [DESIGN.md](DESIGN.md) argues them in full.

**Sampling, not every frame.** At 1,389× realtime, tagging every frame would need ~41,667 inferences/second. One frame every two seconds makes it ~694/second — about one GPU.

**Decode once, embed once, tag many times.** Frame embeddings run ~3.5 MB/hour against ~450 MB/hour of video — about 0.8%, a ~128× reduction. Persisting them means a new tag costs one text embedding plus a dot-product backfill, never a re-decode. It is also what makes open-vocabulary search possible without re-processing anything.

**The sidecar knows nothing about tags.** It maps pixels and text to vectors; the vocabulary, prompts, thresholds, and scoring all live in Go. That keeps inference a pure stateless function — replicable behind a load balancer, swappable for a larger model, with no Go code changing.

**The vocabulary is a versioned DAG, not a list.** Three facets as separate trees (`kitchen → cooking` is a context relation, not is-a); multiple parents allowed, since `plastering` is genuinely both construction work and surface finishing. Scoring cascades over siblings with a chain-rule confidence, so the walk can **stop at the depth the evidence supports** — emitting `cleaning` confidently rather than guessing between `scrubbing` and `wiping` on a blurry frame. Ancestors are materialised at write time, which keeps a roll-up query a single indexed lookup.
