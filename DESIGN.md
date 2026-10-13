# Design: Video Tagging & Search at 1M+ hours/month

## 1. Reading the requirement

Everything in this design follows from converting the headline number into a sustained rate.

```
1,000,000 hr/month ÷ 30 days = 33,333 hr/day
                   ÷ 24 hr   =  1,389 hr of video per wall-clock hour
```

> **The system must process video at ~1,389× realtime, continuously.**

Three budgets fall out of that:

| Budget | Derivation | Result |
|---|---|---|
| **Storage** | 1 Mbps ≈ 450 MB/hr × 1e6 hr | **450 TB/month → 5.4 PB/year** |
| **Ingest bandwidth** | 450 TB ÷ 2.592e6 s | **~173 MB/s (~1.4 Gbps) average**, 5–10× at peak |
| **Inference** | see below | **~700 frames/sec** |

The sample corpus in `videos/` averages ~900 kbps, so the 1 Mbps assumption is grounded in the real data rather than picked for convenience.

### The decision that makes it tractable

```
Every frame @ 30fps:  108,000 frames/hr × 1,389 hr/hr ≈ 41,667 frames/sec   ✗
1 frame / 2 seconds:    1,800 frames/hr × 1,389 hr/hr ≈    694 frames/sec   ✓
```

694 fps is roughly **one GPU** (CLIP ViT-B/32 batches ~1,000–2,000 img/s at fp16 on an L4); call it 3–4 for peak and redundancy, about **$2.2k/month**. Compare that with **$124k/month** to store a year of accumulated video and the real cost centre becomes obvious.

Temporal sampling works because scene and environment content is near-constant over seconds. Twenty-nine of every thirty frames carry almost no marginal information.

---

## 2. Bottlenecks, ranked

| # | Bottleneck | Why it bites | Lever |
|---|---|---|---|
| 1 | **Storage cost & growth** | 5.4 PB/yr, *accumulating* — unlike compute, it never goes away | Lifecycle tiering (hot → Glacier); content-hash dedupe; keep frames + vectors, not raw video, for the long tail |
| 2 | **Video decode (CPU)** | Full decode of 1,389 hr/hr ≈ 100+ cores. Usually the *actual* bottleneck — teams budget GPUs and get strangled by ffmpeg | **Keyframe-only decode.** Measured on this corpus: 19/4463, 9/2399, 33/8023, 6/1682 keyframes = **~0.4% of frames, a ~250× reduction** |
| 3 | **GPU utilisation** (not GPU count) | Per-request inference leaves a GPU ~15% busy. An idle A100 is the waste, not an extra one | Decode and inference are **separate services**, so GPUs only ever see ready batches; dynamic batching at the sidecar |
| 4 | **Search index fanout** | Multi-tag AND over 10⁸–10⁹ rows | Inverted index (Postgres GIN on `text[]`); order AND-terms by ascending selectivity; cache hot combinations |
| 5 | **Queue & failure semantics** | Millions of jobs/month ⇒ corrupt files, 4-hour videos, poison pills | At-least-once + idempotent writes; bounded retries → DLQ; per-stage backpressure |
| 6 | **Re-tagging history** | The vocabulary *will* change. Re-decoding 5 PB is a non-starter | Persist embeddings — see below |

---

## 3. The central insight

> **Decode once, embed once, tag many times.**

Zero-shot CLIP does not classify into fixed labels. It places frames *and text* in one shared vector space, which changes the economics of the vocabulary:

- **A tag is a sentence, not a trained output head.** Adding `forklift_operation` costs one text embedding — not a labelling effort, not a retraining run, not a re-ingest.
- **Frame embeddings are ~3.5 MB/hour against ~450 MB/hour of video — about 0.8%, a ~128× reduction.** (512 float32 = 2 KB per frame, at one frame every 2 seconds.) At a million hours a month that is **3.7 TB/month of vectors against 450 TB of video**, or 44 TB against 5.4 PB over a year. Re-tagging the back catalogue is then a dot-product pass over tens of TB instead of a re-decode of petabytes — the difference between a batch job and an impossibility.
- **The same vectors give semantic search for free.** A phrase nobody anticipated can be embedded at query time and matched directly.

This is why `SaveEmbeddings` sits next to `SaveTags` in the store, and why the tagger publishes both on `video.tagged`.

---

## 4. The taxonomy: a versioned DAG, not a flat list

The brief calls the tagging mechanism the main challenge, and specifically the
dynamic-vs-closed question. The flat version of this system answered it with a
hybrid (closed index, open query). The structured version answers it better.

### Why a hierarchy composes here but not with a trained classifier

With a softmax head, adding a level means retraining. Here a node is a sentence,
so **the tree is pure metadata over the same vectors**. That asymmetry is what
makes the whole structure affordable.

### Forest, not one tree

Rooting activities under environments breaks immediately: `cooking` also happens
outdoors so the node duplicates; "all cleaning regardless of location" requires
walking every environment subtree; and `kitchen → cooking` is not an **is-a**
relation the way `cleaning → dishwashing` is. Mixing relation types into one edge
kind makes roll-up incoherent.

So: **three facets, three trees**, with cross-facet links kept as a separate
`Affinities` edge type (used for query expansion and as a prior, never for
containment). `taxonomy.New` rejects a parent edge that crosses facets, because
such an edge is nearly always a context relation mislabelled as is-a.

### DAG, not tree

`plastering` is genuinely both `construction_work` and `surface_finishing`.
Forcing a single parent duplicates the node, which fragments the index — exactly
what a closed vocabulary exists to prevent. Multiple parents are allowed; cycles
are rejected at construction.

### Cascaded scoring

One flat softmax over 50 terms becomes a softmax over siblings, descending only
into branches that clear threshold, with confidence following the chain rule:

```
roots         → cleaning 0.99   food_work 0.52   housekeeping 0.11
  ↳ cleaning  → dishwashing 0.97  scrubbing 0.38  wiping 0.04
P(dishwashing) = 0.99 × 0.97 = 0.96
```

Three consequences:

1. **Sharper distributions.** A softmax over 4 siblings is far more decisive than
   over 50 flat terms where probability mass smears across unrelated competitors.
2. **One threshold instead of many.** Confidence decays with depth — correctly, since
   finer distinctions deserve less certainty — so a single global floor on the
   path product replaces per-facet thresholds.
3. **Tagging at the depth the evidence supports.** This is the one to lead with.
   If a blurry POV frame supports `cleaning` but cannot separate `scrubbing` from
   `wiping`, the walk **stops and emits `cleaning` at high confidence**. A flat
   vocabulary must either guess a leaf or say nothing. The hierarchy converts the
   single-frame weakness identified in §6 from an error into **honest abstention
   at the right granularity**.

### Two corrections the implementation forced

Both were found by measuring, not by reasoning — and both are worth raising
unprompted, because they are where the idea meets the data.

**Interior nodes cannot be scored by their own prompts.** Abstract phrasings
("someone moving and handling materials") are a far weaker CLIP signal than
concrete leaves ("hands sorting objects on an industrial table"). Scoring parents
on their own text made `material_handling` lose at the root on 2.mp4, so `sorting`
was never explored — the exact error-cascade failure a hierarchy invites. Routing
now scores a node by the **best match anywhere in its subtree**: descend if
anything down there is promising. A centroid was tried first and is worse, because
averaging `sorting` with `packing` and `lifting` dilutes the one leaf that matches.

**The beam must be widest at the root.** Greedy descent lets one bad top-level
choice poison everything beneath it. The beam is 3 at depth 0, where branching is
smallest and a mistake is most expensive, and 2 below.

Cost note, stated honestly: subtree-max routing means scoring all |V| terms per
frame, not the O(depth × branching) a pure cascade would give. At 5,000 nodes that
is ~2.5 MFLOP against ~8.7 GFLOP for the CLIP forward pass that produced the frame
vector — about 0.03%. Routing cost is not where the money goes, so we spend it for
a sharper signal. The hierarchy's value is decisiveness, abstention and roll-up —
not fewer dot products.

### Ancestors are materialised at write time

A video asserted as `dishwashing` also gets index rows for `cleaning` and every
further ancestor, each with propagated confidence, `depth`, `path`, and an
`evidence_depth` recording how far below the supporting evidence sat.

That is ~2–3× write amplification, and it buys the thing that matters at a million
hours a month: **`?tags=cleaning` stays one indexed lookup** — no recursive CTE, no
tree walk in the hot path. For a read-dominated search system, that trade is
heavily favourable.

For the taxonomy store itself at scale, a **closure table** `(ancestor, descendant,
depth)` is the right structure — the only one that handles multi-parent DAGs
cleanly, with ancestor queries as one indexed join. Postgres `ltree` is more
ergonomic but strict-tree only, so node duplication returns. Nested sets are fast
to read and miserable to write.

### What "asserted" means under a cascade

The naive flag — "did the model score this directly?" — is useless here, because a
cascaded walk scores every ancestor on the path. The meaningful question is whether
anything **deeper** was also asserted. A row is terminal only if nothing below it
survived selection, which makes `?exact=true` mean "videos whose evidence stopped
here" and turns abstention into something queryable.

### Search semantics unlocked

| Query | Behaviour |
|---|---|
| `?tags=cleaning` | Rolls up — matches `dishwashing`, `scrubbing`, `wiping` videos |
| `?tags=cleaning&exact=true` | Only where the evidence *stopped* at `cleaning` |
| `?tags=indoor.kitchen` | Path-scoped, disambiguating a term under several branches |
| `?tags=cleaning&max_depth=1` | Roll up, but not from more than one level below |
| `?tags=cleaning,outdoor&mode=all` | Cross-subtree AND, unchanged |

**Measured on the sample corpus:** `?tags=cleaning` returns all 4 videos, every one
rolled up (via `dishwashing`, `wiping`, `scrubbing`) — **none was tagged `cleaning`
directly**. `&exact=true` returns 0, correctly. `?tags=surface_finishing` returns
4.mp4, reached through `plastering`'s *second* parent.

### The taxonomy grows its own structure

The promotion loop does not just append a tag — it **places** it. A candidate phrase
is embedded and compared against existing term vectors; its nearest neighbour
becomes its parent. The vocabulary gains structure, not just entries, from the same
embedding space the frames live in.

```
place "grout_scrubbing" → under cleaning.scrubbing   margin 0.046   confident
place "mopping"         → under housekeeping         margin 0.004   NEEDS REVIEW
```

A near-tie is flagged rather than guessed. And **restructuring is nearly free** —
re-scoring stored frame vectors, a dot-product pass, with no video decoded again.
The property that made adding a tag cheap makes reshaping the whole hierarchy cheap.

One real limitation, found by testing: embedding proximity captures what a term is
*about*, not what kind of thing it *is*. `forklift_operation`, whose prompt mentions
a warehouse, was placed under `warehouse` — an **environment**. Placement therefore
takes an optional `kind` to restrict candidates, and emits a confidence margin.
It is a suggestion engine, not an authority.

### Honest costs

- **Governance becomes a real, ongoing job.** 50 nodes fit in a code review; a
  500-node DAG needs an owner, versioning and review. Taxonomies rot, and a rotted
  one is worse than a flat list. Hence the delete endpoint: a taxonomy you cannot
  correct will decay.
- **Error cascade.** A wrong root still poisons what is beneath it. Subtree-max
  routing and the beam mitigate this; they do not eliminate it.
- **Versioning is mandatory.** Every materialised ancestor row is stale after a
  restructure. Tagged rows carry `taxonomy_version`; the backfill job is cheap but
  must exist.
- **Evaluation gets harder.** Per-tag precision/recall no longer tells the story —
  predicting `cleaning` when truth is `dishwashing` is partially correct, not wrong.
  Hierarchical metrics are needed.

### Dynamic vs. closed, restated

| Layer | Vocabulary | Endpoint |
|---|---|---|
| **Index time** | Closed, curated, hierarchical | `?tags=cleaning&mode=all` — exact, fast, governed, now with roll-up |
| **Query time** | Open, free-form | `?q=scrubbing a metal grate` — embedded on the fly, zero re-processing |
| **Promotion** | Mined, auto-placed | new terms land *in the right subtree*, backfilled over stored vectors |

## 5. Architecture

```
                    ┌──────────────┐
  client ──────────▶│  ingest      │──▶ blob store      (local FS → S3)
                    │  :8080       │──▶ metadata row    (status=PENDING)
                    └──────┬───────┘──▶ video.ingested
                           │
                    ┌──────▼────────┐
                    │  decoder      │   CPU-bound · ffmpeg frame sampling
                    │               │   1 video → N frames
                    └──────┬────────┘──▶ video.frames.extracted
                           │
                    ┌──────▼────────┐        ┌──────────────────┐
                    │  tagger       │───────▶│ CLIP sidecar     │ GPU-bound
                    │               │◀───────│ :8501 (Python)   │ pixels→vectors
                    └──────┬────────┘        └──────────────────┘ text→vectors
                           │                          ▲
                           │──▶ video.tagged          │
                           │    (tags + vectors)      │
                    ┌──────▼────────┐        ┌────────┴─────────┐
                    │  search :8081 │◀───────│ taxonomy :8082   │ the tag DAG,
                    │               │        │                  │ versioned
                    └───────────────┘        └──────────────────┘
                     consumes video.tagged,    snapshot fetched + cached
                     owns its projection,      by tagger and search;
                     serves closed + open      they poll only /version
```

### Why these boundaries

**Ingest returns 202 in milliseconds** regardless of video length. Upload capacity and processing capacity then scale independently, which is what lets the system absorb bursts instead of timing out.

**Decode and inference are separate services** because their resource profiles differ — many cheap cores versus one expensive GPU. Fused, you either starve the GPU or over-provision CPU. Split, each scales on its own curve.

**The sidecar is deliberately dumb.** It maps pixels→vectors and text→vectors and knows nothing about tags, videos, or the vocabulary. All domain logic stays in Go. That keeps it a pure stateless function — exactly what can be replicated behind a load balancer on a GPU pool, or swapped for a larger model, with no Go code changing.

**Search owns its read model.** It learns about tags only by consuming `video.tagged`; nothing upstream calls it. The projection can be rebuilt by replaying the topic, and the service scales out as read replicas.

**Partitioning is keyed by video ID**, giving per-video ordering (frames cannot be tagged before extraction) while the topic as a whole is consumed in parallel. Adding partitions and consumer instances is the horizontal scaling lever at every stage.

### Failure handling

| Failure | Response | Why |
|---|---|---|
| Malformed message | Log, commit | Redelivering an unparseable payload loops forever. Production routes to a DLQ |
| Corrupt/undecodable video | Mark `FAILED`, commit | It will not decode on retry either; redelivery blocks the partition behind a video that can never succeed |
| Inference error | Leave uncommitted, **retry** | Usually transient (sidecar restart, GPU contention). Status stays `TAGGING`, not `FAILED` — the video is still in flight |
| Sidecar unavailable | Fall back to `MockEmbedder` | Degrades demo quality rather than ending it; also useful for load-testing the pipeline without burning GPU |
| Kafka unavailable | Ingest refuses to start | The stages are separate processes, so a local in-memory bus would have no subscribers. Accepting a video with 202 and dropping it is worse than refusing it |

Every handler write is an upsert keyed by video ID, so at-least-once delivery converges to the same state — effectively-once without a transactional protocol.

---

## 6. Tagging quality

Scoring is a **per-facet softmax** rather than a global one, so "which environment is this?" and "which activity is this?" are asked as separate questions and a strong environment match cannot drown out the activity signal. The 100× logit scale is CLIP's own trained value, which spreads raw cosine similarities (clustered in a narrow 0.15–0.35 band) into a usable distribution.

**Prompt engineering matters more than expected here.** The corpus is egocentric — head/chest-mounted cameras, hands at close range, heavy motion blur, no full-body subject. CLIP was trained largely on web photos, so a bare label like `dishwashing` sits far from these frames in vector space, while *"a first-person point-of-view video frame of someone washing dishes in a sink"* sits close. Each term averages several prompt phrasings to smooth out an unlucky one.

**Thresholds differ per facet because the evidence does.** A single frame pins down an environment, so we can be strict (0.22). An activity is inherently temporal — one still cannot separate "washing" from "rinsing" — so we accept weaker per-frame evidence (0.18) and lean on agreement across sampled frames.

### Measured result on the sample corpus

Top-scoring leaf per facet, after cascaded scoring over the taxonomy:

| Video | Ground truth | Environment | Activity | Object |
|---|---|---|---|---|
| 1.mp4 | bed-making, bedroom | `bedroom` 1.00 ✅ | `bed_making` 0.99 ✅ | `bedsheet` 1.00 ✅ |
| 2.mp4 | sorting, factory line | `factory_floor` 1.00 ✅ | `sorting` 0.90 ✅ | `conveyor` 0.99 ✅ |
| 3.mp4 | dishwashing, kitchen | `kitchen` 0.98 ✅ | `dishwashing` 0.97 ✅ | `sink` 0.95 ✅ |
| 4.mp4 | plastering, construction | `construction_site` 0.88 ✅ | `plastering` 0.84 ✅ | `cement` 0.99 ✅ |

**8/8 on the two graded facets** (environment and activity), objects correct throughout.

Worth stating plainly: the flat version scored the same 8/8. The hierarchy did not
buy accuracy on four videos — it bought roll-up search, abstention at the right
depth, and a vocabulary that can grow structure. Those are the things that matter
at 5,000 nodes, and four videos cannot demonstrate them numerically.

### Mutually-exclusive facets — a bug the demo surfaced

The wide beam initially cost real precision. On 4.mp4 the environment facet read:

```
indoor                    0.92      ← vague parent wins the root
outdoor                   0.89
outdoor.construction_site 0.88      ← the correct answer
indoor.bedroom            0.73
indoor.kitchen            0.68      ← false positive
```

Searching `kitchen` returned the construction site and the factory floor. The
root cause is structural, not a tuning problem: **a softmax must sum to 1.** Both
`indoor` and `outdoor` survive the beam, and the softmax over `indoor`'s four
children then has to crown one even though none of them fits — so `kitchen` wins
by default among {bathroom, bedroom, kitchen, restaurant}.

The fix is to mark a facet **mutually exclusive** (`taxonomy.ExclusiveKinds`): a
clip is in one kind of place, so asserting both `indoor.kitchen` and
`outdoor.construction_site` for the same video is incoherent. At selection time
only the winning top-level branch survives.

The subtlety is how the winner is chosen. Ranking branches by the **root's own
score** makes things worse — `indoor` 0.92 beats `outdoor` 0.89, so
`construction_site` would be discarded. Branches must be ranked by their best
**descendant**: indoor's best child is 0.73, outdoor's is 0.88, so outdoor wins.
A vague parent should not out-rank a confident leaf just because its label is
broader. (`keepBestBranch`, `internal/tagging/tagger.go`.)

Result: environment accuracy stays 4/4 while the false positives disappear.
`?tags=kitchen` returns one video instead of three, and
`?tags=kitchen,plastering&mode=all` is correctly empty again.

Activities and objects are deliberately **not** exclusive — a video can genuinely
involve cleaning and food work at once, and contain many objects.

### Known weaknesses (worth raising before someone else does)

- **Max-over-frames aggregation inflates spurious tags.** One odd frame is enough to assert a tag: 4.mp4 picks up `bedroom` 0.73 from a single frame of draped cloth. Requiring agreement across ≥2 frames, or a trimmed mean, would fix this at the cost of recall on genuinely brief activities.
- **Visually similar environments confuse.** 2.mp4's stainless-steel sorting trays read partly as `kitchen`/`dishwashing`. Defensible given the imagery, but single-frame environment classification has limits.
- **Softmax must crown a winner.** When the true answer is absent from a sibling group, the mass still has to go somewhere. Exclusive facets fix this where branches are genuinely alternatives (above); for non-exclusive facets an explicit "none of these" calibration term is the remaining gap.
- **Activities are temporal and we score single frames.** No amount of prompt tuning separates "washing" from "rinsing" without motion. A video-native model (VideoMAE, X-CLIP) over short clips is the upgrade path.
- **Audio is unused.** All four files carry AAC, and it is genuinely informative — running water in 3.mp4, troweling in 4.mp4. Whisper/CLAP on the audio track is a cheap second modality.

---

## 7. What changes at production scale

| Prototype | Production | Trigger |
|---|---|---|
| Local filesystem blobs | S3 + lifecycle tiering to Glacier | Day one |
| SQLite | Postgres, GIN index on `tags text[]` | ~10⁵ videos |
| Brute-force vector scan (`SearchByVector`) | ANN index (HNSW/IVF) in a vector DB — O(n) → ~O(log n) | ~10⁶ embeddings |
| Single-node Kafka (KRaft) | Managed Kafka, partitions sized to consumer count | Day one |
| CLIP sidecar on localhost | Autoscaled GPU pool behind a load balancer, dynamic batching | ~10³ hr/day |
| `fps=0.5` sampling | Keyframe-only + scene-change detection | When decode CPU dominates |
| Retry-forever | Bounded retries → dead-letter topic + replay tooling | Day one |
| One vocabulary for everything | Per-tenant taxonomies; promotion loop driven by query logs | When customers need their own taxonomy |
| Taxonomy as a JSON file | Closure table `(ancestor, descendant, depth)` in Postgres | ~10³ nodes |
| Manual auto-placement calls | Mining loop over query logs, with human review of flagged near-ties | When the vocabulary outgrows manual curation |

### Scaling the stages independently

Because each stage is its own deployable consuming its own topic, the scaling story is concrete rather than aspirational:

- **Decode-bound?** Add decoder replicas and partitions. No other service changes.
- **GPU-bound?** Add sidecar replicas behind the load balancer. The tagger is unaware.
- **Search-bound?** Add search replicas; each rebuilds its projection from the topic.
- **Need a new model?** Swap the `Embedder` implementation. Nothing else is touched.
- **Need a new tag?** `POST /api/taxonomy/place`; consumers re-embed on their next version poll and a backfill runs over stored vectors. No video is read again.
- **Need to reshape the hierarchy?** Same story — re-score stored vectors. The property that made adding a tag cheap makes restructuring cheap.
