// Package tagging turns sampled frames into typed, scored tags.
//
// The split here is the heart of the design. An Embedder is a dumb, stateless
// function — pixels in, vectors out; text in, vectors out — and knows nothing
// about tags. Everything that constitutes domain knowledge (which tags exist,
// how they relate, how they are phrased) lives in the taxonomy service, and the
// scoring policy lives here.
//
// That separation is what delivers "decode once, embed once, tag many times":
// changing the vocabulary never touches the model, and re-tagging history is a
// pass over stored vectors rather than a re-ingest of petabytes of video.
package tagging

import (
	"context"
	"math"
	"sort"
	"sync"

	"github.com/instawork/video-tagging/internal/model"
	"github.com/instawork/video-tagging/internal/taxonomy"
)

// Embedder maps images and text into one shared vector space. In production the
// implementation behind this interface is a GPU pool; here it is a local
// sidecar. Either way the Go services are unaware of which.
type Embedder interface {
	EmbedImages(ctx context.Context, paths []string) ([][]float32, error)
	EmbedText(ctx context.Context, texts []string) ([][]float32, error)
	Name() string
	Healthy(ctx context.Context) bool
}

// Result is a tagging pass over one video's frames.
type Result struct {
	Tags            []model.Tag
	Embeddings      []model.Embedding
	TaxonomyVersion int
}

// ZeroShotTagger scores frames against the taxonomy.
//
// It is "zero-shot" because no classifier was ever trained on these tags: a tag
// is a sentence embedded into the same space as the frames. Adding a node costs
// one text embedding — not a training run, not a re-ingest. That is also why a
// *hierarchy* is viable here at all: with a softmax head, adding a level means
// retraining; here the tree is pure metadata over the same vectors.
type ZeroShotTagger struct {
	emb Embedder

	mu       sync.RWMutex
	graph    *taxonomy.Graph
	termVecs map[string][]float32 // term name -> averaged, renormalised prompt vector
	subtrees map[string][]string  // term name -> itself plus all descendants
}

// NewZeroShotTagger embeds the taxonomy once and caches it. The vocabulary is
// small and changes rarely, so this is a few hundred milliseconds paid once
// rather than per video.
func NewZeroShotTagger(ctx context.Context, emb Embedder, g *taxonomy.Graph) (*ZeroShotTagger, error) {
	t := &ZeroShotTagger{emb: emb}
	if err := t.Reload(ctx, g); err != nil {
		return nil, err
	}
	return t, nil
}

// Reload re-embeds the vocabulary for a new taxonomy version. The taxonomy
// service can therefore be improved while the pipeline is running: consumers
// pick up the new version on their next poll and re-embed in place, with no
// restart and no redeploy.
func (t *ZeroShotTagger) Reload(ctx context.Context, g *taxonomy.Graph) error {
	var texts []string
	var owner []string
	for _, term := range g.Terms {
		for _, p := range term.Prompts {
			texts = append(texts, p)
			owner = append(owner, term.Name)
		}
	}

	vecs, err := t.emb.EmbedText(ctx, texts)
	if err != nil {
		return err
	}

	// Group the prompt vectors by term.
	byTerm := map[string][][]float32{}
	for i, v := range vecs {
		if i >= len(owner) {
			break
		}
		byTerm[owner[i]] = append(byTerm[owner[i]], v)
	}

	// Each node gets a vector from its OWN prompts only. Its subtree membership
	// is cached separately, because routing scores a node by the best match
	// anywhere beneath it rather than by this vector alone (see softmaxOver).
	//
	// That split matters. Interior nodes carry abstract phrasings ("someone
	// working with food") that CLIP scores weakly and unpredictably, while
	// leaves carry concrete ones ("hands sorting objects on an industrial
	// table"). Routing on a parent's own prompt made a correct branch lose at
	// the root and never be explored — one bad decision discarding the decisive
	// evidence two levels below it.
	//
	// Averaging the subtree into the parent was tried first and is worse:
	// `material_handling` as a centroid of {sorting, packing, lifting} is
	// dragged down by the two leaves that do not match. Max over the subtree
	// keeps the branch alive on the strength of its best descendant.
	reps := make(map[string][]float32, len(g.Terms))
	for _, term := range g.Terms {
		var sum []float32
		for _, v := range byTerm[term.Name] {
			if sum == nil {
				sum = make([]float32, len(v))
			}
			for j, f := range v {
				sum[j] += f
			}
		}
		if sum == nil {
			continue
		}
		normalise(sum)
		reps[term.Name] = sum
	}

	subtrees := make(map[string][]string, len(g.Terms))
	for _, term := range g.Terms {
		subtrees[term.Name] = g.Subtree(term.Name)
	}

	t.mu.Lock()
	t.graph, t.termVecs, t.subtrees = g, reps, subtrees
	t.mu.Unlock()
	return nil
}

func (t *ZeroShotTagger) Name() string { return t.emb.Name() }

func (t *ZeroShotTagger) Version() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.graph.Version
}

// Tag embeds every frame once, then walks the taxonomy for each frame.
//
// Note that embedding cost is independent of vocabulary size: growing the
// taxonomy from 50 to 5,000 nodes costs nothing at inference time, and routing
// is negligible beside it (see the note below).
func (t *ZeroShotTagger) Tag(ctx context.Context, frames []model.Frame) (*Result, error) {
	paths := make([]string, len(frames))
	for i, f := range frames {
		paths[i] = f.Path
	}
	vecs, err := t.emb.EmbedImages(ctx, paths)
	if err != nil {
		return nil, err
	}

	embs := make([]model.Embedding, 0, len(vecs))
	for i, v := range vecs {
		if i >= len(frames) {
			break
		}
		embs = append(embs, model.Embedding{
			VideoID: frames[i].VideoID, FrameIdx: frames[i].Index,
			TSSec: frames[i].TSSec, Vec: v,
		})
	}

	t.mu.RLock()
	g, termVecs, subtrees := t.graph, t.termVecs, t.subtrees
	t.mu.RUnlock()

	// best[term] is the term's strongest path-probability over all sampled
	// frames. Max rather than mean because an activity occupying only part of a
	// clip is still true of the clip.
	best := map[string]float64{}
	for _, fv := range vecs {
		// Score every term against this frame once, then reuse those numbers for
		// every routing decision below.
		//
		// This is O(|V|) dot products rather than the O(depth x branching) a pure
		// cascade would use, and that is a deliberate trade. At 5,000 nodes it is
		// ~2.5 MFLOP per frame against ~8.7 GFLOP for the CLIP forward pass that
		// produced the frame vector — about 0.03%. Routing cost is simply not
		// where the money goes, so we spend it to get a sharper routing signal.
		sims := make(map[string]float64, len(termVecs))
		for name, v := range termVecs {
			sims[name] = dot(fv, v)
		}
		for _, kind := range g.Kinds() {
			t.descend(g, subtrees, sims, g.Roots(kind), 1.0, 0, best)
		}
	}

	return &Result{Tags: t.selectTags(g, best), Embeddings: embs, TaxonomyVersion: g.Version}, nil
}

// selectTags turns the raw walk output into the tags actually asserted.
//
// The beam explores generously — that is what stops one bad routing decision
// from discarding a correct branch — but everything it touched above the floor
// is not worth asserting. Two rules narrow it, both applied per facet so a
// confident environment cannot crowd out the activity tags:
//
//   - a relative floor, since a term scoring 0.2 beside a sibling at 0.95 is
//     noise, while 0.2 as the best available signal may be the real answer;
//   - a hard cap, because a video realistically occupies one or two places and
//     involves a handful of activities.
func (t *ZeroShotTagger) selectTags(g *taxonomy.Graph, best map[string]float64) []model.Tag {
	const (
		relativeFloor = 0.35 // fraction of the best score within the same facet
		perFacetCap   = 6
	)

	byFacet := map[taxonomy.Kind][]model.Tag{}
	facetBest := map[taxonomy.Kind]float64{}
	for name, score := range best {
		term, ok := g.Get(name)
		if !ok {
			continue
		}
		if score > facetBest[term.Kind] {
			facetBest[term.Kind] = score
		}
		byFacet[term.Kind] = append(byFacet[term.Kind], model.Tag{
			Name:       name,
			Kind:       model.TagKind(term.Kind),
			Confidence: round3(score),
			Depth:      g.Depth(name),
			Path:       g.Path(name),
			Asserted:   true, // scored directly; the store adds rolled-up ancestors
		})
	}

	var tags []model.Tag
	for kind, cands := range byFacet { //nolint:gocritic // grouped per facet on purpose
		// For an exclusive facet, keep only the winning top-level branch.
		//
		// The winner is chosen by its best DESCENDANT, not by the root's own
		// score, and that distinction decides the outcome. On construction
		// footage `indoor` (0.92) narrowly outscores `outdoor` (0.89) at the
		// root, so picking by root would keep the indoor branch and discard
		// `construction_site` (0.88) — the correct answer. Comparing best
		// descendants picks `outdoor` instead, because 0.88 beats anything
		// under indoor.
		if taxonomy.ExclusiveKinds[kind] {
			cands = keepBestBranch(g, cands)
			facetBest[kind] = 0
			for _, c := range cands {
				if c.Confidence > facetBest[kind] {
					facetBest[kind] = c.Confidence
				}
			}
		}

		floor := facetBest[kind] * relativeFloor
		kept := cands[:0]
		for _, c := range cands {
			if c.Confidence >= floor {
				kept = append(kept, c)
			}
		}
		sort.Slice(kept, func(a, b int) bool {
			if kept[a].Confidence != kept[b].Confidence {
				return kept[a].Confidence > kept[b].Confidence
			}
			return kept[a].Name < kept[b].Name
		})
		if len(kept) > perFacetCap {
			kept = kept[:perFacetCap]
		}
		tags = append(tags, kept...)
	}

	// A row is terminal only if nothing deeper survived selection. Where the
	// walk stopped because a blurry frame could not separate siblings, the
	// parent IS the assertion — that is abstention at the right granularity
	// rather than a low-confidence guess at a leaf.
	in := make(map[string]bool, len(tags))
	for _, t := range tags {
		in[t.Name] = true
	}
	for i := range tags {
		deepest := tags[i].Depth
		terminal := true
		for _, c := range g.Subtree(tags[i].Name) {
			if c == tags[i].Name || !in[c] {
				continue
			}
			terminal = false
			if d := g.Depth(c); d > deepest {
				deepest = d
			}
		}
		tags[i].Asserted = terminal
		tags[i].EvidenceDepth = deepest
	}

	sort.Slice(tags, func(a, b int) bool {
		if tags[a].Confidence != tags[b].Confidence {
			return tags[a].Confidence > tags[b].Confidence
		}
		return tags[a].Name < tags[b].Name
	})
	return tags
}

// descend is the cascaded scoring walk: softmax over siblings, then recurse
// into the strongest few.
//
// Three properties fall out, and the third is the important one:
//
//   - A softmax over 4 siblings is far more decisive than one over 50 flat
//     terms, where probability mass smears across unrelated competitors.
//
//   - Cost is O(depth × branching) rather than O(|V|). Irrelevant at 50 terms,
//     decisive at 5,000 — which is where a real workforce taxonomy lands.
//
//   - Descent stops where the evidence stops. If a blurry POV frame supports
//     "cleaning" but cannot separate "scrubbing" from "wiping", the walk emits
//     `cleaning` at high confidence and goes no deeper. A flat vocabulary is
//     forced either to guess a specific leaf or to say nothing. This turns the
//     single-frame limitation on fine-grained activities from an error into
//     honest abstention at the right granularity.
func (t *ZeroShotTagger) descend(
	g *taxonomy.Graph, subtrees map[string][]string, sims map[string]float64,
	siblings []string, parentProb float64, depth int, best map[string]float64,
) {
	if len(siblings) == 0 {
		return
	}
	probs := softmaxOver(sims, subtrees, siblings)

	order := make([]int, len(siblings))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })

	// A beam rather than greedy descent: committing to a single root lets one
	// wrong top-level choice poison everything beneath it. The beam is widest at
	// the root, where a mistake is most expensive and branching is smallest.
	for n, idx := range order {
		if n >= taxonomy.BeamAt(depth) {
			break
		}
		name := siblings[idx]

		// Chain rule down the path. Confidence decays with depth, which is
		// correct — finer distinctions deserve less certainty — and it collapses
		// threshold tuning into one global floor on the path product.
		pathProb := parentProb * probs[idx]
		if pathProb < taxonomy.MinPathConfidence {
			continue
		}
		if pathProb > best[name] {
			best[name] = pathProb
		}
		// Use the captured graph, never t.graph: Reload can swap it mid-walk when
		// the taxonomy service publishes a new version, and a walk must see one
		// consistent snapshot throughout.
		t.descend(g, subtrees, sims, g.Children(name), pathProb, depth+1, best)
	}
}

// softmaxOver scores a frame against a set of sibling terms.
//
// A sibling's logit is the BEST similarity anywhere in its subtree, not its own.
// Routing should ask "is anything promising down this branch?", and max answers
// that where a centroid does not: `material_handling` averaged over {sorting,
// packing, lifting} is dragged down by two irrelevant leaves, and loses to
// `food_work` on footage that is plainly sorting. Max keeps the branch alive on
// the strength of its best descendant.
//
// The 100x scale is CLIP's trained logit scale, which turns raw cosine
// similarities (clustered in a narrow 0.15-0.35 band) into a usable
// distribution.
func softmaxOver(sims map[string]float64, subtrees map[string][]string, names []string) []float64 {
	const logitScale = 100.0
	logits := make([]float64, len(names))
	maxL := math.Inf(-1)
	for i, n := range names {
		bestSim := math.Inf(-1)
		for _, m := range subtrees[n] {
			if s, ok := sims[m]; ok && s > bestSim {
				bestSim = s
			}
		}
		if math.IsInf(bestSim, -1) {
			bestSim = 0
		}
		logits[i] = bestSim * logitScale
		if logits[i] > maxL {
			maxL = logits[i]
		}
	}
	var sum float64
	for i := range logits {
		logits[i] = math.Exp(logits[i] - maxL) // shift for numerical stability
		sum += logits[i]
	}
	if sum == 0 {
		return logits
	}
	for i := range logits {
		logits[i] /= sum
	}
	return logits
}

func dot(a, b []float32) float64 {
	if len(a) != len(b) || a == nil || b == nil {
		return 0
	}
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func normalise(v []float32) {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	n := math.Sqrt(sum)
	if n == 0 {
		return
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / n)
	}
}

func round3(f float64) float64 { return math.Round(f*1000) / 1000 }

// keepBestBranch resolves a mutually-exclusive facet down to one branch.
//
// Candidates are grouped by the facet root they descend from; the group
// containing the single highest-scoring term wins and the rest are dropped.
// Using the best term anywhere in the branch — rather than the root's own score
// — is what makes this correct, because a root can win the softmax on vague
// evidence while a sibling branch holds the one confident leaf.
func keepBestBranch(g *taxonomy.Graph, cands []model.Tag) []model.Tag {
	if len(cands) < 2 {
		return cands
	}

	// Rank branches by their best DESCENDANT, excluding the roots' own scores.
	//
	// Including a root would defeat the purpose: a vague parent like `indoor` can
	// score 0.92 on "it's under a roof" while the branch holds nothing specific,
	// and would then beat `outdoor` 0.89 whose child `construction_site` is the
	// confident 0.88 answer. Branches earn their place on evidence, not on how
	// broad their label is.
	rootsOf := make(map[string][]string, len(cands))
	for _, c := range cands {
		rootsOf[c.Name] = g.RootsOf(c.Name)
	}

	bestByRoot := map[string]float64{}
	for _, c := range cands {
		if c.Depth == 0 {
			continue
		}
		for _, r := range rootsOf[c.Name] {
			if c.Confidence > bestByRoot[r] {
				bestByRoot[r] = c.Confidence
			}
		}
	}
	// No branch produced a specific match — fall back to the roots themselves so
	// a coarse answer still beats no answer.
	if len(bestByRoot) == 0 {
		for _, c := range cands {
			for _, r := range rootsOf[c.Name] {
				if c.Confidence > bestByRoot[r] {
					bestByRoot[r] = c.Confidence
				}
			}
		}
	}
	if len(bestByRoot) < 2 {
		return cands
	}

	winner, best := "", -1.0
	for r, s := range bestByRoot {
		// Ties broken by name so the result is deterministic across runs.
		if s > best || (s == best && r < winner) {
			winner, best = r, s
		}
	}

	out := cands[:0]
	for _, c := range cands {
		for _, r := range rootsOf[c.Name] {
			if r == winner {
				out = append(out, c)
				break
			}
		}
	}
	return out
}
