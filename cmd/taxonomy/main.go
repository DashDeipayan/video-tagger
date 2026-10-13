// Command taxonomy owns the tag vocabulary: its structure, its version, and the
// rules for growing it.
//
// It is a separate service because the taxonomy's lifecycle has nothing to do
// with the pipeline's. It is curated by humans, versioned, reviewed, and
// gradually extended — on a cadence measured in weeks, while the pipeline runs
// continuously. Separating it means the vocabulary can be improved without
// redeploying the tagger, and gives governance a single place to live.
//
// Consumers fetch a snapshot and cache it; they poll only /version. The graph
// is a few KB and is consulted for every frame scored, so chatty reads would be
// waste.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"sort"
	"time"

	"github.com/instawork/video-tagging/internal/config"
	"github.com/instawork/video-tagging/internal/httpx"
	"github.com/instawork/video-tagging/internal/tagging"
	"github.com/instawork/video-tagging/internal/taxonomy"
)

type server struct {
	store *taxonomy.Store
	emb   tagging.Embedder
	log   *slog.Logger
}

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := config.Load()

	store, err := taxonomy.OpenStore(cfg.TaxonomyPath)
	if err != nil {
		log.Error("taxonomy", "err", err)
		os.Exit(1)
	}
	g := store.Snapshot()
	log.Info("taxonomy loaded", "version", g.Version, "terms", len(g.Terms),
		"facets", len(g.Kinds()), "path", cfg.TaxonomyPath)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s := &server{store: store, emb: selectEmbedder(ctx, cfg, log), log: log}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/taxonomy", s.handleGet)
	mux.HandleFunc("GET /api/taxonomy/version", s.handleVersion)
	mux.HandleFunc("GET /api/taxonomy/tree", s.handleTree)
	mux.HandleFunc("GET /api/taxonomy/ancestors", s.handleAncestors)
	mux.HandleFunc("POST /api/taxonomy/terms", s.handleAdd)
	mux.HandleFunc("POST /api/taxonomy/place", s.handlePlace)
	mux.HandleFunc("DELETE /api/taxonomy/terms/{name}", s.handleRemove)
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) {
		httpx.JSON(w, http.StatusOK, map[string]any{
			"service": "taxonomy", "status": "ok",
			"version": s.store.Snapshot().Version,
		})
	})

	srv := &http.Server{Addr: cfg.TaxonomyAddr, Handler: httpx.CORS(httpx.Log(log, mux))}
	go func() {
		log.Info("taxonomy listening", "addr", cfg.TaxonomyAddr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("listen", "err", err)
			os.Exit(1)
		}
	}()

	waitForSignal()
	sc, c2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer c2()
	srv.Shutdown(sc)
}

func (s *server) handleGet(w http.ResponseWriter, r *http.Request) {
	g := s.store.Snapshot()
	httpx.JSON(w, http.StatusOK, map[string]any{"version": g.Version, "terms": g.Terms})
}

func (s *server) handleVersion(w http.ResponseWriter, r *http.Request) {
	httpx.JSON(w, http.StatusOK, map[string]any{"version": s.store.Snapshot().Version})
}

// handleTree renders the DAG as nested nodes, for the UI's taxonomy browser.
func (s *server) handleTree(w http.ResponseWriter, r *http.Request) {
	g := s.store.Snapshot()

	type node struct {
		Name     string   `json:"name"`
		Kind     string   `json:"kind"`
		Depth    int      `json:"depth"`
		Path     string   `json:"path"`
		Source   string   `json:"source,omitempty"`
		Parents  []string `json:"parents,omitempty"`
		Children []*node  `json:"children,omitempty"`
	}
	var build func(name string) *node
	build = func(name string) *node {
		t, _ := g.Get(name)
		n := &node{Name: name, Kind: string(t.Kind), Depth: g.Depth(name),
			Path: g.Path(name), Source: t.Source, Parents: t.Parents}
		for _, c := range g.Children(name) {
			n.Children = append(n.Children, build(c))
		}
		return n
	}

	out := map[string][]*node{}
	for _, k := range g.Kinds() {
		for _, root := range g.Roots(k) {
			out[string(k)] = append(out[string(k)], build(root))
		}
	}
	httpx.JSON(w, http.StatusOK, map[string]any{"version": g.Version, "facets": out})
}

func (s *server) handleAncestors(w http.ResponseWriter, r *http.Request) {
	g := s.store.Snapshot()
	tag := r.URL.Query().Get("tag")
	if !g.Has(tag) {
		httpx.Error(w, http.StatusNotFound, "unknown term: "+tag)
		return
	}
	httpx.JSON(w, http.StatusOK, map[string]any{
		"term": tag, "path": g.Path(tag), "depth": g.Depth(tag),
		"ancestors": g.Ancestors(tag), "children": g.Children(tag),
	})
}

// handleAdd inserts a term. Validation (unknown parent, cross-facet parent,
// cycle, duplicate) happens in taxonomy.New before anything is mutated.
func (s *server) handleAdd(w http.ResponseWriter, r *http.Request) {
	var t taxonomy.Term
	if err := json.NewDecoder(r.Body).Decode(&t); err != nil {
		httpx.Error(w, http.StatusBadRequest, err.Error())
		return
	}
	g, err := s.store.AddTerm(t)
	if err != nil {
		httpx.Error(w, http.StatusBadRequest, err.Error())
		return
	}
	s.log.Info("term added", "name", t.Name, "parents", t.Parents, "version", g.Version)
	httpx.JSON(w, http.StatusCreated, map[string]any{
		"version": g.Version, "term": t,
		"note": "consumers will pick this up on their next version poll; " +
			"existing videos need a backfill over stored embeddings",
	})
}

// handlePlace is the auto-placement endpoint, and the mechanism behind the
// promotion loop.
//
// Given a candidate phrase, it embeds it and finds its nearest neighbours among
// the existing term vectors. Because tags and frames share one vector space,
// the taxonomy can work out where a new term *belongs* rather than having it
// appended flat. The vocabulary gains structure, not just entries.
//
// With commit=true it also inserts the term under the suggested parent.
// handleRemove undoes a placement. Governance needs this: automatic placement
// will sometimes be wrong, and a taxonomy you cannot correct rots.
func (s *server) handleRemove(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	g, err := s.store.RemoveTerm(name)
	if err != nil {
		httpx.Error(w, http.StatusBadRequest, err.Error())
		return
	}
	s.log.Info("term removed", "name", name, "version", g.Version)
	httpx.JSON(w, http.StatusOK, map[string]any{"removed": name, "version": g.Version})
}

func (s *server) handlePlace(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name    string   `json:"name"`
		Prompts []string `json:"prompts"`
		// Kind restricts candidate parents to one facet.
		//
		// Worth supplying. Embedding proximity captures topical similarity, not
		// relational type: "driving a forklift in a warehouse" sits nearest to
		// `warehouse`, an environment, so unrestricted placement would file an
		// activity under a place. The embedding knows what a term is ABOUT, not
		// what kind of thing it IS.
		Kind   string `json:"kind"`
		Commit bool   `json:"commit"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Name == "" {
		httpx.Error(w, http.StatusBadRequest, `expected {"name":"...","prompts":["..."]}`)
		return
	}
	if len(req.Prompts) == 0 {
		req.Prompts = []string{req.Name}
	}

	g := s.store.Snapshot()

	// Embed the candidate alongside every existing term, in one call, so both
	// sides are produced by the same model on the same request.
	texts := req.Prompts
	owner := make([]int, len(req.Prompts))
	for i := range owner {
		owner[i] = -1 // -1 marks the candidate
	}
	for i, t := range g.Terms {
		for _, p := range t.Prompts {
			texts = append(texts, p)
			owner = append(owner, i)
		}
	}

	vecs, err := s.emb.EmbedText(r.Context(), texts)
	if err != nil {
		httpx.Error(w, http.StatusServiceUnavailable, "embedder unavailable: "+err.Error())
		return
	}

	cand := meanVec(vecs, owner, -1)
	type scored struct {
		Name  string  `json:"name"`
		Kind  string  `json:"kind"`
		Path  string  `json:"path"`
		Score float64 `json:"similarity"`
	}
	var ranked []scored
	for i, t := range g.Terms {
		if req.Kind != "" && string(t.Kind) != req.Kind {
			continue
		}
		v := meanVec(vecs, owner, i)
		ranked = append(ranked, scored{t.Name, string(t.Kind), g.Path(t.Name), dot(cand, v)})
	}
	if len(ranked) == 0 {
		httpx.Error(w, http.StatusBadRequest, "no candidate parents for kind: "+req.Kind)
		return
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].Score > ranked[b].Score })
	if len(ranked) > 8 {
		ranked = ranked[:8]
	}

	// The nearest existing term is the suggested parent; the candidate inherits
	// its facet, which also guarantees the parent edge does not cross facets.
	best := ranked[0]
	suggested := taxonomy.Term{
		Name:    req.Name,
		Kind:    taxonomy.Kind(best.Kind),
		Parents: []string{best.Name},
		Prompts: req.Prompts,
		Source:  "promoted",
	}

	resp := map[string]any{
		"suggested":  suggested,
		"neighbours": ranked,
		"committed":  false,
	}
	// A placement much closer to its parent than to the runner-up is confident;
	// a near-tie is exactly where a human should look. Automatic placement is a
	// suggestion engine, not an authority — embedding proximity captures what a
	// term is ABOUT, not what kind of thing it IS.
	if len(ranked) > 1 {
		margin := ranked[0].Score - ranked[1].Score
		resp["margin"] = margin
		resp["needs_review"] = margin < 0.03
	}
	if req.Commit {
		ng, err := s.store.AddTerm(suggested)
		if err != nil {
			httpx.Error(w, http.StatusConflict, err.Error())
			return
		}
		resp["committed"] = true
		resp["version"] = ng.Version
		s.log.Info("term auto-placed", "name", req.Name, "under", best.Name, "version", ng.Version)
	}
	httpx.JSON(w, http.StatusOK, resp)
}

// meanVec averages the vectors belonging to one owner and renormalises.
func meanVec(vecs [][]float32, owner []int, want int) []float32 {
	var sum []float32
	n := 0
	for i, o := range owner {
		if o != want || i >= len(vecs) {
			continue
		}
		if sum == nil {
			sum = make([]float32, len(vecs[i]))
		}
		for j, f := range vecs[i] {
			sum[j] += f
		}
		n++
	}
	if n == 0 {
		return nil
	}
	var mag float64
	for _, f := range sum {
		mag += float64(f) * float64(f)
	}
	if mag > 0 {
		s := float32(1 / sqrt(mag))
		for j := range sum {
			sum[j] *= s
		}
	}
	return sum
}

func sqrt(f float64) float64 {
	if f <= 0 {
		return 0
	}
	x := f
	for i := 0; i < 24; i++ {
		x = 0.5 * (x + f/x)
	}
	return x
}

func dot(a, b []float32) float64 {
	if len(a) != len(b) || a == nil {
		return 0
	}
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func selectEmbedder(ctx context.Context, cfg config.Config, log *slog.Logger) tagging.Embedder {
	clip := tagging.NewCLIPSidecar(cfg.SidecarURL)
	for i := 0; i < 30; i++ {
		if clip.Healthy(ctx) {
			log.Info("using CLIP sidecar for term placement", "url", cfg.SidecarURL)
			return clip
		}
		time.Sleep(time.Second)
	}
	log.Warn("CLIP sidecar unreachable; auto-placement will be meaningless")
	return tagging.NewMockEmbedder()
}
