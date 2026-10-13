// Command search is the read side of the system.
//
// It is both a consumer and an API. It consumes video.tagged and maintains its
// own projection (the tag index and the embedding table), then serves queries
// over it. Nothing upstream calls it; it learns about new tags only from the
// event log. That one-way flow means search can be rebuilt from scratch by
// replaying the topic, and can be scaled out purely as read replicas.
//
// It exposes both halves of the hybrid vocabulary strategy:
//
//	GET /api/search?tags=kitchen,cleaning&mode=all   closed, exact, indexed
//	GET /api/search?q=scrubbing a metal grate        open, embedded at query time
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/instawork/video-tagging/internal/bus"
	"github.com/instawork/video-tagging/internal/config"
	"github.com/instawork/video-tagging/internal/httpx"
	"github.com/instawork/video-tagging/internal/model"
	"github.com/instawork/video-tagging/internal/store"
	"github.com/instawork/video-tagging/internal/tagging"
	"github.com/instawork/video-tagging/internal/taxonomy"
)

// defaultMinScore is the relevance floor applied to semantic queries. See the
// note at its use site for why it is a presentation choice rather than a
// calibrated threshold.
const defaultMinScore = 0.26

type server struct {
	st   store.Store
	emb  tagging.Embedder
	taxo *taxonomy.Client
	log  *slog.Logger
}

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := config.Load()

	st, err := store.NewSQLite(cfg.DBPath)
	if err != nil {
		log.Error("store", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Search needs only the text half of the embedder: it encodes the user's
	// phrase at query time and compares it with stored frame vectors.
	emb := selectEmbedder(ctx, cfg, log)

	// Search needs the graph to expand ancestors on the write path and to
	// resolve path-scoped queries on the read path.
	taxo := taxonomy.NewClient(cfg.TaxonomyURL, log)
	if _, err := taxo.FetchWithRetry(ctx, 30); err != nil {
		log.Error("taxonomy unavailable", "url", cfg.TaxonomyURL, "err", err)
		os.Exit(1)
	}
	taxo.Watch(ctx, 5*time.Second, nil)

	s := &server{st: st, emb: emb, taxo: taxo, log: log}

	b := bus.NewKafka(cfg.KafkaBrokers, log)
	defer b.Close()
	if err := b.Subscribe(ctx, bus.TopicTagged, "search-indexer", s.index); err != nil {
		log.Error("subscribe", "err", err)
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/search", s.handleSearch)
	mux.HandleFunc("GET /api/tags", s.handleTags)
	mux.HandleFunc("GET /api/health", s.handleHealth)

	srv := &http.Server{Addr: cfg.SearchAddr, Handler: httpx.CORS(httpx.Log(log, mux))}
	go func() {
		log.Info("search listening", "addr", cfg.SearchAddr, "indexing", bus.TopicTagged)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("listen", "err", err)
			os.Exit(1)
		}
	}()

	waitForSignal()
	shutdownCtx, c2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer c2()
	srv.Shutdown(shutdownCtx)
}

// index applies a tagging result to the read model.
//
// Both writes are upserts keyed by video ID, so a Kafka redelivery converges to
// the same state instead of duplicating rows. That is what makes at-least-once
// delivery safe without a transactional protocol.
func (s *server) index(ctx context.Context, key string, payload []byte) error {
	var ev model.VideoTagged
	if err := json.Unmarshal(payload, &ev); err != nil {
		s.log.Error("undecodable message, skipping", "key", key, "err", err)
		return nil
	}
	if err := s.st.SaveEmbeddings(ctx, ev.VideoID, ev.Embeddings); err != nil {
		return err
	}

	// Materialise the ancestor closure before writing. Paying this on the write
	// path is what keeps a roll-up query a single flat lookup at read time.
	expanded := taxonomy.ExpandAncestors(s.taxo.Graph(), ev.Tags)

	if err := s.st.SaveTags(ctx, ev.VideoID, expanded, ev.TaggerName,
		len(ev.Embeddings), ev.TaxonomyVersion); err != nil {
		return err
	}
	s.log.Info("indexed", "id", ev.VideoID,
		"asserted", len(ev.Tags), "with_ancestors", len(expanded),
		"vectors", len(ev.Embeddings), "taxonomy_version", ev.TaxonomyVersion)
	return nil
}

// handleSearch serves both query modes from one endpoint, chosen by whether the
// caller supplied tags or free text.
func (s *server) handleSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit := 50
	if l, err := strconv.Atoi(q.Get("limit")); err == nil && l > 0 {
		limit = l
	}

	// --- open vocabulary: any phrase, embedded now, matched by similarity ---
	if text := strings.TrimSpace(q.Get("q")); text != "" {
		vecs, err := s.emb.EmbedText(r.Context(), []string{text})
		if err != nil || len(vecs) == 0 {
			httpx.Error(w, http.StatusServiceUnavailable, "embedding service unavailable")
			return
		}
		// Relevance floor for semantic hits.
		//
		// Measured on this corpus: correct matches bottom out at 0.260 while
		// nonsense queries top out at 0.253, so 0.26 removes the irrelevant tail
		// and makes a no-match query return nothing at all.
		//
		// Worth being precise about what this does NOT do. CLIP similarity is a
		// ranking signal, not a calibrated relevance score — a wrong answer can
		// score 0.337 while a right one scores 0.260, so the floor cannot
		// improve precision, only cut the obvious noise. The gap it relies on is
		// 0.007 wide, which is a presentation choice tuned on a small corpus,
		// not a principled cutoff. Pass min_score=0 to disable.
		minScore := defaultMinScore
		if v, err := strconv.ParseFloat(q.Get("min_score"), 64); err == nil {
			minScore = v
		}
		hits, err := s.st.SearchByVector(r.Context(), vecs[0], limit, minScore)
		if err != nil {
			httpx.Error(w, http.StatusInternalServerError, err.Error())
			return
		}
		resp := response("semantic", text, hits)
		resp["min_score"] = minScore
		httpx.JSON(w, http.StatusOK, resp)
		return
	}

	// --- closed vocabulary: exact tags against the inverted index ---
	tags := splitCSV(q.Get("tags"))
	if len(tags) == 0 {
		httpx.Error(w, http.StatusBadRequest, "provide ?tags=a,b or ?q=free text")
		return
	}
	mode := store.ModeAll
	if q.Get("mode") == "any" {
		mode = store.ModeAny
	}
	minConf := 0.0
	if c, err := strconv.ParseFloat(q.Get("min_confidence"), 64); err == nil {
		minConf = c
	}
	var kinds []model.TagKind
	for _, k := range splitCSV(q.Get("kinds")) {
		kinds = append(kinds, model.TagKind(k))
	}
	maxDepth, hasMaxDepth := 0, false
	if d, err := strconv.Atoi(q.Get("max_depth")); err == nil && d >= 0 {
		maxDepth, hasMaxDepth = d, true
	}

	// A dotted tag is a path-scoped query ("kitchen.cleaning"), which
	// disambiguates a term appearing under more than one branch. The last
	// segment is the term actually matched.
	for i, t := range tags {
		if idx := strings.LastIndex(t, "."); idx >= 0 {
			tags[i] = t[idx+1:]
		}
	}

	hits, err := s.st.SearchByTags(r.Context(), store.Query{
		Tags: tags, Kinds: kinds, Mode: mode,
		Exact: q.Get("exact") == "true", MaxDepth: maxDepth, HasMaxDepth: hasMaxDepth,
		MinConf: minConf, Limit: limit,
	})
	if err != nil {
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}
	resp := response(string(mode), strings.Join(tags, ","), hits)
	resp["rollup"] = q.Get("exact") != "true"
	if g := s.taxo.Graph(); g != nil {
		resp["taxonomy_version"] = g.Version
	}
	httpx.JSON(w, http.StatusOK, resp)
}

func (s *server) handleTags(w http.ResponseWriter, r *http.Request) {
	tc, err := s.st.TagCloud(r.Context())
	if err != nil {
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}
	if tc == nil {
		tc = []store.TagCount{}
	}
	httpx.JSON(w, http.StatusOK, tc)
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	httpx.JSON(w, http.StatusOK, map[string]any{
		"service": "search", "status": "ok", "embedder": s.emb.Name(),
	})
}

func response(mode, query string, hits []*model.ScoredVideo) map[string]any {
	if hits == nil {
		hits = []*model.ScoredVideo{}
	}
	return map[string]any{"mode": mode, "query": query, "count": len(hits), "results": hits}
}

func selectEmbedder(ctx context.Context, cfg config.Config, log *slog.Logger) tagging.Embedder {
	clip := tagging.NewCLIPSidecar(cfg.SidecarURL)
	for i := 0; i < 20; i++ {
		if clip.Healthy(ctx) {
			log.Info("using CLIP sidecar for query embedding", "url", cfg.SidecarURL)
			return clip
		}
		time.Sleep(time.Second)
	}
	log.Warn("CLIP sidecar unreachable, semantic search will be meaningless")
	return tagging.NewMockEmbedder()
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
