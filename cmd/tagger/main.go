// Command tagger consumes video.frames.extracted, embeds the frames via the
// CLIP sidecar, scores them against the canonical vocabulary, and publishes
// video.tagged.
//
// This is the GPU-bound stage, and the one that would be replicated behind a
// GPU pool in production. Note what it publishes: both the asserted tags and
// the raw frame embeddings. Keeping the vectors is what lets the vocabulary
// change later without any video being touched again.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"time"

	"github.com/instawork/video-tagging/internal/bus"
	"github.com/instawork/video-tagging/internal/config"
	"github.com/instawork/video-tagging/internal/model"
	"github.com/instawork/video-tagging/internal/store"
	"github.com/instawork/video-tagging/internal/tagging"
	"github.com/instawork/video-tagging/internal/taxonomy"
)

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

	// Prefer the real model; fall back to the mock so a missing sidecar
	// degrades the quality of the demo rather than ending it.
	emb := selectEmbedder(ctx, cfg, log)

	// Fetch the taxonomy snapshot and cache it. The graph is a few KB and is
	// consulted for every frame scored, so it is fetched whole rather than
	// queried per lookup.
	taxo := taxonomy.NewClient(cfg.TaxonomyURL, log)
	graph, err := taxo.FetchWithRetry(ctx, 30)
	if err != nil {
		log.Error("taxonomy unavailable", "url", cfg.TaxonomyURL, "err", err)
		os.Exit(1)
	}
	log.Info("taxonomy loaded", "version", graph.Version, "terms", len(graph.Terms))

	// The vocabulary is embedded once here, not per video. Its cost is
	// independent of corpus size, which is why a larger taxonomy is nearly free
	// at inference time.
	tagger, err := tagging.NewZeroShotTagger(ctx, emb, graph)
	if err != nil {
		log.Error("vocabulary embedding failed", "err", err)
		os.Exit(1)
	}

	// Pick up taxonomy changes without a restart: re-embed the new vocabulary in
	// place. This is what makes the taxonomy service genuinely improvable while
	// the pipeline is running.
	taxo.Watch(ctx, 5*time.Second, func(g *taxonomy.Graph) {
		if err := tagger.Reload(ctx, g); err != nil {
			log.Error("re-embedding taxonomy failed", "version", g.Version, "err", err)
			return
		}
		log.Info("tagger re-embedded for new taxonomy", "version", g.Version, "terms", len(g.Terms))
	})

	b := bus.NewKafka(cfg.KafkaBrokers, log)
	defer b.Close()

	t := &taggerSvc{st: st, b: b, tagger: tagger, log: log}
	if err := b.Subscribe(ctx, bus.TopicFrames, "tagger", t.handle); err != nil {
		log.Error("subscribe", "err", err)
		os.Exit(1)
	}
	log.Info("tagger running", "topic", bus.TopicFrames,
		"model", tagger.Name(), "taxonomy_version", tagger.Version())

	waitForSignal()
	log.Info("tagger stopping")
}

func selectEmbedder(ctx context.Context, cfg config.Config, log *slog.Logger) tagging.Embedder {
	clip := tagging.NewCLIPSidecar(cfg.SidecarURL)
	// The sidecar loads model weights on boot, so give it a moment before
	// concluding it is absent.
	for i := 0; i < 20; i++ {
		if clip.Healthy(ctx) {
			log.Info("using CLIP sidecar", "url", cfg.SidecarURL)
			return clip
		}
		time.Sleep(time.Second)
	}
	log.Warn("CLIP sidecar unreachable, falling back to mock embedder", "url", cfg.SidecarURL)
	return tagging.NewMockEmbedder()
}

type taggerSvc struct {
	st     store.Store
	b      bus.Bus
	tagger *tagging.ZeroShotTagger
	log    *slog.Logger
}

func (t *taggerSvc) handle(ctx context.Context, key string, payload []byte) error {
	var ev model.FramesExtracted
	if err := json.Unmarshal(payload, &ev); err != nil {
		t.log.Error("undecodable message, skipping", "key", key, "err", err)
		return nil
	}

	start := time.Now()
	t.st.SetStatus(ctx, ev.VideoID, model.StatusTagging, "")

	res, err := t.tagger.Tag(ctx, ev.Frames)
	if err != nil {
		// Returning the error leaves the message uncommitted, so Kafka will
		// redeliver. Correct here: inference failures are usually transient
		// (sidecar restarting, GPU contention) and worth retrying, unlike a
		// corrupt video file.
		// Status stays TAGGING, not FAILED: the message was not committed, so it
		// is still in flight. Marking it terminal here would lie to the UI about
		// a video that is about to be retried.
		t.log.Error("tagging failed, will retry", "id", ev.VideoID, "err", err)
		return err
	}

	elapsed := time.Since(start)
	t.log.Info("tagged", "id", ev.VideoID,
		"tags", len(res.Tags), "frames", len(res.Embeddings), "ms", elapsed.Milliseconds())

	return t.b.Publish(ctx, bus.TopicTagged, ev.VideoID, model.VideoTagged{
		VideoID:         ev.VideoID,
		Tags:            res.Tags,
		Embeddings:      res.Embeddings,
		TaggerName:      t.tagger.Name(),
		ElapsedMS:       elapsed.Milliseconds(),
		TaxonomyVersion: res.TaxonomyVersion,
	})
}
