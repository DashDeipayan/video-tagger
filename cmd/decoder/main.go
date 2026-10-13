// Command decoder consumes video.ingested, samples frames with ffmpeg, and
// publishes video.frames.extracted.
//
// This is the CPU-bound stage. It is a separate service from the tagger
// precisely because their resource profiles differ: decoding wants many cheap
// cores, inference wants a GPU. Splitting them means each scales on its own
// curve, and the expensive GPU is never idle waiting on ffmpeg.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"time"

	"github.com/instawork/video-tagging/internal/bus"
	"github.com/instawork/video-tagging/internal/config"
	"github.com/instawork/video-tagging/internal/frames"
	"github.com/instawork/video-tagging/internal/model"
	"github.com/instawork/video-tagging/internal/store"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := config.Load()

	if err := os.MkdirAll(cfg.FrameDir, 0o755); err != nil {
		log.Error("frame dir", "err", err)
		os.Exit(1)
	}
	st, err := store.NewSQLite(cfg.DBPath)
	if err != nil {
		log.Error("store", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	b := bus.NewKafka(cfg.KafkaBrokers, log)
	defer b.Close()

	sampler := &frames.Sampler{
		EverySec:      cfg.SampleEvery,
		MaxFrames:     cfg.MaxFrames,
		KeyframesOnly: cfg.Keyframes,
		OutDir:        cfg.FrameDir,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	d := &decoder{st: st, b: b, sampler: sampler, log: log}

	// One consumer group across every decoder replica: Kafka then hands each
	// partition to exactly one member, so adding replicas adds throughput with
	// no coordination code of our own.
	if err := b.Subscribe(ctx, bus.TopicIngested, "decoder", d.handle); err != nil {
		log.Error("subscribe", "err", err)
		os.Exit(1)
	}
	log.Info("decoder running", "topic", bus.TopicIngested,
		"sample_every_sec", cfg.SampleEvery, "max_frames", cfg.MaxFrames, "keyframes_only", cfg.Keyframes)

	waitForSignal()
	log.Info("decoder stopping")
}

type decoder struct {
	st      store.Store
	b       bus.Bus
	sampler *frames.Sampler
	log     *slog.Logger
}

func (d *decoder) handle(ctx context.Context, key string, payload []byte) error {
	var ev model.VideoIngested
	if err := json.Unmarshal(payload, &ev); err != nil {
		// Malformed message: committing is correct, since redelivering an
		// unparseable payload would loop forever. Production routes it to a
		// dead-letter topic instead of dropping it.
		d.log.Error("undecodable message, skipping", "key", key, "err", err)
		return nil
	}

	start := time.Now()
	d.st.SetStatus(ctx, ev.VideoID, model.StatusDecoding, "")

	probe, err := frames.ProbeVideo(ctx, ev.Path)
	if err != nil {
		return d.fail(ctx, ev.VideoID, "probe: "+err.Error())
	}
	d.st.SetProbe(ctx, ev.VideoID, probe.DurationSec, probe.Width, probe.Height)

	fs, err := d.sampler.Extract(ctx, ev.VideoID, ev.Path, probe)
	if err != nil {
		return d.fail(ctx, ev.VideoID, "extract: "+err.Error())
	}

	d.log.Info("decoded",
		"id", ev.VideoID, "frames", len(fs),
		"duration_sec", probe.DurationSec,
		"source_frames", probe.NBFrames,
		"ms", time.Since(start).Milliseconds())

	return d.b.Publish(ctx, bus.TopicFrames, ev.VideoID, model.FramesExtracted{
		VideoID:     ev.VideoID,
		Frames:      fs,
		DurationSec: probe.DurationSec,
		SampledFrom: probe.NBFrames,
	})
}

// fail marks the video failed and commits the message. A file that cannot be
// decoded will not decode on retry either, so redelivering it would only block
// the partition behind a video that can never succeed.
func (d *decoder) fail(ctx context.Context, id, msg string) error {
	d.log.Error("decode failed", "id", id, "err", msg)
	return d.st.SetStatus(ctx, id, model.StatusFailed, msg)
}
