// Package config centralises environment-driven settings.
//
// Every service reads the same struct, so a deployment knob behaves identically
// everywhere and there is one place to look for what is tunable.
package config

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Config struct {
	KafkaBrokers []string // empty disables Kafka and selects the in-process bus
	DBPath       string
	BlobDir      string // where uploaded videos land (S3 in production)
	FrameDir     string // where sampled frames land (ephemeral scratch)
	SidecarURL   string // CLIP inference service
	TaxonomyURL  string // taxonomy service, consumed as a cached snapshot
	TaxonomyPath string // where the taxonomy service persists the graph
	IngestAddr   string
	SearchAddr   string
	TaxonomyAddr string
	WebDir       string
	Workers      int // concurrent in-flight messages per consumer service
	SampleEvery  float64
	MaxFrames    int
	Keyframes    bool
}

func Load() Config {
	c := Config{
		KafkaBrokers: splitCSV(env("KAFKA_BROKERS", "localhost:9092")),
		DBPath:       env("DB_PATH", "./data/catalog.db"),
		BlobDir:      env("BLOB_DIR", "./data/blobs"),
		FrameDir:     env("FRAME_DIR", "./data/frames"),
		SidecarURL:   env("SIDECAR_URL", "http://127.0.0.1:8501"),
		TaxonomyURL:  env("TAXONOMY_URL", "http://127.0.0.1:8082"),
		TaxonomyPath: env("TAXONOMY_PATH", "./data/taxonomy.json"),
		IngestAddr:   env("INGEST_ADDR", ":8080"),
		SearchAddr:   env("SEARCH_ADDR", ":8081"),
		TaxonomyAddr: env("TAXONOMY_ADDR", ":8082"),
		WebDir:       env("WEB_DIR", "./web"),
		Workers:      envInt("WORKERS", 4),
		SampleEvery:  envFloat("SAMPLE_EVERY_SEC", 2.0),
		MaxFrames:    envInt("MAX_FRAMES", 24),
		Keyframes:    env("KEYFRAMES_ONLY", "false") == "true",
	}

	// Blob and frame paths travel between services inside Kafka messages, and
	// those services do not share a working directory (the Python sidecar runs
	// from ./sidecar). Relative paths are therefore meaningless on the far side,
	// so they are resolved to absolute here, once.
	//
	// In production this is the same reason these fields hold S3 URIs rather
	// than filesystem paths: a location is only useful if it means the same
	// thing in every process that reads it.
	c.BlobDir = mustAbs(c.BlobDir)
	c.FrameDir = mustAbs(c.FrameDir)
	return c
}

func mustAbs(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return abs
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return def
}

func envFloat(k string, def float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(k), 64); err == nil {
		return v
	}
	return def
}

func splitCSV(s string) []string {
	if s == "" || s == "none" {
		return nil
	}
	return strings.Split(s, ",")
}
