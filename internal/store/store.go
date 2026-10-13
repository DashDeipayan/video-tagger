// Package store is the persistence seam.
//
// The prototype uses SQLite so the whole system runs with `go run` and no
// external services. The interface is what matters: in production this becomes
// Postgres (a GIN index on a text[] column is the real answer for multi-tag AND)
// plus a dedicated vector database for embeddings. No service above this layer
// would change.
package store

import (
	"context"

	"github.com/instawork/video-tagging/internal/model"
)

// SearchMode selects the boolean semantics of a multi-tag query.
type SearchMode string

const (
	ModeAll SearchMode = "all" // AND — video must carry every requested tag
	ModeAny SearchMode = "any" // OR  — video carries at least one
)

// Query is a closed-vocabulary search over the tag index.
//
// Because ancestors are materialised at write time, hierarchical semantics come
// almost for free here: a query for `cleaning` matches a video tagged
// `dishwashing` because the roll-up row already exists. Exact and MaxDepth
// narrow that default behaviour.
type Query struct {
	Tags  []string
	Kinds []model.TagKind // optional facet filter
	Mode  SearchMode
	// Exact restricts matching to tags the model asserted directly, excluding
	// rows that exist only because an ancestor rolled up.
	Exact bool
	// MaxDepth caps how many levels BELOW the matched term its supporting
	// evidence may sit. 0 means "terminal matches only" — equivalent to Exact.
	// HasMaxDepth distinguishes an unset filter from an explicit 0.
	MaxDepth    int
	HasMaxDepth bool
	MinConf     float64
	Limit       int
}

// Store persists video metadata, the tag index, and frame embeddings.
type Store interface {
	// Ingest-side writes.
	CreateVideo(ctx context.Context, v *model.Video) error
	SetStatus(ctx context.Context, id string, s model.Status, errMsg string) error
	SetProbe(ctx context.Context, id string, dur float64, w, h int) error

	// Tagger-side writes. Both are upserts keyed by video ID so redelivery is safe.
	SaveTags(ctx context.Context, videoID string, tags []model.Tag, taggerName string, frameCount, taxonomyVersion int) error
	SaveEmbeddings(ctx context.Context, videoID string, embs []model.Embedding) error

	// Read side.
	GetVideo(ctx context.Context, id string) (*model.Video, error)
	ListVideos(ctx context.Context, limit int) ([]*model.Video, error)
	SearchByTags(ctx context.Context, q Query) ([]*model.ScoredVideo, error)
	// SearchByVector ranks videos by similarity to vec. minScore drops hits
	// below a relevance floor; 0 disables it and returns everything ranked.
	SearchByVector(ctx context.Context, vec []float32, limit int, minScore float64) ([]*model.ScoredVideo, error)
	TagCloud(ctx context.Context) ([]TagCount, error)
	Stats(ctx context.Context) (Stats, error)

	Close() error
}

// TagCount powers the UI's tag picker and shows vocabulary coverage at a glance.
//
// Count includes rolled-up matches; Direct counts only videos the model
// asserted the tag on. The gap between them is exactly what the hierarchy buys:
// `cleaning` with Direct=0 and Count=2 means two videos are reachable under it
// that were never tagged `cleaning` themselves.
type TagCount struct {
	Name   string        `json:"name"`
	Kind   model.TagKind `json:"kind"`
	Depth  int           `json:"depth"`
	Path   string        `json:"path"`
	Count  int           `json:"count"`
	Direct int           `json:"direct"`
}

// Stats is the corpus summary rendered on the dashboard.
type Stats struct {
	Videos        int     `json:"videos"`
	Tagged        int     `json:"tagged"`
	Pending       int     `json:"pending"`
	Failed        int     `json:"failed"`
	TotalDuration float64 `json:"total_duration_sec"`
	Embeddings    int     `json:"embeddings"`
	DistinctTags  int     `json:"distinct_tags"`
}
