// Package model holds the domain types shared by every service in the monorepo.
//
// These types are the contract between services: they are serialised as JSON
// onto Kafka topics, so changes here ripple across the whole pipeline.
package model

import "time"

// Status tracks a video through the pipeline. Each transition is owned by
// exactly one service, which keeps the state machine easy to reason about:
//
//	PENDING   -> written by ingest
//	DECODING  -> written by decoder
//	TAGGING   -> written by tagger
//	TAGGED    -> written by tagger (terminal, success)
//	FAILED    -> written by whichever stage gave up (terminal, failure)
type Status string

const (
	StatusPending  Status = "PENDING"
	StatusDecoding Status = "DECODING"
	StatusTagging  Status = "TAGGING"
	StatusTagged   Status = "TAGGED"
	StatusFailed   Status = "FAILED"
)

// TagKind is the facet a tag belongs to. Typing tags (rather than keeping one
// flat bag of strings) lets search filter per facet and lets us apply a
// different confidence threshold to each: a single frame identifies an
// environment reliably, but an activity needs more evidence.
type TagKind string

const (
	KindEnvironment TagKind = "environment"
	KindActivity    TagKind = "activity"
	KindObject      TagKind = "object"
)

// Tag is a scored, typed label attached to a video.
//
// Depth and Path come from the taxonomy and are denormalised onto every row so
// that hierarchical queries (roll-up, path-scoping, depth limits) stay flat
// indexed lookups instead of recursive tree walks in the hot path.
type Tag struct {
	Name       string  `json:"name"`       // canonical slug, e.g. "dishwashing"
	Kind       TagKind `json:"kind"`       // facet
	Confidence float64 `json:"confidence"` // 0..1, chain-rule product along the path
	Depth      int     `json:"depth"`      // 0 = facet root
	Path       string  `json:"path"`       // e.g. "cleaning.dishwashing"
	// Asserted marks the deepest point the evidence actually supported. A
	// cascaded walk scores interior nodes too, so "was it scored?" would be true
	// of every ancestor and tell you nothing. What matters is whether anything
	// DEEPER was also asserted: if so this row is a roll-up, and the real
	// evidence lives below it. `?exact=true` filters to the terminal rows.
	Asserted bool `json:"asserted"`
	// EvidenceDepth is the depth of the deepest descendant supporting this row
	// (equal to Depth for a terminal assertion). It lets `max_depth` express
	// "roll up, but not from more than N levels below", which the row's own
	// depth cannot.
	EvidenceDepth int `json:"evidence_depth"`
}

// Video is the central aggregate.
type Video struct {
	ID          string  `json:"id"`
	Filename    string  `json:"filename"`
	Path        string  `json:"path"` // blob location (local FS here, S3 in production)
	SizeBytes   int64   `json:"size_bytes"`
	DurationSec float64 `json:"duration_sec"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
	Status      Status  `json:"status"`
	Error       string  `json:"error,omitempty"`
	Tags        []Tag   `json:"tags"`
	FrameCount  int     `json:"frame_count"`
	TaggerName  string  `json:"tagger_name,omitempty"` // which backend produced the tags
	// TaxonomyVersion records the graph these tags were written under.
	// Restructuring the taxonomy makes materialised ancestor rows stale, so a
	// backfill job uses this to find rows that need re-expanding.
	TaxonomyVersion int       `json:"taxonomy_version"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// Frame is a single still sampled from a video, written to disk by the decoder
// and read back by the tagger.
type Frame struct {
	VideoID string  `json:"video_id"`
	Index   int     `json:"index"`
	TSSec   float64 `json:"ts_sec"` // presentation timestamp within the video
	Path    string  `json:"path"`
}

// Embedding is a frame's position in CLIP's joint image/text vector space.
//
// Persisting these is the single most important storage decision in the system.
// Video runs ~450 MB/hour; its embeddings ~3.5 MB/hour (512 float32 = 2 KB per
// frame, one frame every 2 seconds) — about 0.8%, a ~128x reduction. Keeping
// them means the tag vocabulary can grow forever without ever decoding a video
// twice, and it is what makes open-vocabulary search possible at query time.
type Embedding struct {
	VideoID  string    `json:"video_id"`
	FrameIdx int       `json:"frame_idx"`
	TSSec    float64   `json:"ts_sec"`
	Vec      []float32 `json:"vec"` // L2-normalised, so cosine similarity is a dot product
}

// ScoredVideo is a search hit carrying its relevance score.
type ScoredVideo struct {
	Video `json:"video"`
	Score float64 `json:"score"`
	// MatchTS is the timestamp of the best-matching frame, so the UI can deep-link
	// into the moment that caused the hit rather than the start of the video.
	MatchTS float64 `json:"match_ts"`
}
