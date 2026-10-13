package model

// Events are the contracts carried on Kafka topics. They are deliberately thin:
// they reference the blob by path rather than embedding bytes, so a message
// stays a few hundred bytes regardless of whether the video is 50 MB or 5 GB.
// The large payloads travel through object storage; only pointers travel
// through the log.

// VideoIngested is published by the ingest service once a video is durably
// stored. Topic: video.ingested
type VideoIngested struct {
	VideoID     string  `json:"video_id"`
	Path        string  `json:"path"`
	Filename    string  `json:"filename"`
	SizeBytes   int64   `json:"size_bytes"`
	DurationSec float64 `json:"duration_sec"`
}

// FramesExtracted is published by the decoder after sampling.
// Topic: video.frames.extracted
type FramesExtracted struct {
	VideoID     string  `json:"video_id"`
	Frames      []Frame `json:"frames"`
	DurationSec float64 `json:"duration_sec"`
	SampledFrom int     `json:"sampled_from"` // total frames in the source, for the sampling-ratio stat
}

// VideoTagged is published by the tagger. It carries both the asserted tags and
// the raw frame embeddings: the search service needs the former for exact tag
// queries and the latter for open-vocabulary similarity queries.
// Topic: video.tagged
type VideoTagged struct {
	VideoID    string      `json:"video_id"`
	Tags       []Tag       `json:"tags"`
	Embeddings []Embedding `json:"embeddings"`
	TaggerName string      `json:"tagger_name"`
	ElapsedMS  int64       `json:"elapsed_ms"`
	// TaxonomyVersion lets the search service record which graph produced these
	// tags, and detect rows needing backfill after a restructure.
	TaxonomyVersion int `json:"taxonomy_version"`
}
