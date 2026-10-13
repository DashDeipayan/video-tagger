package store

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/instawork/video-tagging/internal/model"
	_ "modernc.org/sqlite" // pure-Go driver: no cgo, so the binary stays portable
)

// SQLite implements Store. Chosen for the prototype because it needs no server
// while still being a real relational engine with real indexes, so the search
// queries below are the same shape you would run against Postgres.
type SQLite struct{ db *sql.DB }

const schema = `
CREATE TABLE IF NOT EXISTS videos (
  id           TEXT PRIMARY KEY,
  filename     TEXT NOT NULL,
  path         TEXT NOT NULL,
  size_bytes   INTEGER NOT NULL DEFAULT 0,
  duration_sec REAL    NOT NULL DEFAULT 0,
  width        INTEGER NOT NULL DEFAULT 0,
  height       INTEGER NOT NULL DEFAULT 0,
  status       TEXT    NOT NULL,
  error        TEXT    NOT NULL DEFAULT '',
  frame_count  INTEGER NOT NULL DEFAULT 0,
  tagger_name  TEXT    NOT NULL DEFAULT '',
  taxonomy_version INTEGER NOT NULL DEFAULT 0,
  created_at   TEXT    NOT NULL,
  updated_at   TEXT    NOT NULL
);

-- The inverted index backing closed-vocabulary search. In Postgres this would
-- be a GIN index on a text[] column; the row-per-tag shape here is the portable
-- equivalent and keeps multi-tag AND a single indexed query.
--
-- Ancestors are materialised here at WRITE time: a video tagged 'dishwashing'
-- also gets rows for 'cleaning' (its parent) and any further ancestors, each
-- carrying the propagated confidence. That costs roughly 2-3x write
-- amplification and buys the thing that actually matters at a million hours a
-- month: a roll-up query like ?tags=cleaning stays ONE indexed lookup, with no
-- recursive CTE and no tree walk in the read path.
CREATE TABLE IF NOT EXISTS video_tags (
  video_id   TEXT NOT NULL,
  name       TEXT NOT NULL,
  kind       TEXT NOT NULL,
  confidence REAL NOT NULL,
  depth      INTEGER NOT NULL DEFAULT 0,  -- 0 = facet root
  path       TEXT    NOT NULL DEFAULT '', -- e.g. 'cleaning.dishwashing'
  asserted   INTEGER NOT NULL DEFAULT 1,  -- 1 = terminal, 0 = ancestor of deeper evidence
  evidence_depth INTEGER NOT NULL DEFAULT 0, -- depth of the deepest supporting descendant
  PRIMARY KEY (video_id, name)
);
CREATE INDEX IF NOT EXISTS idx_tags_name  ON video_tags(name, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_tags_depth ON video_tags(name, depth);

-- Frame embeddings backing open-vocabulary search. Small next to the video
-- itself (~3.5 MB per hour against ~450 MB, roughly 0.8%), which is what makes
-- re-tagging the entire back catalogue a cheap batch job rather than a re-ingest.
CREATE TABLE IF NOT EXISTS embeddings (
  video_id  TEXT NOT NULL,
  frame_idx INTEGER NOT NULL,
  ts_sec    REAL NOT NULL,
  vec       BLOB NOT NULL,
  PRIMARY KEY (video_id, frame_idx)
);
`

func NewSQLite(path string) (*SQLite, error) {
	db, err := sql.Open("sqlite", path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)")
	if err != nil {
		return nil, err
	}
	// Several services share this file; WAL plus a busy timeout keeps concurrent
	// readers from tripping over the tagger's writes.
	if _, err := db.Exec(schema); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return &SQLite{db: db}, nil
}

func (s *SQLite) Close() error { return s.db.Close() }

func (s *SQLite) CreateVideo(ctx context.Context, v *model.Video) error {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO videos (id, filename, path, size_bytes, duration_sec, width, height, status, created_at, updated_at)
		VALUES (?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO NOTHING`,
		v.ID, v.Filename, v.Path, v.SizeBytes, v.DurationSec, v.Width, v.Height, string(v.Status), now, now)
	return err
}

func (s *SQLite) SetStatus(ctx context.Context, id string, st model.Status, errMsg string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE videos SET status=?, error=?, updated_at=? WHERE id=?`,
		string(st), errMsg, time.Now().UTC().Format(time.RFC3339Nano), id)
	return err
}

func (s *SQLite) SetProbe(ctx context.Context, id string, dur float64, w, h int) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE videos SET duration_sec=?, width=?, height=?, updated_at=? WHERE id=?`,
		dur, w, h, time.Now().UTC().Format(time.RFC3339Nano), id)
	return err
}

// SaveTags replaces a video's tags atomically. Delete-then-insert makes the
// write idempotent, so a Kafka redelivery produces the same end state rather
// than duplicate rows.
//
// Callers pass tags already expanded to include ancestors (see
// taxonomy.ExpandAncestors). Expansion happens on the write path so that reads
// never have to walk the graph.
func (s *SQLite) SaveTags(ctx context.Context, videoID string, tags []model.Tag, taggerName string, frameCount, taxonomyVersion int) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(ctx, `DELETE FROM video_tags WHERE video_id=?`, videoID); err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO video_tags (video_id, name, kind, confidence, depth, path, asserted, evidence_depth)
		 VALUES (?,?,?,?,?,?,?,?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, t := range tags {
		asserted := 0
		if t.Asserted {
			asserted = 1
		}
		if _, err := stmt.ExecContext(ctx, videoID, t.Name, string(t.Kind),
			t.Confidence, t.Depth, t.Path, asserted, t.EvidenceDepth); err != nil {
			return err
		}
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE videos SET status=?, tagger_name=?, frame_count=?, taxonomy_version=?, error='', updated_at=? WHERE id=?`,
		string(model.StatusTagged), taggerName, frameCount, taxonomyVersion,
		time.Now().UTC().Format(time.RFC3339Nano), videoID); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *SQLite) SaveEmbeddings(ctx context.Context, videoID string, embs []model.Embedding) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(ctx, `DELETE FROM embeddings WHERE video_id=?`, videoID); err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO embeddings (video_id, frame_idx, ts_sec, vec) VALUES (?,?,?,?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, e := range embs {
		if _, err := stmt.ExecContext(ctx, videoID, e.FrameIdx, e.TSSec, encodeVec(e.Vec)); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *SQLite) GetVideo(ctx context.Context, id string) (*model.Video, error) {
	row := s.db.QueryRowContext(ctx, selectVideo+` WHERE id=?`, id)
	v, err := scanVideo(row)
	if err != nil {
		return nil, err
	}
	if v.Tags, err = s.tagsFor(ctx, id); err != nil {
		return nil, err
	}
	return v, nil
}

func (s *SQLite) ListVideos(ctx context.Context, limit int) ([]*model.Video, error) {
	rows, err := s.db.QueryContext(ctx, selectVideo+` ORDER BY created_at DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []*model.Video
	for rows.Next() {
		v, err := scanVideo(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return s.attachTags(ctx, out)
}

// SearchByTags is the closed-vocabulary path.
//
// The AND case is the interesting one: group the index rows by video and keep
// only those matching every requested tag. This is one indexed pass, not a scan
// — the property that keeps multi-tag search fast as the corpus grows.
func (s *SQLite) SearchByTags(ctx context.Context, q Query) ([]*model.ScoredVideo, error) {
	if len(q.Tags) == 0 {
		return nil, nil
	}
	ph := strings.TrimSuffix(strings.Repeat("?,", len(q.Tags)), ",")
	args := make([]any, 0, len(q.Tags)+6)
	for _, t := range q.Tags {
		args = append(args, t)
	}
	args = append(args, q.MinConf)
	for _, k := range q.Kinds {
		args = append(args, string(k))
	}

	// Roll-up is the default and costs nothing extra, because the ancestor rows
	// were written at index time. These clauses only narrow it.
	filters := ""
	if q.Exact {
		filters += " AND asserted = 1"
	}
	if len(q.Kinds) > 0 {
		kph := strings.TrimSuffix(strings.Repeat("?,", len(q.Kinds)), ",")
		filters += " AND kind IN (" + kph + ")"
	}
	if q.MaxDepth >= 0 && q.HasMaxDepth {
		// Distance from the matched row down to the evidence that produced it,
		// which is what "roll up, but not from too far below" actually means.
		filters += " AND (evidence_depth - depth) <= ?"
	}

	having := ""
	if q.Mode == ModeAll {
		having = fmt.Sprintf("HAVING COUNT(DISTINCT name) = %d", len(q.Tags))
	}
	sqlStr := fmt.Sprintf(`
		SELECT video_id, AVG(confidence) AS score
		FROM video_tags
		WHERE name IN (%s) AND confidence >= ?%s
		GROUP BY video_id %s
		ORDER BY score DESC
		LIMIT ?`, ph, filters, having)
	if q.MaxDepth >= 0 && q.HasMaxDepth {
		args = append(args, q.MaxDepth)
	}
	args = append(args, q.Limit)

	rows, err := s.db.QueryContext(ctx, sqlStr, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type hit struct {
		id    string
		score float64
	}
	var hits []hit
	for rows.Next() {
		var h hit
		if err := rows.Scan(&h.id, &h.score); err != nil {
			return nil, err
		}
		hits = append(hits, h)
	}
	out := make([]*model.ScoredVideo, 0, len(hits))
	for _, h := range hits {
		v, err := s.GetVideo(ctx, h.id)
		if err != nil {
			continue
		}
		out = append(out, &model.ScoredVideo{Video: *v, Score: h.score})
	}
	return out, nil
}

// SearchByVector is the open-vocabulary path: an arbitrary phrase is embedded at
// query time and compared against every stored frame vector.
//
// A video scores as its single best-matching frame, and we keep that frame's
// timestamp so the UI can deep-link to the moment rather than the start.
//
// The brute-force scan is honest for a prototype and wrong at scale: production
// replaces it with an ANN index (HNSW/IVF) in a vector database, turning this
// O(n) pass into roughly O(log n).
func (s *SQLite) SearchByVector(ctx context.Context, vec []float32, limit int) ([]*model.ScoredVideo, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT video_id, ts_sec, vec FROM embeddings`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type best struct {
		score float64
		ts    float64
	}
	bests := map[string]best{}
	for rows.Next() {
		var id string
		var ts float64
		var blob []byte
		if err := rows.Scan(&id, &ts, &blob); err != nil {
			return nil, err
		}
		// Both sides are L2-normalised, so the dot product is the cosine similarity.
		sc := dot(vec, decodeVec(blob))
		if b, ok := bests[id]; !ok || sc > b.score {
			bests[id] = best{sc, ts}
		}
	}

	type kv struct {
		id string
		b  best
	}
	ranked := make([]kv, 0, len(bests))
	for id, b := range bests {
		ranked = append(ranked, kv{id, b})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].b.score > ranked[j].b.score })
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}

	out := make([]*model.ScoredVideo, 0, len(ranked))
	for _, r := range ranked {
		v, err := s.GetVideo(ctx, r.id)
		if err != nil {
			continue
		}
		out = append(out, &model.ScoredVideo{Video: *v, Score: r.b.score, MatchTS: r.b.ts})
	}
	return out, nil
}

func (s *SQLite) TagCloud(ctx context.Context) ([]TagCount, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT name, kind, MIN(depth), MIN(path),
		       COUNT(*)                      AS total,
		       COALESCE(SUM(asserted), 0)    AS direct
		FROM video_tags
		GROUP BY name, kind ORDER BY total DESC, name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []TagCount
	for rows.Next() {
		var t TagCount
		if err := rows.Scan(&t.Name, &t.Kind, &t.Depth, &t.Path, &t.Count, &t.Direct); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}

func (s *SQLite) Stats(ctx context.Context) (Stats, error) {
	var st Stats
	err := s.db.QueryRowContext(ctx, `
		SELECT COUNT(*),
		       COALESCE(SUM(status='TAGGED'),0),
		       COALESCE(SUM(status IN ('PENDING','DECODING','TAGGING')),0),
		       COALESCE(SUM(status='FAILED'),0),
		       COALESCE(SUM(duration_sec),0)
		FROM videos`).
		Scan(&st.Videos, &st.Tagged, &st.Pending, &st.Failed, &st.TotalDuration)
	if err != nil {
		return st, err
	}
	s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM embeddings`).Scan(&st.Embeddings)
	s.db.QueryRowContext(ctx, `SELECT COUNT(DISTINCT name) FROM video_tags`).Scan(&st.DistinctTags)
	return st, nil
}

// ---- helpers ----

const selectVideo = `SELECT id, filename, path, size_bytes, duration_sec, width, height,
	status, error, frame_count, tagger_name, taxonomy_version, created_at, updated_at FROM videos`

type scanner interface{ Scan(dest ...any) error }

func scanVideo(sc scanner) (*model.Video, error) {
	var v model.Video
	var status, created, updated string
	if err := sc.Scan(&v.ID, &v.Filename, &v.Path, &v.SizeBytes, &v.DurationSec, &v.Width, &v.Height,
		&status, &v.Error, &v.FrameCount, &v.TaggerName, &v.TaxonomyVersion, &created, &updated); err != nil {
		return nil, err
	}
	v.Status = model.Status(status)
	v.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
	v.UpdatedAt, _ = time.Parse(time.RFC3339Nano, updated)
	v.Tags = []model.Tag{}
	return &v, nil
}

func (s *SQLite) tagsFor(ctx context.Context, videoID string) ([]model.Tag, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT name, kind, confidence, depth, path, asserted, evidence_depth
		 FROM video_tags WHERE video_id=? ORDER BY confidence DESC`, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	tags := []model.Tag{}
	for rows.Next() {
		var t model.Tag
		var asserted int
		if err := rows.Scan(&t.Name, &t.Kind, &t.Confidence, &t.Depth, &t.Path, &asserted, &t.EvidenceDepth); err != nil {
			return nil, err
		}
		t.Asserted = asserted == 1
		tags = append(tags, t)
	}
	return tags, nil
}

func (s *SQLite) attachTags(ctx context.Context, vs []*model.Video) ([]*model.Video, error) {
	for _, v := range vs {
		t, err := s.tagsFor(ctx, v.ID)
		if err != nil {
			return nil, err
		}
		v.Tags = t
	}
	return vs, nil
}

// Vectors are stored as packed little-endian float32 — compact and decode-free
// on read compared with JSON.
func encodeVec(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func decodeVec(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

func dot(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}
