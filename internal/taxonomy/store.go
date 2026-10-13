package taxonomy

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Store owns the authoritative taxonomy and its version counter.
//
// Versioning is not optional once the graph can change at runtime: every
// materialised ancestor row in the search index was written under some version,
// and restructuring the graph makes those rows stale. Tagged rows record the
// version they were written under so a backfill can find and reconcile them.
type Store struct {
	mu   sync.RWMutex
	g    *Graph
	path string
}

type snapshot struct {
	Version int    `json:"version"`
	Terms   []Term `json:"terms"`
}

// OpenStore loads the taxonomy from disk, seeding it on first run.
func OpenStore(path string) (*Store, error) {
	s := &Store{path: path}

	if b, err := os.ReadFile(path); err == nil {
		var snap snapshot
		if err := json.Unmarshal(b, &snap); err != nil {
			return nil, fmt.Errorf("parse taxonomy: %w", err)
		}
		g, err := New(snap.Version, snap.Terms)
		if err != nil {
			return nil, err
		}
		s.g = g
		return s, nil
	}

	terms := make([]Term, len(Seed))
	copy(terms, Seed)
	for i := range terms {
		terms[i].Source = "curated"
	}
	g, err := New(1, terms)
	if err != nil {
		return nil, fmt.Errorf("seed taxonomy is invalid: %w", err)
	}
	s.g = g
	return s, s.persist()
}

// Snapshot returns the current immutable graph. Consumers cache this in memory
// and re-fetch only when the version changes — the taxonomy is small and read
// on every frame scored, so a per-lookup RPC would be pure waste.
func (s *Store) Snapshot() *Graph {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.g
}

// AddTerm inserts a node and bumps the version.
//
// This is the endpoint the promotion loop calls: a free-text query that recurs
// and scores well gets promoted into the canonical vocabulary, placed under the
// subtree its embedding is nearest to.
func (s *Store) AddTerm(t Term) (*Graph, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.g.Has(t.Name) {
		return nil, fmt.Errorf("term %q already exists", t.Name)
	}
	if len(t.Prompts) == 0 {
		return nil, fmt.Errorf("term %q needs at least one prompt", t.Name)
	}
	if t.Source == "" {
		t.Source = "promoted"
	}

	// Build the candidate graph first; New validates parents, facets and cycles,
	// so an invalid addition is rejected before anything is mutated.
	next := append(append([]Term(nil), s.g.Terms...), t)
	g, err := New(s.g.Version+1, next)
	if err != nil {
		return nil, err
	}
	s.g = g
	return g, s.persist()
}

// RemoveTerm deletes a leaf and bumps the version.
//
// Only leaves: removing an interior node would orphan its children, and a
// taxonomy that can silently orphan nodes is worse than one that refuses.
// Governance needs an undo for a bad automatic placement.
func (s *Store) RemoveTerm(name string) (*Graph, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.g.Has(name) {
		return nil, fmt.Errorf("unknown term %q", name)
	}
	if kids := s.g.Children(name); len(kids) > 0 {
		return nil, fmt.Errorf("term %q has %d children; remove those first", name, len(kids))
	}

	next := make([]Term, 0, len(s.g.Terms)-1)
	for _, t := range s.g.Terms {
		if t.Name != name {
			next = append(next, t)
		}
	}
	g, err := New(s.g.Version+1, next)
	if err != nil {
		return nil, err
	}
	s.g = g
	return g, s.persist()
}

func (s *Store) persist() error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(snapshot{Version: s.g.Version, Terms: s.g.Terms}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, b, 0o644)
}
