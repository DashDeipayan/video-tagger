package taxonomy

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// Client is the consumer side of the taxonomy service.
//
// It fetches the whole graph once and caches it, then polls only the cheap
// /version endpoint. The taxonomy is a few KB and is consulted for every frame
// scored, so a per-lookup RPC would add a network hop to the hot path for no
// benefit. Snapshot-and-cache is how configuration services are meant to be
// consumed.
type Client struct {
	baseURL string
	hc      *http.Client
	log     *slog.Logger

	mu       sync.RWMutex
	cached   *Graph
	onChange func(*Graph) // invoked when a new version is observed
}

func NewClient(baseURL string, log *slog.Logger) *Client {
	return &Client{
		baseURL: baseURL,
		hc:      &http.Client{Timeout: 10 * time.Second},
		log:     log,
	}
}

// Graph returns the cached snapshot.
func (c *Client) Graph() *Graph {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cached
}

// Fetch pulls the full taxonomy and replaces the cache.
func (c *Client) Fetch(ctx context.Context) (*Graph, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/taxonomy", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("taxonomy fetch: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("taxonomy fetch: %s", resp.Status)
	}

	var snap snapshot
	if err := json.NewDecoder(resp.Body).Decode(&snap); err != nil {
		return nil, err
	}
	g, err := New(snap.Version, snap.Terms)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	c.cached = g
	c.mu.Unlock()
	return g, nil
}

// Watch polls for a version change and re-fetches when one appears, invoking
// onChange so the tagger can re-embed the vocabulary without a restart. This is
// what makes the taxonomy improvable while the system is running.
func (c *Client) Watch(ctx context.Context, every time.Duration, onChange func(*Graph)) {
	go func() {
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				v, err := c.remoteVersion(ctx)
				if err != nil {
					continue
				}
				if cur := c.Graph(); cur != nil && cur.Version == v {
					continue
				}
				g, err := c.Fetch(ctx)
				if err != nil {
					c.log.Error("taxonomy refetch failed", "err", err)
					continue
				}
				c.log.Info("taxonomy updated", "version", g.Version, "terms", len(g.Terms))
				if onChange != nil {
					onChange(g)
				}
			}
		}
	}()
}

func (c *Client) remoteVersion(ctx context.Context) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/taxonomy/version", nil)
	if err != nil {
		return 0, err
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	var out struct {
		Version int `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, err
	}
	return out.Version, nil
}

// FetchWithRetry blocks until the taxonomy service answers, so a consumer
// starting alongside it in `make up` does not race its boot.
func (c *Client) FetchWithRetry(ctx context.Context, attempts int) (*Graph, error) {
	var err error
	for i := 0; i < attempts; i++ {
		var g *Graph
		if g, err = c.Fetch(ctx); err == nil {
			return g, nil
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Second):
		}
	}
	return nil, err
}
