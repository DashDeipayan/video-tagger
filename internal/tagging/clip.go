package tagging

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// CLIPSidecar talks to the Python inference service over HTTP.
//
// Go is the right language for ingestion, orchestration and search, but it has
// no mature native tensor runtime, so inference lives in a separate process.
// That is not a workaround — it is how you would deploy this anyway, because
// inference scales on GPUs while the API tier scales on cheap CPU. The HTTP
// boundary here is the same boundary that becomes a load-balanced GPU pool in
// production.
type CLIPSidecar struct {
	baseURL string
	hc      *http.Client
	// batch is how many images are sent per request. Batching is what keeps a
	// GPU busy; sending one image at a time typically leaves it ~15% utilised.
	batch int
}

func NewCLIPSidecar(baseURL string) *CLIPSidecar {
	return &CLIPSidecar{
		baseURL: baseURL,
		hc:      &http.Client{Timeout: 120 * time.Second},
		batch:   16,
	}
}

func (c *CLIPSidecar) Name() string { return "clip-vit-base-patch32" }

func (c *CLIPSidecar) Healthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (c *CLIPSidecar) EmbedImages(ctx context.Context, paths []string) ([][]float32, error) {
	out := make([][]float32, 0, len(paths))
	for i := 0; i < len(paths); i += c.batch {
		end := min(i+c.batch, len(paths))
		vecs, err := c.post(ctx, "/embed/images", map[string]any{"paths": paths[i:end]})
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (c *CLIPSidecar) EmbedText(ctx context.Context, texts []string) ([][]float32, error) {
	return c.post(ctx, "/embed/text", map[string]any{"texts": texts})
}

func (c *CLIPSidecar) post(ctx context.Context, path string, body any) ([][]float32, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sidecar %s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var msg bytes.Buffer
		msg.ReadFrom(resp.Body)
		return nil, fmt.Errorf("sidecar %s: %s: %s", path, resp.Status, msg.String())
	}
	var out struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Embeddings, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
