package tagging

import (
	"context"
	"hash/fnv"
	"math"
	"math/rand"
)

// MockEmbedder produces deterministic pseudo-vectors without any model.
//
// It exists so the pipeline stays demonstrable when the sidecar is unavailable:
// the same failure-tolerance argument as the in-process bus. Tags it produces
// are meaningless, but every other part of the system — ingestion, Kafka,
// decode, indexing, both search paths — exercises identically, which makes it
// genuinely useful for load-testing the pipeline without burning GPU.
type MockEmbedder struct{ dim int }

func NewMockEmbedder() *MockEmbedder { return &MockEmbedder{dim: 512} }

func (m *MockEmbedder) Name() string                 { return "mock-embedder" }
func (m *MockEmbedder) Healthy(context.Context) bool { return true }

func (m *MockEmbedder) EmbedImages(_ context.Context, paths []string) ([][]float32, error) {
	return m.embed(paths), nil
}

func (m *MockEmbedder) EmbedText(_ context.Context, texts []string) ([][]float32, error) {
	return m.embed(texts), nil
}

// embed derives a stable unit vector from each input string, so the same input
// always yields the same vector and results are reproducible across runs.
func (m *MockEmbedder) embed(keys []string) [][]float32 {
	out := make([][]float32, len(keys))
	for i, k := range keys {
		h := fnv.New64a()
		h.Write([]byte(k))
		rng := rand.New(rand.NewSource(int64(h.Sum64())))
		v := make([]float32, m.dim)
		var sum float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			sum += float64(v[j]) * float64(v[j])
		}
		n := float32(math.Sqrt(sum))
		for j := range v {
			v[j] /= n
		}
		out[i] = v
	}
	return out
}
