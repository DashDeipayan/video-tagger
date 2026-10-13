package taxonomy

import "github.com/instawork/video-tagging/internal/model"

// ExpandAncestors materialises the full ancestor closure of a tag set.
//
// This runs on the WRITE path, deliberately. A video the model asserted as
// `dishwashing` also gets rows for `cleaning` and any further ancestors, so a
// roll-up query stays a single flat indexed lookup rather than a recursive tree
// walk per request. We pay ~2-3x write amplification once, and keep reads O(1)
// forever — the right trade when reads vastly outnumber writes and the pitch is
// about search latency.
//
// A rolled-up ancestor inherits the strongest confidence among its descendants:
// if the best evidence for `cleaning` is a `dishwashing` scored 0.50, then
// `cleaning` is supported at least that well. Ancestors scored directly by the
// cascaded walk keep their own (higher) score, since that is stronger evidence
// than anything inherited.
//
// Asserted=false marks the inherited rows, so `?exact=true` can exclude them.
func ExpandAncestors(g *Graph, tags []model.Tag) []model.Tag {
	if g == nil {
		return tags
	}

	byName := make(map[string]model.Tag, len(tags)*3)
	for _, t := range tags {
		if cur, ok := byName[t.Name]; !ok || t.Confidence > cur.Confidence {
			byName[t.Name] = t
		}
	}

	for _, t := range tags {
		for _, anc := range g.Ancestors(t.Name) {
			at, ok := g.Get(anc)
			if !ok {
				continue
			}
			existing, seen := byName[anc]
			// Never downgrade a directly-scored ancestor, and never lower a
			// confidence already established by a stronger descendant.
			if seen && (existing.Asserted || existing.Confidence >= t.Confidence) {
				continue
			}
			byName[anc] = model.Tag{
				Name:          anc,
				Kind:          model.TagKind(at.Kind),
				Confidence:    t.Confidence,
				Depth:         g.Depth(anc),
				Path:          g.Path(anc),
				Asserted:      false,
				EvidenceDepth: t.EvidenceDepth,
			}
		}
	}

	out := make([]model.Tag, 0, len(byName))
	for _, t := range byName {
		out = append(out, t)
	}
	return out
}
