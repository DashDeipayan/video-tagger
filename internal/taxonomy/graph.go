// Package taxonomy models the tag vocabulary as a versioned DAG.
//
// Three design points, each of which the flat vocabulary could not express:
//
//   - It is a *forest*, one tree per facet, not one tree rooted at environments.
//     "cooking happens in kitchens" is a context relation, not an is-a relation;
//     mixing the two makes roll-up semantics incoherent and forces nodes like
//     `cooking` to be duplicated under every environment they occur in.
//
//   - It is a *DAG*, not a tree. `plastering` is genuinely both construction
//     work and surface finishing. Forcing a single parent duplicates the node,
//     which fragments the index — the exact problem a closed vocabulary exists
//     to avoid.
//
//   - It is *versioned*. Restructuring the graph invalidates every materialised
//     ancestor row, so tagged rows record the version they were written under
//     and a backfill job reconciles them.
package taxonomy

import (
	"fmt"
	"sort"
)

// Kind is the facet a term belongs to. Facets are separate trees.
type Kind string

const (
	KindEnvironment Kind = "environment"
	KindActivity    Kind = "activity"
	KindObject      Kind = "object"
)

// Term is one node in the DAG.
type Term struct {
	Name string `json:"name"`
	Kind Kind   `json:"kind"`
	// Parents is a slice, not a string: a node may have several parents.
	// Empty means the term is a root of its facet.
	Parents []string `json:"parents"`
	// Prompts are the natural-language phrasings embedded to represent this
	// node. Several are averaged to smooth out an unlucky phrasing.
	Prompts []string `json:"prompts"`
	// Affinities are cross-facet context edges (kitchen <-> cooking). They are
	// deliberately NOT parent edges: they inform query expansion and act as a
	// prior, but they never roll up, because cooking is not a kind of kitchen.
	Affinities []string `json:"affinities,omitempty"`
	// Source records where a term came from, so machine-placed nodes can be
	// reviewed separately from curated ones.
	Source string `json:"source,omitempty"` // "curated" | "promoted"
}

// Graph is an immutable snapshot of the taxonomy at one version.
type Graph struct {
	Version int            `json:"version"`
	Terms   []Term         `json:"terms"`
	index   map[string]int // name -> position in Terms
	kids    map[string][]string
}

// New builds a Graph and validates its structure.
func New(version int, terms []Term) (*Graph, error) {
	g := &Graph{Version: version, Terms: terms,
		index: make(map[string]int, len(terms)), kids: map[string][]string{}}

	for i, t := range terms {
		if _, dup := g.index[t.Name]; dup {
			return nil, fmt.Errorf("duplicate term %q", t.Name)
		}
		g.index[t.Name] = i
	}
	for _, t := range terms {
		for _, p := range t.Parents {
			pt, ok := g.Get(p)
			if !ok {
				return nil, fmt.Errorf("term %q has unknown parent %q", t.Name, p)
			}
			if pt.Kind != t.Kind {
				// A parent edge that crosses facets is almost always a context
				// relation mislabelled as is-a. Those belong in Affinities.
				return nil, fmt.Errorf("term %q (%s) cannot have cross-facet parent %q (%s); use an affinity edge",
					t.Name, t.Kind, p, pt.Kind)
			}
			g.kids[p] = append(g.kids[p], t.Name)
		}
	}
	if cyc := g.findCycle(); cyc != "" {
		return nil, fmt.Errorf("taxonomy contains a cycle through %q", cyc)
	}
	return g, nil
}

func (g *Graph) Get(name string) (Term, bool) {
	i, ok := g.index[name]
	if !ok {
		return Term{}, false
	}
	return g.Terms[i], true
}

func (g *Graph) Has(name string) bool { _, ok := g.index[name]; return ok }

// Roots returns the top-level terms of one facet.
func (g *Graph) Roots(k Kind) []string {
	var out []string
	for _, t := range g.Terms {
		if t.Kind == k && len(t.Parents) == 0 {
			out = append(out, t.Name)
		}
	}
	sort.Strings(out)
	return out
}

// Children returns the direct descendants of a term.
func (g *Graph) Children(name string) []string {
	c := append([]string(nil), g.kids[name]...)
	sort.Strings(c)
	return c
}

// Kinds lists the facets present in the graph.
func (g *Graph) Kinds() []Kind {
	seen := map[Kind]bool{}
	var out []Kind
	for _, t := range g.Terms {
		if !seen[t.Kind] {
			seen[t.Kind] = true
			out = append(out, t.Kind)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Ancestors returns every term above name, nearest first, deduplicated.
//
// This is what gets materialised into the index at write time so that a
// roll-up query stays a single flat lookup instead of a recursive tree walk in
// the hot path.
func (g *Graph) Ancestors(name string) []string {
	seen := map[string]bool{name: true}
	var out []string
	frontier := []string{name}
	for len(frontier) > 0 {
		var next []string
		for _, n := range frontier {
			t, ok := g.Get(n)
			if !ok {
				continue
			}
			for _, p := range t.Parents {
				if !seen[p] {
					seen[p] = true
					out = append(out, p)
					next = append(next, p)
				}
			}
		}
		frontier = next
	}
	return out
}

// Depth is the length of the shortest path from a facet root. Roots are 0.
func (g *Graph) Depth(name string) int {
	t, ok := g.Get(name)
	if !ok || len(t.Parents) == 0 {
		return 0
	}
	best := -1
	for _, p := range t.Parents {
		if d := g.Depth(p) + 1; best < 0 || d < best {
			best = d
		}
	}
	return best
}

// Path renders the shortest root-to-node path, e.g. "cleaning.dishwashing".
// Stored alongside each tag so queries can be scoped to one branch and
// disambiguate `kitchen.cleaning` from `bathroom.cleaning`.
func (g *Graph) Path(name string) string {
	t, ok := g.Get(name)
	if !ok {
		return name
	}
	if len(t.Parents) == 0 {
		return name
	}
	best, bestDepth := "", -1
	for _, p := range t.Parents {
		if d := g.Depth(p); bestDepth < 0 || d < bestDepth {
			bestDepth, best = d, p
		}
	}
	return g.Path(best) + "." + name
}

// findCycle reports a term involved in a cycle, or "" if the graph is acyclic.
func (g *Graph) findCycle() string {
	const (
		white = 0
		grey  = 1
		black = 2
	)
	state := map[string]int{}
	var visit func(string) string
	visit = func(n string) string {
		switch state[n] {
		case grey:
			return n
		case black:
			return ""
		}
		state[n] = grey
		t, _ := g.Get(n)
		for _, p := range t.Parents {
			if c := visit(p); c != "" {
				return c
			}
		}
		state[n] = black
		return ""
	}
	for _, t := range g.Terms {
		if c := visit(t.Name); c != "" {
			return c
		}
	}
	return ""
}

// Subtree returns name plus every term beneath it, deduplicated.
//
// Used to build an interior node's representative vector. An abstract parent
// prompt ("someone moving and handling materials") is a far weaker CLIP signal
// than its concrete leaves ("hands sorting objects on an industrial table"), so
// routing on the parent's own prompt alone makes the cascade drop correct
// branches at the root.
func (g *Graph) Subtree(name string) []string {
	seen := map[string]bool{name: true}
	out := []string{name}
	for i := 0; i < len(out); i++ {
		for _, c := range g.Children(out[i]) {
			if !seen[c] {
				seen[c] = true
				out = append(out, c)
			}
		}
	}
	return out
}
