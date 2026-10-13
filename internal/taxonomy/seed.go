package taxonomy

// Seed is the curated starting taxonomy, tuned to a workforce/gig-labour
// corpus. Three facets, each its own tree, with one genuine multi-parent node
// to exercise the DAG (`plastering` is both construction work and surface
// finishing).
//
// Adding a node costs one text embedding and a backfill over stored frame
// vectors. No video is ever decoded again.
var Seed = []Term{
	// ───────────────────────── environment ─────────────────────────
	{Name: "indoor", Kind: KindEnvironment, Prompts: []string{
		"a first-person point-of-view photo taken indoors"}},
	{Name: "outdoor", Kind: KindEnvironment, Prompts: []string{
		"a wearable camera view of an outdoor area"}},
	{Name: "industrial", Kind: KindEnvironment, Prompts: []string{
		"a wearable camera view of an industrial workplace"}},

	{Name: "kitchen", Kind: KindEnvironment, Parents: []string{"indoor"},
		Affinities: []string{"food_work", "cleaning"},
		Prompts: []string{
			"a first-person point-of-view photo taken inside a kitchen",
			"a kitchen countertop and sink seen from a wearable camera"}},
	{Name: "bedroom", Kind: KindEnvironment, Parents: []string{"indoor"},
		Affinities: []string{"housekeeping"},
		Prompts: []string{
			"a first-person point-of-view photo taken inside a bedroom",
			"a bed with sheets and pillows seen from above"}},
	{Name: "bathroom", Kind: KindEnvironment, Parents: []string{"indoor"},
		Affinities: []string{"cleaning"},
		Prompts:    []string{"a first-person point-of-view photo taken inside a bathroom"}},
	{Name: "restaurant", Kind: KindEnvironment, Parents: []string{"indoor"},
		Affinities: []string{"food_work"},
		Prompts:    []string{"a first-person point-of-view photo taken inside a restaurant kitchen"}},

	{Name: "construction_site", Kind: KindEnvironment, Parents: []string{"outdoor"},
		Affinities: []string{"construction_work"},
		Prompts: []string{
			"a wearable camera view of a construction site",
			"unfinished walls and scaffolding seen from a first-person view"}},
	{Name: "garden", Kind: KindEnvironment, Parents: []string{"outdoor"},
		Prompts: []string{"a wearable camera view of a garden or yard"}},

	{Name: "factory_floor", Kind: KindEnvironment, Parents: []string{"industrial"},
		Affinities: []string{"material_handling"},
		Prompts: []string{
			"a wearable camera view of a factory production line",
			"an industrial sorting table with stainless steel trays"}},
	{Name: "warehouse", Kind: KindEnvironment, Parents: []string{"industrial"},
		Affinities: []string{"material_handling"},
		Prompts:    []string{"a wearable camera view of a warehouse with shelving and pallets"}},
	{Name: "workshop", Kind: KindEnvironment, Parents: []string{"industrial"},
		Prompts: []string{"a wearable camera view of a workshop with tools"}},

	// ───────────────────────── activity ─────────────────────────
	{Name: "cleaning", Kind: KindActivity, Prompts: []string{
		"a first-person point-of-view video frame of someone cleaning",
		"hands wiping and scrubbing something clean"}},
	{Name: "housekeeping", Kind: KindActivity, Prompts: []string{
		"a first-person point-of-view video frame of someone tidying a room"}},
	{Name: "construction_work", Kind: KindActivity, Prompts: []string{
		"a first-person point-of-view video frame of someone doing construction work"}},
	{Name: "surface_finishing", Kind: KindActivity, Prompts: []string{
		"a first-person point-of-view video frame of someone coating or finishing a surface"}},
	{Name: "material_handling", Kind: KindActivity, Prompts: []string{
		"a first-person point-of-view video frame of someone moving and handling materials"}},
	{Name: "food_work", Kind: KindActivity, Prompts: []string{
		"a first-person point-of-view video frame of someone working with food"}},

	{Name: "dishwashing", Kind: KindActivity, Parents: []string{"cleaning"},
		Affinities: []string{"kitchen", "sink"},
		Prompts: []string{
			"a first-person point-of-view video frame of someone washing dishes in a sink",
			"hands scrubbing a dish under running water"}},
	{Name: "scrubbing", Kind: KindActivity, Parents: []string{"cleaning"},
		Prompts: []string{"a first-person point-of-view video frame of someone scrubbing a surface hard"}},
	{Name: "wiping", Kind: KindActivity, Parents: []string{"cleaning"},
		Prompts: []string{"a first-person point-of-view video frame of someone wiping a surface with a cloth"}},

	{Name: "bed_making", Kind: KindActivity, Parents: []string{"housekeeping"},
		Affinities: []string{"bedroom", "bedsheet"},
		Prompts: []string{
			"a first-person point-of-view video frame of someone making a bed",
			"hands spreading a bedsheet over a mattress"}},
	{Name: "laundry", Kind: KindActivity, Parents: []string{"housekeeping"},
		Prompts: []string{"a first-person point-of-view video frame of someone folding or washing laundry"}},

	{Name: "masonry", Kind: KindActivity, Parents: []string{"construction_work"},
		Prompts: []string{"a first-person point-of-view video frame of someone doing masonry and bricklaying work"}},
	{Name: "carpentry", Kind: KindActivity, Parents: []string{"construction_work"},
		Prompts: []string{"a first-person point-of-view video frame of someone doing carpentry with wood"}},

	// The DAG node: plastering is genuinely both. Forcing one parent would mean
	// duplicating it, which is what fragments a flat index.
	{Name: "plastering", Kind: KindActivity,
		Parents:    []string{"construction_work", "surface_finishing"},
		Affinities: []string{"construction_site", "trowel", "cement"},
		Prompts: []string{
			"a first-person point-of-view video frame of someone applying cement plaster to a wall",
			"a hand holding a trowel spreading wet concrete"}},
	{Name: "painting", Kind: KindActivity, Parents: []string{"surface_finishing"},
		Prompts: []string{"a first-person point-of-view video frame of someone painting a wall with a brush or roller"}},

	{Name: "sorting", Kind: KindActivity, Parents: []string{"material_handling"},
		Affinities: []string{"factory_floor", "conveyor"},
		Prompts: []string{
			"a first-person point-of-view video frame of someone sorting small items by hand",
			"hands sorting objects on an industrial table"}},
	{Name: "packing", Kind: KindActivity, Parents: []string{"material_handling"},
		Prompts: []string{"a first-person point-of-view video frame of someone packing items into a box"}},
	{Name: "lifting", Kind: KindActivity, Parents: []string{"material_handling"},
		Prompts: []string{"a first-person point-of-view video frame of someone lifting and carrying a heavy object"}},

	{Name: "cooking", Kind: KindActivity, Parents: []string{"food_work"},
		Affinities: []string{"kitchen"},
		Prompts:    []string{"a first-person point-of-view video frame of someone cooking food on a stove"}},
	{Name: "food_prep", Kind: KindActivity, Parents: []string{"food_work"},
		Prompts: []string{"a first-person point-of-view video frame of someone chopping and preparing food"}},

	// ───────────────────────── object ─────────────────────────
	{Name: "tool", Kind: KindObject, Prompts: []string{
		"a close-up first-person view of a hand tool"}},
	{Name: "fixture", Kind: KindObject, Prompts: []string{
		"a close-up first-person view of a fixed installation or appliance"}},
	{Name: "material", Kind: KindObject, Prompts: []string{
		"a close-up first-person view of raw building material"}},
	{Name: "linen", Kind: KindObject, Prompts: []string{
		"a close-up first-person view of cloth and linen"}},
	{Name: "container", Kind: KindObject, Prompts: []string{
		"a close-up first-person view of a container"}},

	{Name: "trowel", Kind: KindObject, Parents: []string{"tool"},
		Prompts: []string{"a close-up first-person view of a mason's trowel"}},
	{Name: "sponge", Kind: KindObject, Parents: []string{"tool"},
		Prompts: []string{"a close-up first-person view of a cleaning sponge"}},
	{Name: "hand_tools", Kind: KindObject, Parents: []string{"tool"},
		Prompts: []string{"a close-up first-person view of assorted hand tools"}},

	{Name: "sink", Kind: KindObject, Parents: []string{"fixture"},
		Prompts: []string{"a close-up first-person view of a kitchen sink"}},
	{Name: "faucet", Kind: KindObject, Parents: []string{"fixture"},
		Prompts: []string{"a close-up first-person view of a running faucet tap"}},
	{Name: "stove_grate", Kind: KindObject, Parents: []string{"fixture"},
		Prompts: []string{"a close-up first-person view of a cast iron stove grate"}},
	{Name: "conveyor", Kind: KindObject, Parents: []string{"fixture"},
		Prompts: []string{"a close-up first-person view of a stainless steel conveyor sorting table"}},

	{Name: "cement", Kind: KindObject, Parents: []string{"material"},
		Prompts: []string{"a close-up first-person view of wet cement mortar"}},
	{Name: "scaffolding", Kind: KindObject, Parents: []string{"material"},
		Prompts: []string{"a close-up first-person view of bamboo or metal scaffolding"}},

	{Name: "bedsheet", Kind: KindObject, Parents: []string{"linen"},
		Prompts: []string{"a close-up first-person view of a patterned bedsheet"}},
	{Name: "pillow", Kind: KindObject, Parents: []string{"linen"},
		Prompts: []string{"a close-up first-person view of a pillow"}},

	{Name: "bucket", Kind: KindObject, Parents: []string{"container"},
		Prompts: []string{"a close-up first-person view of a bucket"}},
	{Name: "cardboard_box", Kind: KindObject, Parents: []string{"container"},
		Prompts: []string{"a close-up first-person view of a cardboard box"}},
}

// MinPathConfidence is the single global floor on the chain-rule product along
// a path. It replaces the per-facet thresholds the flat vocabulary needed:
// because confidence decays with depth, one floor naturally stops descent at
// whatever granularity the evidence actually supports.
const MinPathConfidence = 0.15

// ExclusiveKinds marks facets whose top-level branches are mutually exclusive.
//
// A clip is in ONE kind of place, so asserting both `indoor.kitchen` and
// `outdoor.construction_site` for the same video is incoherent — and it is
// exactly what a softmax produces, because the mass must sum to 1 whether or
// not the right answer is among the siblings. When `indoor` narrowly wins the
// root and none of its children fit, one of them still gets crowned.
//
// Activities and objects are genuinely multiple (a video can involve cleaning
// AND food work, and contain many objects), so they are not exclusive.
var ExclusiveKinds = map[Kind]bool{
	KindEnvironment: true,
}

// BeamAt is how many siblings to descend into at a given depth.
//
// Greedy descent (beam 1) lets one wrong top-level choice poison everything
// beneath it. The beam is widest at the root, where branching is smallest and a
// mistake is most expensive, and narrows with depth where siblings are
// genuinely fine-grained alternatives.
func BeamAt(depth int) int {
	if depth == 0 {
		return 3
	}
	return 2
}
