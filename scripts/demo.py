#!/usr/bin/env python3
"""Scripted end-to-end demo.

Registers the sample corpus, waits for the pipeline to finish, then exercises
both halves of the hybrid vocabulary: exact multi-tag queries against the
inverted index, and free-text queries resolved against stored frame vectors.

Run with `make demo` once the services are up.
"""

import json
import sys
import time
import urllib.parse
import urllib.request

INGEST = "http://localhost:8080"
SEARCH = "http://localhost:8081"
TAXO = "http://localhost:8082"

BOLD, DIM, GREEN, CYAN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[0m"

# Video IDs registered by this run. The catalog persists across runs and
# registering a path again creates a new video (deliberately — re-ingest is a
# real operation), so every result list is filtered to this run's own IDs to
# keep the demo deterministic.
MINE = set()


def mine(results):
    """Keep only hits belonging to videos this run registered."""
    return [r for r in results if r["video"]["id"] in MINE]


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def rule(title):
    print()
    print(BOLD + "── " + title + " " + "─" * max(0, 58 - len(title)) + RESET)


def register(paths):
    """Register each video and return the IDs we just created."""
    rule("INGEST — register the sample corpus")
    ids = []
    for p in paths:
        v = post(INGEST + "/api/videos/register", {"path": p})
        # 202 Accepted: the API returns before any processing happens.
        print("  %-10s %s  status=%s" % (v["filename"], v["id"][:8], v["status"]))
        ids.append(v["id"])
        MINE.add(v["id"])
    return ids


def wait_for_tagging(ids, timeout=180):
    """Wait for exactly the videos we just registered.

    Polling a global tagged count is wrong: a catalog may already hold rows from
    an earlier run, so the condition can be satisfied before our own videos have
    been touched at all.
    """
    rule("PIPELINE — asynchronous processing")
    start = time.time()
    terminal = {"TAGGED", "FAILED"}
    # TAGGING and TAGGED both truncate to "tagg", so abbreviate explicitly.
    short = {"PENDING": "pend", "DECODING": "deco", "TAGGING": "tag?",
             "TAGGED": "DONE", "FAILED": "FAIL"}
    while time.time() - start < timeout:
        states = []
        for vid in ids:
            try:
                states.append(get("%s/api/videos/%s" % (INGEST, vid))["status"])
            except Exception:
                states.append("PENDING")
        done = sum(1 for st in states if st == "TAGGED")
        failed = sum(1 for st in states if st == "FAILED")
        sys.stdout.write("\r  tagged %d/%d   failed %d   elapsed %.0fs   [%s]   "
                         % (done, len(ids), failed, time.time() - start,
                            " ".join(short.get(st, st[:4]) for st in states)))
        sys.stdout.flush()
        if all(st in terminal for st in states):
            print()
            return
        time.sleep(1)
    print()


def show_library(ids):
    rule("TAGS — what the model found")
    print(DIM + "  ● terminal (evidence stopped here)   ○ rolled up from deeper"
          + RESET)
    mine = set(ids)
    vids = [v for v in get(INGEST + "/api/videos") if v["id"] in mine]
    for v in sorted(vids, key=lambda x: x["filename"]):
        print("\n  %s%s%s  (%.0fs, %d frames, taxonomy v%d)" % (
            BOLD, v["filename"], RESET, v["duration_sec"],
            v["frame_count"], v["taxonomy_version"]))
        by_kind = {}
        for t in sorted(v["tags"], key=lambda t: (t["depth"], -t["confidence"])):
            mark = "●" if t["asserted"] else "○"
            by_kind.setdefault(t["kind"], []).append(
                "%s%s%s%s %.2f" % (mark, CYAN, t["path"], RESET, t["confidence"]))
        for kind in ("environment", "activity", "object"):
            if kind in by_kind:
                print("    %-12s %s" % (kind, "  ".join(by_kind[kind])))


def rollup_demo():
    """The roll-up query: the clearest demonstration of what the hierarchy buys."""
    rule("SEARCH — hierarchical roll-up")
    print(DIM + "  No video was tagged 'cleaning' as its terminal activity." + RESET)

    hits = mine(get(SEARCH + "/api/search?tags=cleaning&limit=200")["results"])
    print("\n  %s?tags=cleaning%s → %d hits" % (BOLD, RESET, len(hits)))
    for r in hits:
        v = r["video"]
        row = [t for t in v["tags"] if t["name"] == "cleaning"][0]
        via = [t["name"] for t in v["tags"]
               if t["path"].startswith("cleaning.") and t["asserted"]]
        print("    %-8s %.2f  %svia %s%s" % (
            v["filename"], row["confidence"], CYAN, ", ".join(via) or "-", RESET))

    hits = mine(get(SEARCH + "/api/search?tags=cleaning&exact=true&limit=200")["results"])
    print("\n  %s?tags=cleaning&exact=true%s → %d hits" % (BOLD, RESET, len(hits)))
    print("    " + DIM + "(no video's evidence STOPPED at 'cleaning' — "
          "every match came from a child)" + RESET)

    hits = mine(get(SEARCH + "/api/search?tags=surface_finishing&limit=200")["results"])
    print("\n  %s?tags=surface_finishing%s → %d hits  "
          "(reached through plastering's SECOND parent — the DAG)"
          % (BOLD, RESET, len(hits)))
    for r in hits:
        print("    %-8s %s" % (r["video"]["filename"], DIM + "rolled up" + RESET))


def taxonomy_growth():
    """The promotion loop: the taxonomy learns its own structure."""
    rule("TAXONOMY — growing the vocabulary at runtime")
    before = get(TAXO + "/api/taxonomy/version")["version"]
    print("  current version: %d" % before)

    for name, kind, prompt in [
        ("grout_scrubbing", "activity",
         "a first-person view of someone scrubbing tile grout with a stiff brush"),
        ("mopping", "activity",
         "a first-person view of someone mopping a floor"),
    ]:
        d = post(TAXO + "/api/taxonomy/place",
                 {"name": name, "kind": kind, "prompts": [prompt]})
        flag = (GREEN + "confident" + RESET) if not d["needs_review"] \
            else ("\033[33mNEEDS REVIEW\033[0m")
        print("\n  place %s%s%s → under %s%s%s   margin %.3f   %s" % (
            BOLD, name, RESET, CYAN, d["suggested"]["parents"][0], RESET,
            d["margin"], flag))
        for n in d["neighbours"][:3]:
            print("      %.3f  %s" % (n["similarity"], n["path"]))

    print("\n  " + DIM + "Placement is embedding-driven: the new term lands beside "
          "whatever it\n  is nearest to in the same space the frames live in. A "
          "near-tie is\n  flagged rather than guessed — the taxonomy is a "
          "suggestion engine,\n  not an authority." + RESET)


def tag_search(tags, mode):
    url = "%s/api/search?tags=%s&mode=%s&limit=200" % (
        SEARCH, urllib.parse.quote(",".join(tags)), mode)
    hits = mine(get(url)["results"])
    print("\n  %s%s%s of [%s] → %d hit(s)" % (
        BOLD, mode.upper(), RESET, ", ".join(tags), len(hits)))
    for r in hits:
        names = [t["name"] for t in r["video"]["tags"]][:5]
        print("    %.2f  %-8s %s" % (r["score"], r["video"]["filename"], DIM + " ".join(names) + RESET))
    if not hits:
        print("    " + DIM + "(correctly empty — no video satisfies that combination)" + RESET)


def semantic_search(q):
    url = SEARCH + "/api/search?limit=200&q=" + urllib.parse.quote(q)
    hits = mine(get(url)["results"])
    print('\n  "%s%s%s"' % (BOLD, q, RESET))
    for r in hits[:3]:
        print("    %.3f  %-8s  best frame @%ds" % (
            r["score"], r["video"]["filename"], round(r["match_ts"])))


def main():
    videos = ["videos/1.mp4", "videos/2.mp4", "videos/3.mp4", "videos/4.mp4"]
    ids = register(videos)
    wait_for_tagging(ids)

    show_library(ids)

    rule("SEARCH — closed vocabulary (exact, indexed)")
    tag_search(["kitchen"], "all")
    tag_search(["kitchen", "dishwashing"], "all")
    tag_search(["bed_making", "plastering"], "all")
    tag_search(["bedroom", "construction_site"], "any")

    rollup_demo()

    rule("SEARCH — open vocabulary (embedded at query time)")
    print(DIM + "  None of these phrases exist in the vocabulary." + RESET)
    for q in ("scrubbing a metal grate under running water",
              "someone spreading wet concrete overhead",
              "folding fabric onto a mattress",
              "workers inspecting items on an assembly line"):
        semantic_search(q)

    taxonomy_growth()

    rule("CORPUS")
    stats = get(INGEST + "/api/stats")
    print("  %d videos · %.1f min · %d frame vectors · %d distinct tags · taxonomy v%d" % (
        stats["videos"], stats["total_duration_sec"] / 60,
        stats["embeddings"], stats["distinct_tags"],
        get(TAXO + "/api/taxonomy/version")["version"]))
    print("\n  " + GREEN + "UI → http://localhost:8080" + RESET + "\n")


if __name__ == "__main__":
    main()
