#!/usr/bin/env python3
"""Render the taxonomy DAG from `GET /api/taxonomy/tree` as an indented tree."""

import json
import sys

C = {"environment": "\033[36m", "activity": "\033[33m", "object": "\033[35m"}
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def walk(node, facet, prefix="", last=True, seen=None):
    seen = seen if seen is not None else set()
    branch = "└─ " if last else "├─ "
    # A node with several parents is printed under each of them; the marker
    # flags the repeat so the DAG is visible rather than looking like a dupe.
    dag = ""
    if len(node.get("parents") or []) > 1:
        dag = DIM + "  ⇄ also under " + ", ".join(
            p for p in node["parents"] if p != node.get("_via")) + RESET
    promoted = DIM + " ✦promoted" + RESET if node.get("source") == "promoted" else ""
    print("%s%s%s%s%s%s%s" % (
        prefix, branch, C.get(facet, ""), node["name"], RESET, dag, promoted))

    kids = node.get("children") or []
    for i, k in enumerate(kids):
        k["_via"] = node["name"]
        walk(k, facet, prefix + ("   " if last else "│  "), i == len(kids) - 1, seen)


def main():
    d = json.load(sys.stdin)
    print("\n%staxonomy v%d%s" % (BOLD, d["version"], RESET))
    total = 0
    for facet, roots in sorted(d["facets"].items()):
        print("\n%s%s%s" % (C.get(facet, "") + BOLD, facet, RESET))
        for i, r in enumerate(roots):
            walk(r, facet, "  ", i == len(roots) - 1)
        total += sum(count(r) for r in roots)
    print("\n%d nodes\n" % total)


def count(n):
    return 1 + sum(count(c) for c in (n.get("children") or []))


if __name__ == "__main__":
    main()
