#!/usr/bin/env python3
"""Build a cluster expert placement for ranks of unequal size in one box.

The balanced builder (build_expert_placement.py) splits the experts evenly
across ranks. Here the ranks are unequal: rank 0 is the head with its own
three tiers (hot experts on its card, a secondary expert owner, the streamed
tier), the other ranks are discrete cards that hold their share in VRAM. Each
rank r >= 1 takes the hottest experts that fit its byte budget, the same number
in every layer, in rank order; rank 0 owns the rest. Protected experts stay on
rank 0 (the head keeps them on its card, --ds4-protected-experts).

    build_budget_placement.py --hotness routing.csv --ranks 2 \
        --budget-gib 1=14 --expert-mib 10.53 --out rankmap.json

    build_budget_placement.py --placement placement_lucebox.json --n-expert-used 6 \
        --ranks 3 --budget-gib 1=14 --budget-gib 2=9 --expert-mib 10.53 --out rankmap.json

Hotness: the CSV written by LUCE_DS4_ROUTING_STATS_OUT (one row per layer,
expert activation counts; the "# hotness table:" header gives the shape).
Without it, --placement ranks experts by their tier in a three-tier placement
file ({"owners": [...]}, index layer * n_expert + expert): primary first, then
secondary, then stream, by expert id inside a tier; a coarse order, replace it
with learned counts after the first run.

Output: the JSON ClusterExpertPlacement::load_json reads
(--cluster-expert-placement <file.json>), no replicated experts.

stdlib only.
"""

import argparse
import json
import re
import sys

TIER_RANK = {"primary": 0, "secondary": 1, "stream": 2}


def load_hotness_csv(path):
    header_re = re.compile(r"#\s*hotness table:\s*n_layer=(\d+)\s+n_expert=(\d+)\s+n_expert_used=(\d+)")
    shape = None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = header_re.match(line)
                if m:
                    shape = tuple(int(m.group(i)) for i in (1, 2, 3))
                continue
            rows.append([int(x) for x in line.split(",") if x.strip() != ""])
    if shape is None:
        raise ValueError("missing '# hotness table:' header")
    n_layer, n_expert, n_expert_used = shape
    if len(rows) != n_layer:
        raise ValueError("expected %d layer rows, found %d" % (n_layer, len(rows)))
    for il, row in enumerate(rows):
        if len(row) != n_expert:
            raise ValueError("layer %d has %d columns, expected %d" % (il, len(row), n_expert))
    return n_layer, n_expert, n_expert_used, rows


def load_tier_placement(path, n_expert):
    with open(path, "r", encoding="utf-8") as f:
        owners = json.load(f)["owners"]
    if n_expert <= 0 or len(owners) % n_expert:
        raise ValueError("%d owners do not divide into layers of %d experts" % (len(owners), n_expert))
    n_layer = len(owners) // n_expert
    rows = []
    for il in range(n_layer):
        tiers = owners[il * n_expert:(il + 1) * n_expert]
        bad = [t for t in tiers if t not in TIER_RANK]
        if bad:
            raise ValueError("unknown owner %r in layer %d" % (bad[0], il))
        # a pseudo-count: hotter tiers first, then lower expert ids (stable)
        rows.append([(3 - TIER_RANK[t]) * n_expert + (n_expert - e) for e, t in enumerate(tiers)])
    return n_layer, rows


def load_protected(path, n_layer, n_expert):
    protected = [set() for _ in range(n_layer)]
    if not path:
        return protected
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)
    for layer, experts in table.items():
        il = int(layer)
        if il < 0 or il >= n_layer:
            raise ValueError("protected experts: layer %d out of range" % il)
        for e in experts:
            if e < 0 or e >= n_expert:
                raise ValueError("protected experts: expert %d out of range" % e)
            protected[il].add(int(e))
    return protected


def parse_budgets(items, n_ranks):
    budgets = {}
    for item in items:
        rank, _, gib = item.partition("=")
        r = int(rank)
        if r < 1 or r >= n_ranks:
            raise ValueError("--budget-gib rank must be in [1, ranks), got %d" % r)
        budgets[r] = float(gib)
    for r in range(1, n_ranks):
        if r not in budgets:
            raise ValueError("--budget-gib %d=<GiB> is missing" % r)
    return budgets


def ranked_experts(counts):
    return sorted(range(len(counts)), key=lambda e: (-counts[e], e))


def build_budget(n_ranks, n_layer, n_expert, rows, per_layer, protected):
    """per_layer[r]: experts rank r takes in every layer (r >= 1), hottest first."""
    owner = []
    for il in range(n_layer):
        row = [0] * n_expert
        free = [e for e in ranked_experts(rows[il]) if e not in protected[il]]
        at = 0
        for r in range(1, n_ranks):
            take = free[at:at + per_layer[r]]
            if len(take) < per_layer[r]:
                raise ValueError("layer %d: rank %d wants %d experts, only %d are left"
                                 % (il, r, per_layer[r], len(take)))
            for e in take:
                row[e] = r
            at += per_layer[r]
        if at >= n_expert - len(protected[il]):
            raise ValueError("layer %d: rank 0 would own only protected experts" % il)
        owner.append(row)
    return owner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hotness", help="routing hotness CSV (LUCE_DS4_ROUTING_STATS_OUT)")
    src.add_argument("--placement", help="three-tier placement JSON instead of counts (coarse)")
    ap.add_argument("--n-expert", type=int, default=384, help="experts per layer with --placement (default 384)")
    ap.add_argument("--n-expert-used", type=int, default=6, help="routed experts per token with --placement (default 6)")
    ap.add_argument("--ranks", type=int, required=True, help="number of ranks, the head is rank 0")
    ap.add_argument("--budget-gib", action="append", default=[], metavar="RANK=GIB",
                    help="VRAM a rank >= 1 spends on experts, e.g. 1=14 (once per rank)")
    ap.add_argument("--expert-mib", type=float, required=True,
                    help="bytes of one expert in MiB (ds41-lucebox-final: 158 GiB / 15,360 = 10.53)")
    ap.add_argument("--protected", help="--ds4-protected-experts JSON; these stay on rank 0")
    ap.add_argument("--out", required=True, help="placement JSON for --cluster-expert-placement")
    a = ap.parse_args(argv)
    if a.ranks < 2 or a.ranks > 8:
        ap.error("--ranks must be in [2, 8]")
    if a.hotness:
        n_layer, n_expert, n_expert_used, rows = load_hotness_csv(a.hotness)
        source = "learned counts " + a.hotness
    else:
        n_layer, rows = load_tier_placement(a.placement, a.n_expert)
        n_expert, n_expert_used = a.n_expert, a.n_expert_used
        source = "tiers of " + a.placement + " (coarse: replace with learned counts)"
    budgets = parse_budgets(a.budget_gib, a.ranks)
    expert_bytes = a.expert_mib * 1024 * 1024
    per_layer = {r: int(budgets[r] * 1024 ** 3 // (n_layer * expert_bytes)) for r in budgets}
    protected = load_protected(a.protected, n_layer, n_expert)
    owner = build_budget(a.ranks, n_layer, n_expert, rows, per_layer, protected)
    out = {"n_ranks": a.ranks, "n_layer": n_layer, "n_expert": n_expert,
           "n_expert_used": n_expert_used, "replicate_hot": 0, "owner": owner}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")
    counts = [0] * a.ranks
    for row in owner:
        for o in row:
            counts[o] += 1
    print("placement from %s: %d layers x %d experts" % (source, n_layer, n_expert))
    for r in range(a.ranks):
        gib = counts[r] * expert_bytes / 1024 ** 3
        print("  rank %d: %d experts (%d per layer), %.1f GiB%s" % (
            r, counts[r], counts[r] // n_layer, gib, "" if r else " (its own three tiers decide what is resident)"))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
