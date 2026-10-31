"""Print exact-impact diagnostics for a domain's decision tables.

Example:
    python -m experiments.impact_table --domain commute
"""

from __future__ import annotations

import argparse

from simulator import diagnostics
from simulator.domains import get_domain


def main(argv: list[str] | None = None) -> int:
    """Print the attribute x task impact table, pair-flip tables, and dead value pairs."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", default="commute")
    p.add_argument("--pairs-for", nargs="*", default=None,
                   help="attributes to show pair-flip tables for (default: fast ones)")
    args = p.parse_args(argv)

    domain = get_domain(args.domain)
    states = diagnostics.reference_states(domain)
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print(f"domain={domain.name}, reference memory states={len(states)} (uniform)\n")

    table = diagnostics.impact_table(domain, states)
    table.insert(0, "class", [a.volatility for a in domain.attributes])
    table["max"] = table.drop(columns="class").max(axis=1)
    print("mean exact impact (attribute x task):\n" + table.to_string(float_format=fmt))

    pair_attrs = args.pairs_for
    if pair_attrs is None:
        pair_attrs = [a.name for a in domain.attributes if a.volatility == "fast"]
    for attr in pair_attrs:
        print(f"\nP(answer flips) when {attr} switches between two values:")
        print(diagnostics.pair_flip_table(domain, attr, states).to_string(float_format=fmt))

    dead = diagnostics.dead_value_pairs(domain, states)
    print("\nvalue pairs that never change any answer:",
          "none" if not dead else "".join(f"\n  {a}: {v} <-> {w}" for a, v, w in dead))
    return 0 if not dead else 1


if __name__ == "__main__":
    raise SystemExit(main())
