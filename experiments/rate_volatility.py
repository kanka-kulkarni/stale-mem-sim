"""LLM-rated volatility: rate how often each fact typically
changes, once per fact, and write calibration/llm_volatility_<domain>_<tag>.csv.

Example:
    python -m experiments.rate_volatility --model qwen2.5:7b
    python -m experiments.rate_volatility --model qwen3:8b --no-think
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simulator.domains import get_domain
from simulator.llm import OllamaClient
from simulator.volatility_llm import rate_volatility

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Rate every fact of the requested domains once."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--domain", nargs="+", default=["commute", "meals_calibrated"])
    p.add_argument("--no-think", action="store_true", help="send think=false")
    args = p.parse_args(argv)
    tag = args.model.replace(":", "-") + ("-nothink" if args.no_think else "")
    label = f"{args.model} [think off]" if args.no_think else args.model
    client = OllamaClient(args.model, cache_path=ROOT / "calibration" / "llm_cache.jsonl",
                          think=False if args.no_think else None)
    for name in args.domain:
        df = rate_volatility(get_domain(name), client, label)
        out = ROOT / "calibration" / f"llm_volatility_{name}_{tag}.csv"
        df.to_csv(out, index=False)
        fresh = df[~df.cached]
        print(f"{name} {label}: invalid={int((~df.valid).sum())} new={len(fresh)} "
              f"latency_mean={fresh.latency_s.mean() if len(fresh) else 0:.1f}s")
        print(df[["attribute", "volatility", "rating", "rate"]].to_string(index=False))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
