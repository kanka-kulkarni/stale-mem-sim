"""Build the LLM impact table for a domain with a local Ollama model.

Example:
    python -m experiments.rate_impacts --domain commute --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from simulator.domains import get_domain
from simulator.impact_llm import agreement, load_impact_table, rate_impacts
from simulator.llm import OllamaClient

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Rate every (task, attribute) pair once and write calibration/llm_impact_*.csv."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", default="commute")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-think", action="store_true",
                   help="send think=false (thinking models only); labelled \"think off\"")
    args = p.parse_args(argv)

    domain = get_domain(args.domain)
    tag = args.model.replace(":", "-")
    if args.no_think:
        tag += "-nothink"
    out = args.out or ROOT / "calibration" / f"llm_impact_{domain.name}_{tag}.csv"
    client = OllamaClient(args.model, cache_path=ROOT / "calibration" / "llm_cache.jsonl",
                          think=False if args.no_think else None)
    df = rate_impacts(domain, client)
    if args.no_think:
        df["model"] = f"{args.model} [think off]"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 160)
    wide = df.pivot(index="attribute", columns="task", values="rating")
    wide = wide.loc[[a.name for a in domain.attributes], [k.name for k in domain.tasks]]
    print(f"{domain.name}, model={args.model}: ratings 0-3 (NaN = invalid reply)\n")
    print(wide.to_string(na_rep="inv"))
    fresh = df[~df.cached]
    print(f"\ncalls={len(df)} invalid={int((~df.valid).sum())} new={len(fresh)} "
          f"latency_mean={fresh.latency_s.mean() if len(fresh) else 0:.2f}s "
          f"prompt_tokens={int(df.prompt_tokens.sum())} "
          f"completion_tokens={int(df.completion_tokens.sum())}")
    print("\nagreement with declared dependencies:\n"
          + agreement(domain, load_impact_table(out, domain)).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
