"""More rater families and LLM-rated volatility. CPU only: uses the cached rating tables written by
`experiments.rate_impacts` and `experiments.rate_volatility`; no model calls.

Stages:
  sims     rater simulations for Table 4 (sxi/budget with each new rater's table)
  e1ext    new relevance raters as extra "does it matter?" columns of E1 (own Holm family)
  volrow   "LLM-rated change rate" row: staleness x impact with mapped rates, coarse and
           exact impact, per rater; tests vs the age-only and true-rate rows (Holm)
  report   volatility agreement (Spearman, confusion), E3 over-inclusion for new raters

Example:
    python -m experiments.rater_families sims e1ext volrow report
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from experiments import run
from experiments.value_study import (OUT, SELECTS, TARGETS, TOL, Recorder, ValuePolicy,
                                     cell_family, e2_rows, setup)
from experiments.value_study import CHANGED as E1_CHANGED
from experiments.value_study import MATTERS as E1_MATTERS
from simulator import core, metrics, stats
from simulator.core import Domain
from simulator.impact_llm import load_impact_table
from simulator.matching import calibrate
from simulator.policies import Family
from simulator.volatility_llm import CLASS_ORDER, rate_table

ROOT = OUT.parents[1]
CAL = ROOT / "calibration"
EXISTING = {"3b": ("qwen2.5-3b", "Qwen2.5-3B"), "7b": ("qwen2.5-7b", "Qwen2.5-7B"),
            "14b": ("qwen2.5-14b", "Qwen2.5-14B"), "llama8b": ("llama3.1-8b", "Llama-3.1-8B"),
            "qwen35": ("qwen3.5-9b-nothink", "Qwen3.5-9B (thinking off)")}
# filled after the pull: thinking off for models whose Ollama capabilities list thinking
NEW = {"gemma4": ("gemma4-e4b-nothink", "Gemma 4 E4B (thinking off)"),
       "phi4mini": ("phi4-mini-3.8b", "Phi-4-mini"),
       "granite": ("granite4.1-8b", "Granite 4.1 8B"),
       "qwen3": ("qwen3-8b-nothink", "Qwen3-8B (thinking off)")}
ALL = {**EXISTING, **NEW}
DOMAINS = ("commute", "meals_calibrated")
VOL = OUT.parent / "rater_families"


def short(domain: str) -> str:
    return "commute" if domain == "commute" else "meals"


def impact_table(d: Domain, key: str) -> dict:
    return load_impact_table(CAL / f"llm_impact_{short(d.name)}_{ALL[key][0]}.csv", d)


def volatility(d: Domain, key: str) -> pd.DataFrame:
    return pd.read_csv(CAL / f"llm_volatility_{d.name}_{ALL[key][0]}.csv")


# --------------------------------------------------------------------------- sims


def stage_sims() -> None:
    """Table 4 rater simulations for the new raters (same command as the old ones)."""
    for key, (tag, _) in NEW.items():
        for dom in DOMAINS:
            out = ROOT / "results" / "raters" / f"{dom}_{tag}"
            if (out / "summary.csv").exists():
                continue
            run.main(["--domain", dom, "--policies", "never", "always", "--match-families",
                      "sxi[llm]", "budget[llm]", "--llm-impact",
                      str(CAL / f"llm_impact_{short(dom)}_{tag}.csv"), "--users", "50",
                      "--sessions", "20", "--seeds", "0", "1", "2", "3", "4", "--tune-seeds",
                      "1000", "1001", "1002", "1003", "1004", "--out", str(out)])


# --------------------------------------------------------------------------- shared runner


def run_families(S, fams: list[Family], with_e2: bool = False) -> tuple[pd.DataFrame, list, list]:
    """Tune each family to 25% and 50%, evaluate on held-out users, add the bracketing
    settings as sweep points (for the cost check)."""
    d, records, cals, e2 = S.domain, [], [], []
    for fam in fams:
        params = set()
        for target in TARGETS:
            cal = calibrate(d, S.tuning, fam, target, TOL)
            cals.append(cal.to_dict())
            pol = cal.build(fam)
            if pol is None:
                continue
            params |= {p for p in (cal.param_a, cal.param_b) if p is not None}
            rec = Recorder(pol)
            records += core.run(d, S.held, [rec])
            if with_e2:
                e2.append({"family": fam.name, "policy": pol.name, "target_cost": target,
                           **e2_rows(d, rec, S.states, fam.make(0.0))})
        records += core.run(d, S.held, [fam.make(p) for p in sorted(params)])
    return pd.DataFrame(records), cals, e2


def paired_tests(pairs: list[tuple[str, str, str]], sessions: dict, summaries: dict) -> pd.DataFrame:
    """Exact McNemar for (domain, policy_a, policy_b) pairs plus the cost check."""
    rows = []
    for dom, a, b in pairs:
        s = summaries[dom]
        comp = metrics.matched_comparisons(s[s.policy.isin([a, b]) | (s.role == "sweep")
                                             & s.family.isin(s[s.policy.isin([a, b])].family)])
        comp = comp[((comp.policy_a == a) & (comp.policy_b == b)) |
                    ((comp.policy_a == b) & (comp.policy_b == a))]
        rows.append(stats.add_mcnemar(comp, sessions[dom]))
    out = pd.concat(rows, ignore_index=True)
    out["p_holm"] = stats.holm(out.p_mcnemar.tolist())
    return out


def e1_frames(dom: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = OUT / f"e1_{dom}"
    return pd.read_csv(sub / "summary.csv"), pd.read_csv(sub / "sessions_matched.csv.gz")


# --------------------------------------------------------------------------- e1ext


class RaterCell(ValuePolicy):
    """E1 cell with a new rater's impact table (keys outside value_study.RATERS)."""

    def __init__(self, d: Domain, changed: str, key: str, select: str, param: float, table: dict):
        super().__init__(d, changed, f"llm-{key}", select, param, table)


def stage_e1ext() -> None:
    """New raters as extra E1 columns; tests vs cells sharing a row or column."""
    for dom in DOMAINS:
        S = setup(dom)
        d = S.domain
        fams = []
        for key in NEW:
            table = impact_table(d, key)
            for sel in SELECTS:
                for chg in E1_CHANGED:
                    hi = 20.0 if chg == "age" else 1.0
                    mk = (lambda x, c=chg, s=sel, t=table, k=key: RaterCell(d, c, k, s, x, t))
                    name = cell_family(sel, chg, f"llm-{key}")
                    fams.append(Family(name, "integer", mk, values=tuple(range(len(d.attributes) + 1)))
                                if sel == "budget" else Family(name, "continuous", mk, bounds=(0.0, hi)))
        df, cals, e2 = run_families(S, fams, with_e2=True)
        out = VOL / f"e1ext_{dom}"
        out.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.concat([df, pd.read_csv(OUT / f"e1_{dom}" / "sessions_reference.csv.gz")])
        metrics.summarize(metrics_df).to_csv(out / "summary.csv", index=False)
        df[df.role == "matched"].to_csv(out / "sessions_matched.csv.gz", index=False)
        pd.DataFrame(e2).to_csv(out / "e2_scores.csv", index=False)
        (out / "calibration.json").write_text(json.dumps(cals, indent=1))
    # tests: each new cell vs every cell sharing its row or column (old or new)
    pairs, sessions, summaries = [], {}, {}
    for dom in DOMAINS:
        old_s, old_x = e1_frames(dom)
        new_s = pd.read_csv(VOL / f"e1ext_{dom}" / "summary.csv")
        new_x = pd.read_csv(VOL / f"e1ext_{dom}" / "sessions_matched.csv.gz")
        summaries[dom] = pd.concat([old_s, new_s[~new_s.policy.isin(old_s.policy)]])
        sessions[dom] = pd.concat([old_x, new_x])
        cols = list(E1_MATTERS) + [f"llm-{k}" for k in NEW]
        for key in NEW:
            for sel in SELECTS:
                for t in TARGETS:
                    for chg in E1_CHANGED:
                        me = f"{cell_family(sel, chg, f'llm-{key}')}@{t:g}"
                        mates = [f"{cell_family(sel, chg, m)}@{t:g}" for m in cols if m != f"llm-{key}"]
                        mates += [f"{cell_family(sel, c, f'llm-{key}')}@{t:g}" for c in E1_CHANGED if c != chg]
                        pairs += [(dom, me, m) for m in mates]
    uniq = list(dict.fromkeys(tuple(sorted(p[1:])) + (p[0],) for p in pairs))
    comp = paired_tests([(u[2], u[0], u[1]) for u in uniq], sessions, summaries)
    comp.to_csv(VOL / "e1ext_comparisons.csv", index=False)
    print(f"E1 extension: {len(comp)} tests, {(comp.p_holm < 0.05).sum()} significant (Holm)")


# --------------------------------------------------------------------------- volrow


class RatedRatePolicy(ValuePolicy):
    """Staleness x impact (Threshold) with LLM-rated change rates replacing p_a."""

    def __init__(self, d: Domain, key: str, impact: str, param: float, rates: dict):
        super().__init__(d, "rate", impact, "threshold", param)
        self.rates = dict(rates)
        self.family = f"threshold[vol-{key} x {impact}]"
        self.name = f"{self.family}({param:.6g})"


def stage_volrow() -> None:
    """The LLM-rated change-rate row, per rater, coarse and exact impact."""
    pairs, sessions, summaries = [], {}, {}
    for dom in DOMAINS:
        S = setup(dom)
        d = S.domain
        fams = []
        for key in ALL:
            rates = rate_table(volatility(d, key))
            for imp in ("coarse", "exact"):
                mk = (lambda x, k=key, i=imp, r=rates: RatedRatePolicy(d, k, i, x, r))
                fams.append(Family(f"threshold[vol-{key} x {imp}]", "continuous", mk, bounds=(0.0, 1.0)))
        df, cals, _ = run_families(S, fams)
        out = VOL / f"volrow_{dom}"
        out.mkdir(parents=True, exist_ok=True)
        ref = pd.read_csv(OUT / f"e1_{dom}" / "sessions_reference.csv.gz")
        new_s = metrics.summarize(pd.concat([df, ref]))
        new_s.to_csv(out / "summary.csv", index=False)
        df[df.role == "matched"].to_csv(out / "sessions_matched.csv.gz", index=False)
        (out / "calibration.json").write_text(json.dumps(cals, indent=1))
        old_s, old_x = e1_frames(dom)
        summaries[dom] = pd.concat([old_s, new_s[~new_s.policy.isin(old_s.policy)]])
        sessions[dom] = pd.concat([old_x, df[df.role == "matched"]])
        for key in ALL:
            for imp in ("coarse", "exact"):
                for t in TARGETS:
                    me = f"threshold[vol-{key} x {imp}]@{t:g}"
                    for row in ("age", "rate"):
                        pairs.append((dom, me, f"{cell_family('threshold', row, imp)}@{t:g}"))
    comp = paired_tests(pairs, sessions, summaries)
    comp.to_csv(VOL / "volrow_comparisons.csv", index=False)
    print(f"Volatility row: {len(comp)} tests, {(comp.p_holm < 0.05).sum()} significant (Holm)")


# --------------------------------------------------------------------------- report


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman correlation via average ranks (no SciPy)."""
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def stage_report_impl() -> None:
    """Volatility agreement and E3 over-inclusion for every rater."""
    from simulator.domains import get_domain
    agree, conf = [], []
    for key, (tag, name) in ALL.items():
        frames = [volatility(get_domain(dom), key) for dom in DOMAINS]
        both = pd.concat(frames)
        cls = both.volatility.map(CLASS_ORDER)
        rating = both.rating.fillna(0)
        row = {"rater": name, "key": key, "spearman_all": spearman(cls.tolist(), rating.tolist()),
               "invalid": int((~both.valid).sum())}
        for dom, f in zip(DOMAINS, frames):
            row[f"spearman_{short(dom)}"] = spearman(f.volatility.map(CLASS_ORDER).tolist(),
                                                     f.rating.fillna(0).tolist())
        agree.append(row)
        for c in ("static", "slow", "fast"):
            sub = both[both.volatility == c]
            conf.append({"rater": name, "class": c, "n": len(sub),
                         **{f"rated_{r}": int((sub.rating.fillna(0) == r).sum()) for r in range(4)}})
    VOL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agree).to_csv(VOL / "volatility_agreement.csv", index=False)
    pd.DataFrame(conf).to_csv(VOL / "volatility_confusion.csv", index=False)
    # E3: measured over-inclusion c and the share of the gap it explains
    rows = []
    for dom in DOMAINS:
        d = get_domain(dom)
        e3 = pd.read_csv(OUT / f"e3_{dom}" / "summary.csv")
        e3 = e3[e3.role == "matched"].set_index("policy").accuracy
        e1 = pd.read_csv(OUT / f"e1_{dom}" / "summary.csv").set_index("policy").accuracy
        ext = VOL / f"e1ext_{dom}" / "summary.csv"
        e1x = pd.read_csv(ext).set_index("policy").accuracy if ext.exists() else pd.Series(dtype=float)
        for key, (tag, name) in ALL.items():
            table = impact_table(d, key)
            c = float(np.mean([v for (k, a), v in table.items() if a not in d.task(k).depends_on]))
            for t in TARGETS:
                lv = np.array([0, 0.25, 0.5, 0.75, 1.0])
                curve = np.array([e3[f"threshold[over={x:g}]@{t:g}"] for x in lv])
                pred = float(np.interp(c, lv, curve))
                exact = e1[f"threshold[rate x exact]@{t:g}"]
                pol = f"threshold[rate x llm-{key}]@{t:g}"
                actual = e1.get(pol, e1x.get(pol, np.nan))
                rows.append({"domain": dom, "rater": name, "key": key, "target_cost": t, "c": c,
                             "acc_exact": exact, "acc_overinclusion_only": pred,
                             "acc_rater": actual,
                             "share_explained": (exact - pred) / (exact - actual)})
    pd.DataFrame(rows).to_csv(VOL / "e3_overinclusion.csv", index=False)
    print("wrote", VOL)


CONFUSION_NAMES = {"3b": "Qwen2.5-3B", "7b": "Qwen2.5-7B", "14b": "Qwen2.5-14B",
                   "llama8b": "Llama-3.1-8B", "qwen35": "Qwen3.5-9B think-off",
                   "gemma4": "Gemma 4 E4B think-off", "phi4mini": "Phi-4-mini",
                   "granite": "Granite 4.1 8B", "qwen3": "Qwen3-8B think-off"}


def stage_confusion() -> None:
    """calibration/rater_confusion.csv for every rater (rating > 0 vs dependency list)."""
    from simulator.domains import get_domain
    rows = []
    for dom in ("commute", "meals"):
        d = get_domain(dom)
        for key, (tag, _) in ALL.items():
            df = pd.read_csv(CAL / f"llm_impact_{dom}_{tag}.csv")
            dep = [a in d.task(k).depends_on for k, a in zip(df.task, df.attribute)]
            rel = (df.rating.fillna(0) > 0).tolist()
            counts = "/".join(str(int((df.rating == r).sum())) for r in range(4))
            tp = sum(x and y for x, y in zip(dep, rel))
            fn = sum(x and not y for x, y in zip(dep, rel))
            fp = sum(y and not x for x, y in zip(dep, rel))
            tn = sum(not x and not y for x, y in zip(dep, rel))
            rows.append({"domain": dom, "rater": CONFUSION_NAMES[key], "ratings 0/1/2/3": counts,
                         "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                         "invalid": int((~df.valid).sum()), "disagree": fn + fp})
    pd.DataFrame(rows).to_csv(CAL / "rater_confusion.csv", index=False)


TAB = ROOT / "paper" / "tables"
SHORT = {"3b": "Qwen2.5-3B", "7b": "Qwen2.5-7B", "14b": "Qwen2.5-14B", "llama8b": "Llama-3.1-8B",
         "qwen35": r"Qwen3.5-9B$^{\dagger}$", "gemma4": r"Gemma 4 E4B$^{\dagger}$",
         "phi4mini": "Phi-4-mini", "granite": "Granite 4.1 8B", "qwen3": r"Qwen3-8B$^{\dagger}$"}


def stage_tables() -> None:
    """LaTeX tables: volatility agreement, and the LLM-rated change-rate row."""
    agree = pd.read_csv(VOL / "volatility_agreement.csv").set_index("key")
    conf = pd.read_csv(VOL / "volatility_confusion.csv")
    f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"  # noqa: E731
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Spearman with class} & \multicolumn{3}{c}{Ratings 0/1/2/3 by class} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             r"Rater & All & Commute & Meals & Static (1) & Slow (10) & Fast (4) \\", r"\midrule"]
    for key in ALL:
        a = agree.loc[key]
        cells = [f2(a.spearman_all), f2(a.spearman_commute), f2(a.spearman_meals)]
        for c in ("static", "slow", "fast"):
            r = conf[(conf.rater == a.rater) & (conf["class"] == c)].iloc[0]
            cells.append("/".join(str(int(r[f"rated_{i}"])) for i in range(4)))
        lines.append(SHORT[key] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "volatility_agreement.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    comp = pd.read_csv(VOL / "volrow_comparisons.csv")

    def mark(dom: str, pol: str, other: str) -> str:
        c = comp[comp.domain == dom]
        r = c[((c.policy_a == pol) & (c.policy_b == other)) |
              ((c.policy_b == pol) & (c.policy_a == other))]
        assert len(r) <= 1, (dom, pol, other)
        if r.empty or r.p_holm.iloc[0] >= 0.05:
            return ""
        return "+" if r.leader.iloc[0] == pol else "-"

    acc = {}
    for dom in DOMAINS:
        s = pd.read_csv(VOL / f"volrow_{dom}" / "summary.csv").set_index("policy").accuracy
        e1 = pd.read_csv(OUT / f"e1_{dom}" / "summary.csv").set_index("policy").accuracy
        acc[dom] = (s, e1)
    cols = [(d, i, t) for d in DOMAINS for i in ("coarse", "exact") for t in TARGETS]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{4}{c}{Commute} & \multicolumn{4}{c}{Meal planning} \\",
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
             r" & \multicolumn{2}{c}{Coarse} & \multicolumn{2}{c}{Exact} & "
             r"\multicolumn{2}{c}{Coarse} & \multicolumn{2}{c}{Exact} \\",
             r"Change source & 25\% & 50\% & 25\% & 50\% & 25\% & 50\% & 25\% & 50\% \\", r"\midrule"]
    for label, row in (("Age only", "age"), ("True rate and age", "rate")):
        lines.append(label + " & " + " & ".join(
            f"{acc[d][1][f'threshold[{row} x {i}]@{t:g}']:.3f}".lstrip("0") for d, i, t in cols) + r" \\")
    lines.append(r"\midrule")
    for key in ALL:
        cells = []
        for d, i, t in cols:
            pol = f"threshold[vol-{key} x {i}]@{t:g}"
            m1 = mark(d, pol, f"threshold[age x {i}]@{t:g}")
            m2 = mark(d, pol, f"threshold[rate x {i}]@{t:g}")
            sup = (r"$^{\uparrow}$" if m1 == "+" else r"$^{\downarrow}$" if m1 == "-" else "") + \
                  (r"$^{\circ}$" if m2 == "-" else r"$^{\star}$" if m2 == "+" else "")
            cells.append(f"{acc[d][0][pol]:.3f}".lstrip("0") + sup)
        lines.append(SHORT[key] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "volatility_row.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


STAGES = {"tables": stage_tables, "confusion": stage_confusion, "sims": stage_sims, "e1ext": stage_e1ext, "volrow": stage_volrow,
          "report": stage_report_impl}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stages", nargs="+", choices=list(STAGES))
    for name in p.parse_args(argv).stages:
        STAGES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
