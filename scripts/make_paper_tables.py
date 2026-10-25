"""Write the paper's tables as LaTeX fragments, from artifacts/official_protocol/paper_numbers.json only.

Every number in a table comes from that file, so tables and released artifacts cannot drift apart.
The manuscript includes the fragments with \\input{tables/<name>.tex}.

    python -m scripts.make_paper_tables --out paper/icde2027/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ART = Path("artifacts/official_protocol")
SUB = ("Sub_KNN", "Sub_MCD", "Sub_OCSVM", "Sub_HBOS", "Sub_IForest", "Sub_LOF")


def f3(x: float) -> str:
    return f"{x:.3f}"


def s3(x: float) -> str:
    return f"${x:+.3f}$"


def ci(t: dict[str, Any], nd: int = 3) -> str:
    lo, hi = t["ci95"]
    return f"$[{lo:.{nd}f}$, ${hi:.{nd}f}]$".replace("$, $", ",\\,")


def delta_ci(t: dict[str, Any], nd: int = 3) -> str:
    lo, hi = t["ci95"]
    return f"${t['delta']:+.{nd}f}$ [${lo:.{nd}f}$, ${hi:.{nd}f}$]"


def table_toggle(n: dict[str, Any]) -> str:
    t = n["toggle"]["TSB-AD-U"]
    rows = []
    for d in SUB:
        r = t["detectors"][d]
        g = r["global_minus_window"]
        r0, r1 = t["rank_change"][d.replace("_", "-")]
        name = d.replace("_", "-") + ("$^{a}$" if d == "Sub_MCD" else "")
        rows.append(f"{name} & {f3(g['mean_b'])} & {f3(g['mean_a'])} & {delta_ci(g)} & "
                    f"{r0} $\\to$ {r1} \\\\")
    pca = t["detectors"]["Sub_PCA"]
    p0, p1 = t["rank_change"]["Sub-PCA"]
    rows.append("\\midrule")
    rows.append(f"Sub-PCA$^{{b}}$ & --- & {f3(pca['global_mean'])} & --- & {p0} $\\to$ {p1} \\\\")
    return "\n".join(rows) + "\n"


def table_bench(n: dict[str, Any]) -> str:
    out = []
    for subset, rows in (("TSB-AD-U", (("raw_flatten", "Raw-LW (global)"), ("Sub-PCA", "Sub-PCA (best published)"),
                                         ("KShapeAD", "KShapeAD"), ("MOMENT (FT)", "MOMENT (FT)"),
                                         ("raw_knn_wz", "Raw-1NN (per-window z)"), ("Sub-KNN", "Sub-KNN (published)"),
                                         ("TimesFM", "TimesFM"), ("Chronos", "Chronos"))),
                         ("TSB-AD-M", (("raw_flatten", "Raw-LW (global)"), ("CNN", "CNN (best published)"),
                                       ("OmniAnomaly", "OmniAnomaly"), ("raw_knn_wz", "Raw-1NN (per-window z)")))):
        b = n["baselines"][subset]
        mins = b["raw_knn_minus"]
        lines = [f"{subset}|Raw-1NN (global) & {f3(b['means']['raw_knn'])} & --- \\\\"]
        for key, label in rows:
            t = mins[key]
            lines.append(f"{subset}|{label} & {f3(t['mean_b'])} & {delta_ci(t)} \\\\")
        out += lines
    return "\n".join(out) + "\n"


def table_regime(n: dict[str, Any]) -> str:
    ter = n["tercile_sub_detectors"]
    reg = n["regime"]

    def r2(v: float) -> str:
        s = f"{v:.2f}"
        return "0.00" if s in ("-0.00", "0.00") else s

    def cell(x: dict[str, Any]) -> str:
        lo, hi = x["ci95"]
        m = f"{x['mean']:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00")
        return f"${m}$ [${r2(lo)}$,\\,${r2(hi)}$] ({x['wins']})"

    lines = []
    for d in SUB:
        r = ter[d]
        lines.append(f"{d.replace('_', '-')} & " + " & ".join(cell(r[k]) for k in ("low", "mid", "high"))
                     + f" & ${r['spearman_zr']:+.2f}$ \\\\")
    for key, label in (("LW factorial: per-window z minus global", "Raw-LW$^{*}$"),
                       ("raw_knn: per-window minus global", "Raw-1NN")):
        r = reg["TSB-AD-U"][key]
        lines.append(f"{label} & " + " & ".join(cell(r[k]) for k in ("low", "mid", "high"))
                     + f" & ${r['spearman_zr']:+.2f}$ \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Multivariate} (60 series per tercile)} \\\\")
    r = reg["TSB-AD-M"]["raw_knn: per-window minus global"]
    lines.append("Raw-1NN & " + " & ".join(cell(r[k]) for k in ("low", "mid", "high"))
                 + f" & ${r['spearman_zr']:+.2f}$ \\\\")
    return "\n".join(lines) + "\n"


def table_factorial(n: dict[str, Any]) -> str:
    c = n["factorial"]["cell_means"]
    lines = []
    for norm, label in (("global", "global (train z)"), ("window_z", "per-window z"),
                        ("window_minmax", "per-window min-max")):
        vals = [c[f"{norm}|{e}|{w}"] for e in ("lw", "mcd") for w in ("w100", "period3")]
        cells = [f3(v) for v in vals]
        if norm == "window_z":
            cells[3] += "$^{\\dagger}$"
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def table_threshold(n: dict[str, Any]) -> str:
    t = n["threshold2"]
    lines = []
    for drift, label in (("none", "none"), ("level1", "level $+1$"), ("level3", "level $+3$"),
                         ("scale2", "scale $\\times2$"), ("trend3", "trend $+3$")):
        cal = [t[f"{drift}|{m}"]["calibrated"] for m in ("global", "per_window", "rolling")]
        eq = [t[f"{drift}|{m}"]["equal_far_0.01"]["event_recall"] for m in ("global", "per_window", "rolling")]
        cells = [f"{c['far']:.2f}/{c['event_recall']:.2f}" for c in cal] + [f"{e:.2f}" for e in eq]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def table_scaling(n: dict[str, Any]) -> str:
    lines = []
    # The paper shows a subset (growth in reference size, stream length and channels); all
    # configurations are in artifacts/official_protocol/scaling.json.
    keep = {(1000, 3000, 1), (10000, 3000, 1), (100000, 3000, 1), (10000, 100000, 1), (10000, 3000, 51)}
    rows = sorted((r for r in n["scaling"]["rows"] if (r["n_ref"], r["n_eval"], r["channels"]) in keep),
                  key=lambda r: (r["method"], r["channels"], r["n_ref"], r["n_eval"]))
    for r in rows:
        name = {"raw_knn": "Raw-1NN", "raw_lw": "Raw-LW"}[r["method"]]
        head = f"{name} & {r['n_ref']:,} & {r['n_eval']:,} & {r['channels']}".replace(",", "{,}")
        if "skipped" in r:
            lines.append(f"{head} & \\multicolumn{{2}}{{c}}{{not run (memory)}} \\\\")
        else:
            lines.append(f"{head} & {r['total_s']:.2f} & {r['peak_rss_gb']:.2f} \\\\")
    return "\n".join(lines) + "\n"


def table_h9(n: dict[str, Any]) -> str:
    h = n["held_out_472"]

    def p(x: float) -> str:
        if x < 1e-3:
            return f"$<10^{{{int(np.ceil(np.log10(x)))}}}$"
        return f"{x:.2f}"

    lines = []
    for d in SUB:
        t = h["tests"][d]
        src = [h["by_source"][s][d] for s in ("YAHOO", "UCR", "WSD", "Stock")]
        lo, hi = t["ci95"]
        lines.append(f"{d.replace('_', '-')} & ${t['delta']:+.3f}$ [${lo:.2f}$,\\,${hi:.2f}$] & {p(t['p_holm'])} & "
                     + " & ".join(f"${v:+.2f}$".replace("+0.", "+.").replace("-0.", "-.") for v in src)
                     + " \\\\")
    return "\n".join(lines) + "\n"


def table_boundary(n: dict[str, Any]) -> str:
    c = n["boundary_synthetic"]["cells"]
    lines = []
    for kind, mag, label in (("level", 1.0, "level $+1\\sigma$"), ("level", 3.0, "level $+3\\sigma$"),
                             ("scale", 2.0, "scale $\\times2$"), ("scale", 3.0, "scale $\\times3$"),
                             ("shape", 0.0, "shape")):
        cells = []
        for r in (0.25, 1.0, 4.0, 16.0):
            g, w = c[f"{kind}|{mag}|{r}|global"], c[f"{kind}|{mag}|{r}|per_window"]
            cells.append(f"{g['VUS-PR']:.2f}/{w['VUS-PR']:.2f}")
        w16 = c[f"{kind}|{mag}|16.0|per_window"]
        cells.append(f"{w16['auc_interior']:.2f}")
        cells.append(f"{w16['auc_boundary']:.2f}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/icde2027/tables"))
    args = ap.parse_args()
    n = json.loads((ART / "paper_numbers.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "toggle_rows.tex").write_text(table_toggle(n))
    bench = table_bench(n)
    for subset in ("TSB-AD-U", "TSB-AD-M"):
        # The first column is a \multirow label, so each row starts with an empty cell.
        rows = [" & " + ln.split("|", 1)[1] for ln in bench.splitlines()
                if ln.startswith(subset + "|")]
        (args.out / f"bench_rows_{subset[-1]}.tex").write_text("\n".join(rows) + "\n")
    (args.out / "regime_rows.tex").write_text(table_regime(n))
    (args.out / "factorial_rows.tex").write_text(table_factorial(n))
    (args.out / "threshold_rows.tex").write_text(table_threshold(n))
    if n.get("boundary_synthetic"):
        (args.out / "boundary_rows.tex").write_text(table_boundary(n))
    if n.get("held_out_472", {}).get("tests"):
        (args.out / "h9_rows.tex").write_text(table_h9(n))
    if "scaling" in n:
        (args.out / "scaling_rows.tex").write_text(table_scaling(n))
    print("wrote", sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
