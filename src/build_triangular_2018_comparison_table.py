#!/usr/bin/env python3
"""Build a Table-2-style snapshot of Gurobi versus the 2018 SCIP MINLP."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent
INPUT = BASE / "triangular_table2_scip_minlp_comparison.csv"
OUTPUT_MD = BASE / "triangular_2018_comparison_table_current.md"
OUTPUT_CSV = BASE / "triangular_2018_comparison_summary_current.csv"
OUTPUT_TEX = BASE / "triangular_2018_comparison_table_current.tex"
N_VALUES = (100, 200, 300, 400, 500, 600)
BIGM = "CURRENT_TRIANGULAR_BIGM_FIXED_COST"
PERSPECTIVE = "CURRENT_TRIANGULAR_PERSPECTIVE_FIXED_COST"


def present(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def as_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite value: {value!r}")
    return result


def mean_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = [as_float(row[field]) for row in rows if present(row.get(field))]
    return statistics.fmean(values) if values else None


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.3E}"


def tex_cell(value: str) -> str:
    if value == "--":
        return "{--}"
    if "E" not in value:
        return f"${value}$"
    coefficient, exponent = value.split("E", 1)
    return f"${coefficient}\\mathrm{{E}}{{{exponent}}}$"


def is_scip_success(status: str) -> bool:
    text = status.lower()
    return "gap limit reached" in text or "optimal solution found" in text


def main() -> int:
    with INPUT.open("r", encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))

    summary: list[dict[str, object]] = []
    table_rows: list[tuple[str, str, str, list[str]]] = []
    for label, model_id in (("Big-M (9), Gurobi", BIGM), ("Perspective, Gurobi", PERSPECTIVE)):
        values: dict[str, list[str]] = {
            "time_avg": [], "time_all": [], "nodes_avg": [], "nodes_all": [], "limits": []
        }
        for n in N_VALUES:
            rows = [row for row in source if int(row["n"]) == n and row["model_id"] == model_id]
            optimal = [row for row in rows if row["status"] == "optimal"]
            values["time_avg"].append(fmt(mean_field(optimal, "solve_seconds")))
            values["time_all"].append(fmt(mean_field(rows, "solve_seconds")))
            values["nodes_avg"].append(fmt(mean_field(optimal, "node_count")))
            values["nodes_all"].append(fmt(mean_field(rows, "node_count")))
            values["limits"].append(str(sum(row["status"] != "optimal" for row in rows)))
            summary.append({
                "formulation": label, "n": n, "recorded_runs": len(rows),
                "successful_runs": len(optimal), "time_limit_runs": sum(row["status"] != "optimal" for row in rows),
                "avg_seconds_successful": mean_field(optimal, "solve_seconds"),
                "avg_seconds_including_limits": mean_field(rows, "solve_seconds"),
                "avg_nodes_successful": mean_field(optimal, "node_count"),
                "avg_nodes_including_limits": mean_field(rows, "node_count"),
            })
        table_rows.extend([
            (label, "CPU Sec", "Avg", values["time_avg"]),
            (label, "CPU Sec", "Avg inc lim", values["time_all"]),
            (label, "B&B Nodes", "Avg", values["nodes_avg"]),
            (label, "B&B Nodes", "Avg inc lim", values["nodes_all"]),
            (label, "Time Limit", "Number", values["limits"]),
        ])

    # The MINLP statistics are duplicated on the Big-M and perspective
    # reference rows.  Keep one row per instance by selecting Big-M rows.
    label = "2018 cubic MINLP, SCIP"
    values = {"time_avg": [], "time_all": [], "nodes_avg": [], "nodes_all": [], "limits": []}
    for n in N_VALUES:
        rows = [
            row for row in source
            if int(row["n"]) == n and row["model_id"] == BIGM and present(row.get("minlp_status"))
        ]
        successful = [row for row in rows if is_scip_success(row["minlp_status"])]
        limited = [row for row in rows if "time limit reached" in row["minlp_status"].lower()]
        values["time_avg"].append(fmt(mean_field(successful, "minlp_cpu_seconds")))
        values["time_all"].append(fmt(mean_field(rows, "minlp_cpu_seconds")))
        values["nodes_avg"].append(fmt(mean_field(successful, "minlp_node_count")))
        values["nodes_all"].append(fmt(mean_field(rows, "minlp_node_count")))
        values["limits"].append(str(len(limited)))
        summary.append({
            "formulation": label, "n": n, "recorded_runs": len(rows),
            "successful_runs": len(successful), "time_limit_runs": len(limited),
            "avg_seconds_successful": mean_field(successful, "minlp_cpu_seconds"),
            "avg_seconds_including_limits": mean_field(rows, "minlp_cpu_seconds"),
            "avg_nodes_successful": mean_field(successful, "minlp_node_count"),
            "avg_nodes_including_limits": mean_field(rows, "minlp_node_count"),
        })
    table_rows.extend([
        (label, "CPU Sec", "Avg", values["time_avg"]),
        (label, "CPU Sec", "Avg inc lim", values["time_all"]),
        (label, "B&B Nodes", "Avg", values["nodes_avg"]),
        (label, "B&B Nodes", "Avg inc lim", values["nodes_all"]),
        (label, "Time Limit", "Number", values["limits"]),
    ])

    header = ["Formulation", "Metric", "Statistic", *(str(n) for n in N_VALUES)]
    lines = [
        "# Triangular-demand comparison with the 2018 cubic MINLP",
        "",
        "| " + " | ".join(header) + " |",
        "|---|---|---:" + "|---:" * len(N_VALUES) + "|",
    ]
    for formulation, metric, statistic, cells in table_rows:
        lines.append(
            "| " + " | ".join((formulation, metric, statistic, *cells)) + " |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        fields = tuple(summary[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    minlp_n600 = next(
        row for row in summary
        if row["formulation"] == "2018 cubic MINLP, SCIP" and row["n"] == 600
    )
    n600_runs = int(minlp_n600["recorded_runs"])
    n_headers = [f"$n={n}$" for n in N_VALUES]
    if n600_runs < 30:
        n_headers[-1] = "$n=600^{\\dagger}$"
    tex_lines = [
        "\\documentclass[10pt]{article}", "", "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{booktabs}", "\\usepackage{graphicx}", "\\usepackage{multirow}",
        "", "\\begin{document}", "", "\\begin{table}[htbp]", "\\centering",
        "\\caption{Computational running-time comparison for triangular-distributed",
        "demands. The Gurobi Big-M and perspective formulations are compared with the",
        "2018 cubic MINLP formulation solved by SCIP. The relative optimality-gap",
        "target is $0.001$ ($0.1\\%$), with a one-hour time limit per solve.}",
        "\\label{tab:triangular-2018-comparison}",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllrrrrrr}",
        "\\toprule", "Formulation & Metric & Statistic", "  & " + " & ".join(n_headers) + " \\\\",
        "\\midrule",
    ]
    previous = None
    for formulation, metric, statistic, cells in table_rows:
        if previous is not None and formulation != previous:
            tex_lines.append("\\midrule")
        escaped_formulation = formulation.replace("&", "\\&")
        if formulation != previous:
            formulation_cell = f"\\multirow{{5}}{{*}}{{{escaped_formulation}}}"
        else:
            formulation_cell = ""
        metric_cell = metric.replace("&", "\\&")
        tex_lines.append(
            f"  {formulation_cell} & {metric_cell} & {statistic} & "
            + " & ".join(tex_cell(cell) for cell in cells)
            + " \\\\"
        )
        previous = formulation
    tex_lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
    if n600_runs < 30:
        tex_lines.extend([
            "", "\\vspace{0.5em}", "\\begin{minipage}{\\textwidth}", "\\footnotesize",
            f"$^{{\\dagger}}$The SCIP $n=600$ column is a partial snapshot based on {n600_runs} completed",
            "runs. The experiment is still running. The columns $n=100,\\ldots,500$",
            "contain all 30 runs. ``Avg'' uses solves that reached the configured gap;",
            "``Avg inc lim'' also includes time-limited solves.",
            "\\end{minipage}",
        ])
    tex_lines.extend(["\\end{table}", "", "\\end{document}", ""])
    OUTPUT_TEX.write_text("\n".join(tex_lines), encoding="utf-8")

    print(OUTPUT_MD)
    print(OUTPUT_CSV)
    print(OUTPUT_TEX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
