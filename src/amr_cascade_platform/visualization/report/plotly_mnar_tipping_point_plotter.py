"""Interactive MNAR decision-threshold tipping-point figure."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from amr_cascade_platform.visualization.report.organism_labels import format_organism_label


class PlotlyMNARTippingPointPlotter:
    """Export decision-threshold sensitivity figures for MNAR prevalence curves."""

    _DPI = 300
    _FONT = "DejaVu Sans"

    def export(
        self,
        curves: pd.DataFrame,
        tipping_points: pd.DataFrame,
        prevalence_summary: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        organism: str = "",
    ) -> dict[str, Path]:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        for fmt in formats:
            path = output_stem.with_suffix(f".{fmt}")
            if fmt == "html":
                self._write_html(curves, tipping_points, prevalence_summary, path, organism=organism)
            elif fmt in {"png", "svg", "pdf", "tiff"}:
                self._write_static(curves, tipping_points, prevalence_summary, fmt, path, organism=organism)
            else:
                continue
            outputs[path.name] = path
        return outputs

    def _write_html(
        self,
        curves: pd.DataFrame,
        tipping_points: pd.DataFrame,
        prevalence_summary: pd.DataFrame,
        path: Path,
        organism: str = "",
    ) -> None:
        curves_payload = self._records_for_json(
            curves,
            [
                "organism",
                "drug",
                "eligible_n",
                "tested_n",
                "resistant_n",
                "unknown_binary_outcome_n",
                "naive_prevalence_pct",
                "mnar_lambda",
                "mnar_prevalence_pct",
                "mnar_shift_from_naive_pct",
                "mnar_status",
            ],
        )
        tipping_payload = self._records_for_json(
            tipping_points,
            [
                "organism",
                "drug",
                "decision_threshold_pct",
                "mnar_lambda_star",
                "crossing_status",
                "naive_prevalence_pct",
                "prevalence_at_lambda0_pct",
                "shift_at_lambda0_pct",
                "eligible_n",
                "tested_n",
                "unknown_binary_outcome_n",
                "cascade_trigger_fraction",
                "rho_independent_vs_cascade",
            ],
        )
        summary_payload = self._records_for_json(
            prevalence_summary,
            [
                "organism",
                "drug",
                "prevalence_lower_bound_pct",
                "prevalence_upper_bound_pct",
                "cascade_trigger_fraction",
                "rho_independent_vs_cascade",
            ],
        )

        title = "MNAR Tipping Point: When Would Surveillance Interpretation Change?"
        if organism:
            title += f" — {format_organism_label(organism)}"
        if not curves_payload or not tipping_payload:
            path.write_text(
                self._empty_html(
                    title,
                    "No MNAR tipping-point data are available. This usually means no robust/supported "
                    "validated escalation edges met prevalence-shift support thresholds.",
                ),
                encoding="utf-8",
            )
            return

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #101828; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: end; margin-bottom: 12px; }}
    label {{ display: flex; flex-direction: column; font-size: 13px; font-weight: 700; color: #344054; gap: 4px; }}
    select {{ min-width: 240px; padding: 8px 10px; border: 1px solid #D0D5DD; border-radius: 8px; }}
    input {{ padding: 8px 10px; border: 1px solid #D0D5DD; border-radius: 8px; }}
    button {{ padding: 9px 12px; border: 1px solid #98A2B3; border-radius: 8px; background: #FFFFFF; cursor: pointer; font-weight: 700; }}
    button:hover {{ background: #F9FAFB; }}
    #caption {{ max-width: 1100px; color: #475467; font-size: 13px; line-height: 1.45; margin-top: 10px; }}
    #status {{ font-weight: 700; color: #344054; margin-top: 8px; }}
    .simulator {{ margin-top: 28px; padding-top: 20px; border-top: 1px solid #EAECF0; max-width: 1240px; }}
    .simulator h3 {{ margin-bottom: 4px; }}
    .sim-note {{ color: #475467; font-size: 13px; line-height: 1.45; max-width: 1100px; margin-bottom: 12px; }}
    .sim-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; align-items: end; }}
    .sim-grid label {{ min-width: 0; }}
    .sim-grid input[type="range"] {{ padding-left: 0; padding-right: 0; }}
    #simStatus {{ font-weight: 700; color: #344054; margin-top: 8px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="controls">
    <label>Drug
      <select id="drugSelect"></select>
    </label>
    <label>Decision threshold
      <select id="thresholdSelect"></select>
    </label>
  </div>
  <div id="chart" style="width: 100%; height: 680px;"></div>
  <div id="status"></div>
  <div id="caption">
    λ is an assumed residual log-odds shift in resistance among eligible opportunities without a binary-evaluable AST result.
    Positive λ tilts unobserved opportunities toward lower resistance after measured covariates and validated cascade triggers.
    The tipping point is a sensitivity threshold, not an estimate of true prevalence or clinical safety.
  </div>
  <section class="simulator">
    <h3>Interactive what-if simulator</h3>
    <div class="sim-note">
      This calculator is educational. It lets users change the observed tested denominator, observed resistant count,
      unknown eligible count, decision threshold, and λ assumption. It does not replace the pipeline estimates above.
    </div>
    <div class="sim-grid">
      <label>Binary-evaluable tested n
        <input id="simTested" type="number" min="1" step="1" value="400">
      </label>
      <label>Resistant among tested
        <input id="simResistant" type="number" min="0" step="1" value="200">
      </label>
      <label>Unknown eligible n
        <input id="simUnknown" type="number" min="0" step="1" value="600">
      </label>
      <label>Decision threshold (%)
        <input id="simThreshold" type="number" min="0" max="100" step="1" value="20">
      </label>
      <label>λ focus
        <input id="simLambda" type="range" min="-3" max="3" step="0.05" value="0">
        <span id="simLambdaValue">0.00</span>
      </label>
      <button type="button" id="simLoadSelected">Load selected drug</button>
    </div>
    <div id="simChart" style="width: 100%; height: 520px;"></div>
    <div id="simStatus"></div>
  </section>
  <script>
    const curves = {json.dumps(curves_payload)};
    const tipping = {json.dumps(tipping_payload)};
    const summaries = {json.dumps(summary_payload)};

    const drugSelect = document.getElementById("drugSelect");
    const thresholdSelect = document.getElementById("thresholdSelect");
    const statusBox = document.getElementById("status");
    const simTested = document.getElementById("simTested");
    const simResistant = document.getElementById("simResistant");
    const simUnknown = document.getElementById("simUnknown");
    const simThreshold = document.getElementById("simThreshold");
    const simLambda = document.getElementById("simLambda");
    const simLambdaValue = document.getElementById("simLambdaValue");
    const simLoadSelected = document.getElementById("simLoadSelected");
    const simStatus = document.getElementById("simStatus");

    const drugs = [...new Set(curves.map(d => d.drug))].sort();
    const thresholds = [...new Set(tipping.map(d => d.decision_threshold_pct))].sort((a, b) => a - b);
    for (const drug of drugs) {{
      const option = document.createElement("option");
      option.value = drug;
      option.textContent = drug;
      drugSelect.appendChild(option);
    }}
    for (const threshold of thresholds) {{
      const option = document.createElement("option");
      option.value = threshold;
      option.textContent = `${{threshold.toFixed(0)}}%`;
      if (Math.abs(threshold - 20) < 1e-9) option.selected = true;
      thresholdSelect.appendChild(option);
    }}

    function fmt(value, digits=1) {{
      return value === null || Number.isNaN(value) ? "NA" : Number(value).toFixed(digits);
    }}

    function clamp(value, lo, hi) {{
      return Math.max(lo, Math.min(hi, value));
    }}

    function expit(value) {{
      if (value >= 0) {{
        const z = Math.exp(-value);
        return 1 / (1 + z);
      }}
      const z = Math.exp(value);
      return z / (1 + z);
    }}

    function logit(probability) {{
      const p = clamp(probability, 1e-6, 1 - 1e-6);
      return Math.log(p / (1 - p));
    }}

    function simPrevalence(lambdaValue, tested, resistant, unknown) {{
      const eligible = tested + unknown;
      if (eligible <= 0 || tested <= 0) return NaN;
      const naive = clamp(resistant / tested, 1e-6, 1 - 1e-6);
      const unknownResistance = expit(logit(naive) - lambdaValue);
      return 100 * (resistant + unknown * unknownResistance) / eligible;
    }}

    function findSimulatorTippingPoint(points, thresholdPct) {{
      const candidates = [];
      for (let idx = 0; idx < points.length; idx += 1) {{
        if (Math.abs(points[idx].y - thresholdPct) < 1e-9) candidates.push(points[idx].x);
      }}
      for (let idx = 0; idx < points.length - 1; idx += 1) {{
        const y0 = points[idx].y;
        const y1 = points[idx + 1].y;
        if ((y0 - thresholdPct) * (y1 - thresholdPct) > 0) continue;
        if (Math.abs(y1 - y0) < 1e-12) continue;
        const frac = (thresholdPct - y0) / (y1 - y0);
        candidates.push(points[idx].x + frac * (points[idx + 1].x - points[idx].x));
      }}
      if (!candidates.length) return null;
      return candidates.reduce((best, value) => Math.abs(value) < Math.abs(best) ? value : best, candidates[0]);
    }}

    function loadSelectedDrugIntoSimulator() {{
      const drug = drugSelect.value;
      const threshold = Number(thresholdSelect.value);
      const tip = tipping.find(d => d.drug === drug && Math.abs(d.decision_threshold_pct - threshold) < 1e-9)
        || tipping.find(d => d.drug === drug);
      if (!tip) return;
      const tested = Math.max(1, Number(tip.tested_n || 1));
      const unknown = Math.max(0, Number(tip.unknown_binary_outcome_n || 0));
      const resistant = Math.round(clamp(Number(tip.naive_prevalence_pct || 0) / 100, 0, 1) * tested);
      simTested.value = tested;
      simUnknown.value = unknown;
      simResistant.value = resistant;
      simThreshold.value = Number(tip.decision_threshold_pct || threshold || 20).toFixed(0);
      drawSimulator();
    }}

    function drawSimulator() {{
      const tested = Math.max(1, Math.floor(Number(simTested.value) || 1));
      const resistant = clamp(Math.floor(Number(simResistant.value) || 0), 0, tested);
      const unknown = Math.max(0, Math.floor(Number(simUnknown.value) || 0));
      const threshold = clamp(Number(simThreshold.value) || 0, 0, 100);
      const lambdaFocus = Number(simLambda.value) || 0;
      simTested.value = tested;
      simResistant.value = resistant;
      simUnknown.value = unknown;
      simThreshold.value = threshold;
      simLambdaValue.textContent = fmt(lambdaFocus, 2);

      const points = [];
      for (let lambdaValue = -3; lambdaValue <= 3.0001; lambdaValue += 0.05) {{
        points.push({{ x: Number(lambdaValue.toFixed(2)), y: simPrevalence(lambdaValue, tested, resistant, unknown) }});
      }}
      const naivePct = 100 * resistant / tested;
      const eligible = tested + unknown;
      const focusPct = simPrevalence(lambdaFocus, tested, resistant, unknown);
      const lambdaStar = findSimulatorTippingPoint(points, threshold);
      const yValues = points.map(point => point.y).filter(value => Number.isFinite(value));
      const minCurve = Math.min(...yValues);
      const maxCurve = Math.max(...yValues);
      let crossingStatus = "not evaluable";
      if (lambdaStar !== null) crossingStatus = "crosses threshold";
      else if (threshold < minCurve) crossingStatus = "always above threshold";
      else if (threshold > maxCurve) crossingStatus = "always below threshold";

      const traces = [
        {{
          x: points.map(point => point.x),
          y: points.map(point => point.y),
          type: "scatter",
          mode: "lines",
          name: "What-if MNAR curve",
          line: {{ color: "#1F77B4", width: 4 }},
          hovertemplate: "λ=%{{x:.2f}}<br>Prevalence=%{{y:.2f}}%<extra></extra>"
        }},
        {{
          x: [lambdaFocus],
          y: [focusPct],
          type: "scatter",
          mode: "markers",
          name: `Current λ=${{fmt(lambdaFocus, 2)}}`,
          marker: {{ size: 13, color: "#8E44AD", line: {{ color: "#101828", width: 1 }} }},
          hovertemplate: `Current λ: ${{fmt(lambdaFocus, 2)}}<br>Prevalence: ${{fmt(focusPct, 2)}}%<extra></extra>`
        }}
      ];
      if (lambdaStar !== null) {{
        traces.push({{
          x: [lambdaStar],
          y: [threshold],
          type: "scatter",
          mode: "markers",
          name: `Tipping λ*=${{fmt(lambdaStar, 2)}}`,
          marker: {{ size: 13, color: "#2E7D32", line: {{ color: "#101828", width: 1 }} }},
          hovertemplate: `λ*: ${{fmt(lambdaStar, 2)}}<br>Threshold: ${{fmt(threshold, 1)}}%<extra></extra>`
        }});
      }}

      Plotly.react("simChart", traces, {{
        template: "plotly_white",
        title: "Live what-if calculator: changing the denominator changes the tipping point",
        xaxis: {{ title: "λ: residual log-odds tilt for unknown eligible episodes", range: [-3, 3] }},
        yaxis: {{ title: "Eligible-denominator resistance prevalence (%)", rangemode: "tozero" }},
        shapes: [
          {{
            type: "line", xref: "paper", x0: 0, x1: 1, y0: threshold, y1: threshold,
            line: {{ color: "#C0392B", width: 2, dash: "dot" }}
          }},
          {{
            type: "line", xref: "paper", x0: 0, x1: 1, y0: naivePct, y1: naivePct,
            line: {{ color: "#344054", width: 1.5, dash: "dash" }}
          }},
          {{
            type: "line", x0: lambdaFocus, x1: lambdaFocus, yref: "paper", y0: 0, y1: 1,
            line: {{ color: "#8E44AD", width: 1.3, dash: "dot" }}
          }}
        ],
        annotations: [
          {{ xref: "paper", x: 1, y: threshold, xanchor: "right", yanchor: "bottom", showarrow: false,
             text: `Threshold ${{fmt(threshold, 0)}}%`, font: {{ color: "#C0392B" }} }},
          {{ xref: "paper", x: 1, y: naivePct, xanchor: "right", yanchor: "top", showarrow: false,
             text: `Naive ${{fmt(naivePct, 1)}}%`, font: {{ color: "#344054" }} }}
        ],
        legend: {{ orientation: "h", y: -0.22 }},
        margin: {{ l: 80, r: 40, t: 80, b: 120 }},
        hovermode: "closest"
      }}, {{ responsive: true }});

      simStatus.textContent =
        `Eligible n=${{eligible}}, tested n=${{tested}}, unknown n=${{unknown}}, naive prevalence=${{fmt(naivePct, 1)}}%. ` +
        `At λ=${{fmt(lambdaFocus, 2)}}, prevalence=${{fmt(focusPct, 2)}}%. ` +
        `Status: ${{crossingStatus}}; λ*=${{lambdaStar === null ? "NA" : fmt(lambdaStar, 2)}}.`;
    }}

    function update() {{
      const drug = drugSelect.value;
      const threshold = Number(thresholdSelect.value);
      const rows = curves
        .filter(d => d.drug === drug && d.mnar_status === "estimated")
        .sort((a, b) => a.mnar_lambda - b.mnar_lambda);
      const tip = tipping.find(d => d.drug === drug && Math.abs(d.decision_threshold_pct - threshold) < 1e-9);
      const summary = summaries.find(d => d.drug === drug) || {{}};
      const x = rows.map(d => d.mnar_lambda);
      const y = rows.map(d => d.mnar_prevalence_pct);
      const hover = rows.map(d =>
        `Drug: ${{drug}}<br>` +
        `λ: ${{fmt(d.mnar_lambda, 2)}}<br>` +
        `MNAR prevalence: ${{fmt(d.mnar_prevalence_pct, 2)}}%<br>` +
        `Shift from naive: ${{fmt(d.mnar_shift_from_naive_pct, 2)}} pp<br>` +
        `Eligible: ${{d.eligible_n}}<br>` +
        `Tested: ${{d.tested_n}}<br>` +
        `Unknown binary outcome: ${{d.unknown_binary_outcome_n}}`
      );
      const traces = [
        {{
          x, y,
          type: "scatter",
          mode: "lines+markers",
          name: "MNAR prevalence curve",
          line: {{ color: "#1F77B4", width: 4 }},
          marker: {{ size: 8, color: "#1F77B4" }},
          text: hover,
          hovertemplate: "%{{text}}<extra></extra>"
        }}
      ];
      if (tip && tip.naive_prevalence_pct !== null) {{
        traces.push({{
          x: [Math.min(...x), Math.max(...x)],
          y: [tip.naive_prevalence_pct, tip.naive_prevalence_pct],
          type: "scatter",
          mode: "lines",
          name: `Naive prevalence (${{fmt(tip.naive_prevalence_pct, 1)}}%)`,
          line: {{ color: "#344054", width: 2, dash: "dash" }},
          hoverinfo: "skip"
        }});
      }}
      if (tip && tip.crossing_status === "crosses_threshold" && tip.mnar_lambda_star !== null) {{
        traces.push({{
          x: [tip.mnar_lambda_star],
          y: [threshold],
          type: "scatter",
          mode: "markers",
          name: `Tipping point λ*=${{fmt(tip.mnar_lambda_star, 2)}}`,
          marker: {{ size: 14, color: "#2E7D32", line: {{ color: "#101828", width: 1 }} }},
          hovertemplate: `λ*: ${{fmt(tip.mnar_lambda_star, 2)}}<br>Threshold: ${{fmt(threshold, 1)}}%<extra></extra>`
        }});
      }}

      const shapes = [
        {{
          type: "line", xref: "paper", x0: 0, x1: 1, y0: threshold, y1: threshold,
          line: {{ color: "#C0392B", width: 2, dash: "dot" }}
        }},
        {{
          type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1,
          line: {{ color: "#8E44AD", width: 1.5, dash: "dot" }}
        }}
      ];
      if (summary.prevalence_lower_bound_pct !== undefined && summary.prevalence_lower_bound_pct !== null) {{
        shapes.push({{
          type: "rect", xref: "paper", x0: 0, x1: 1,
          y0: summary.prevalence_lower_bound_pct, y1: summary.prevalence_upper_bound_pct,
          fillcolor: "rgba(152, 162, 179, 0.16)", line: {{ width: 0 }}, layer: "below"
        }});
      }}
      const status = tip ? tip.crossing_status.replaceAll("_", " ") : "not evaluable";
      statusBox.textContent =
        `Status: ${{status}}. λ* = ${{tip && tip.mnar_lambda_star !== null ? fmt(tip.mnar_lambda_star, 2) : "NA"}}. ` +
        `Prevalence at λ=0 = ${{tip ? fmt(tip.prevalence_at_lambda0_pct, 2) : "NA"}}%. ` +
        `This is a sensitivity diagnostic, not a recovered truth.`;

      Plotly.react("chart", traces, {{
        template: "plotly_white",
        title: `${{drug}}: MNAR tipping-point sensitivity`,
        xaxis: {{ title: "λ: residual log-odds tilt for unobserved eligible episodes", zeroline: false }},
        yaxis: {{ title: "Eligible-denominator resistance prevalence (%)", rangemode: "tozero" }},
        shapes,
        annotations: [
          {{ xref: "paper", x: 1, y: threshold, xanchor: "right", yanchor: "bottom", showarrow: false,
             text: `Decision threshold: ${{fmt(threshold, 0)}}%`, font: {{ color: "#C0392B" }} }},
          {{ x: 0, yref: "paper", y: 1, xanchor: "left", showarrow: false,
             text: "λ=0", font: {{ color: "#8E44AD" }} }}
        ],
        legend: {{ orientation: "h", y: -0.20 }},
        margin: {{ l: 80, r: 40, t: 80, b: 120 }},
        hovermode: "closest"
      }}, {{ responsive: true }});
    }}
    drugSelect.addEventListener("change", update);
    thresholdSelect.addEventListener("change", update);
    for (const input of [simTested, simResistant, simUnknown, simThreshold, simLambda]) {{
      input.addEventListener("input", drawSimulator);
      input.addEventListener("change", drawSimulator);
    }}
    simLoadSelected.addEventListener("click", loadSelectedDrugIntoSimulator);
    drugSelect.value = drugs[0];
    update();
    loadSelectedDrugIntoSimulator();
  </script>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")

    def _write_static(
        self,
        curves: pd.DataFrame,
        tipping_points: pd.DataFrame,
        prevalence_summary: pd.DataFrame,
        fmt: str,
        path: Path,
        organism: str = "",
    ) -> None:
        if curves.empty or tipping_points.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No MNAR tipping-point data available.",
                ha="center",
                va="center",
                fontsize=13,
                fontfamily=self._FONT,
            )
            fig.savefig(path, format=fmt, dpi=self._DPI, bbox_inches="tight")
            plt.close(fig)
            return

        threshold = self._default_threshold_pct(tipping_points)
        threshold_rows = tipping_points[tipping_points["decision_threshold_pct"].eq(threshold)].copy()
        threshold_rows["_rank"] = threshold_rows["mnar_lambda_star"].abs()
        threshold_rows.loc[threshold_rows["mnar_lambda_star"].isna(), "_rank"] = math.inf
        drugs = threshold_rows.sort_values(["crossing_status", "_rank", "eligible_n"], ascending=[True, True, False])[
            "drug"
        ].dropna().head(9).tolist()
        if not drugs:
            drugs = curves["drug"].dropna().unique().tolist()[:9]

        cols = min(3, len(drugs))
        rows = math.ceil(len(drugs) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.3, rows * 3.8), squeeze=False, constrained_layout=True)
        summary_lookup = prevalence_summary.set_index("drug").to_dict("index") if "drug" in prevalence_summary.columns else {}
        tip_lookup = threshold_rows.set_index("drug").to_dict("index") if "drug" in threshold_rows.columns else {}

        for idx, drug in enumerate(drugs):
            ax = axes[idx // cols][idx % cols]
            drug_curves = curves[(curves["drug"] == drug) & (curves["mnar_status"] == "estimated")].sort_values("mnar_lambda")
            if drug_curves.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{drug}\nnot evaluable", ha="center", va="center", fontsize=10)
                continue
            ax.plot(
                drug_curves["mnar_lambda"],
                drug_curves["mnar_prevalence_pct"],
                color="#1F77B4",
                linewidth=2.2,
                marker="o",
                markersize=4,
            )
            ax.axhline(threshold, color="#C0392B", linestyle=":", linewidth=1.6)
            ax.axvline(0.0, color="#8E44AD", linestyle=":", linewidth=1.2)
            tip = tip_lookup.get(drug, {})
            if tip.get("crossing_status") == "crosses_threshold" and pd.notna(tip.get("mnar_lambda_star")):
                ax.scatter([tip["mnar_lambda_star"]], [threshold], s=60, color="#2E7D32", zorder=5)
            summary = summary_lookup.get(drug, {})
            if pd.notna(summary.get("naive_prevalence_pct")):
                ax.axhline(summary["naive_prevalence_pct"], color="#344054", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_title(drug, fontsize=11, fontweight="bold", fontfamily=self._FONT)
            ax.set_xlabel("λ", fontsize=9, fontfamily=self._FONT)
            ax.set_ylabel("Prevalence (%)", fontsize=9, fontfamily=self._FONT)
            ax.grid(axis="y", color="#E8EEF5", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(len(drugs), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        title = f"MNAR Tipping-Point Sensitivity at {threshold:.0f}% Decision Threshold"
        if organism:
            title += f" — {format_organism_label(organism)}"
        fig.suptitle(title, fontsize=15, fontweight="bold", fontfamily=self._FONT)
        fig.text(
            0.01,
            -0.01,
            "Blue curve = assumption-indexed eligible-denominator prevalence. Red dotted line = decision threshold. "
            "Green point = interpolated tipping point when the curve crosses the threshold.",
            fontsize=10,
            color="#475467",
            fontfamily=self._FONT,
        )
        fig.savefig(path, format=fmt, dpi=self._DPI, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _default_threshold_pct(tipping_points: pd.DataFrame) -> float:
        thresholds = pd.to_numeric(tipping_points.get("decision_threshold_pct"), errors="coerce").dropna().unique()
        if len(thresholds) == 0:
            return 20.0
        return float(min(thresholds, key=lambda value: abs(value - 20.0)))

    @staticmethod
    def _records_for_json(data: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
        if data.empty:
            return []
        available = [column for column in columns if column in data.columns]
        cleaned = data.loc[:, available].copy()
        records: list[dict[str, object]] = []
        for raw in cleaned.to_dict(orient="records"):
            row: dict[str, object] = {}
            for key, value in raw.items():
                if pd.isna(value):
                    row[key] = None
                elif hasattr(value, "item"):
                    row[key] = value.item()
                else:
                    row[key] = value
            records.append(row)
        return records

    @staticmethod
    def _empty_html(title: str, message: str) -> str:
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family: Arial, sans-serif; margin: 32px;">
  <h2>{title}</h2>
  <p>{message}</p>
</body>
</html>
"""
