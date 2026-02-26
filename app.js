const DATA_URL = "./data/leaderboard.json";

let runsViewApi = null;
let runBreakdownModalApi = null;
const runPayloadCache = new Map();

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(digits)}%`;
}

function asrClass(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  if (value <= 0.2) return "good";
  if (value <= 0.45) return "warn";
  return "bad";
}

function inferModelType(modelName) {
  const name = String(modelName || "").toLowerCase();

  const apiPatterns = [
    /^gpt-/,
    /^o[1-9](-|$)/,
    /^claude-/,
    /^gemini/,
    /^deepseek-v/,
    /^doubao-/,
    /^grok-/,
    /^command-r/,
    /^mistral-(small|medium|large)/,
  ];
  if (apiPatterns.some((pattern) => pattern.test(name))) {
    return "api";
  }

  const localPrefixes = [
    "meta-llama_",
    "microsoft_",
    "mistralai_",
    "qwen_",
    "01-ai_",
    "deepseek-ai_",
    "google_gemma",
    "tiiuae_",
    "thudm_",
    "internlm_",
  ];
  if (localPrefixes.some((prefix) => name.startsWith(prefix))) {
    return "local";
  }

  return "local";
}

function withSubRank(rows) {
  return rows.map((row, idx) => ({ ...row, sub_rank: idx + 1 }));
}

function sortByRobustnessDesc(rows) {
  return [...rows].sort((a, b) => {
    const ra = a?.robustness_score;
    const rb = b?.robustness_score;
    const aMissing = ra === null || ra === undefined || Number.isNaN(ra);
    const bMissing = rb === null || rb === undefined || Number.isNaN(rb);
    if (aMissing !== bMissing) return aMissing ? 1 : -1;
    if (!aMissing && ra !== rb) return rb - ra;
    return String(a?.model || "").localeCompare(String(b?.model || ""));
  });
}

function metricHint(text, hint, className = "") {
  const cls = className ? `metric-hint ${className}` : "metric-hint";
  return `<span class="${esc(cls)}" title="${esc(hint)}">${esc(text)}</span>`;
}

function toNum(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") {
    const t = value.trim().toLowerCase();
    if (!t) return null;
    if (["safe", "harmless", "benign", "rejected"].includes(t)) return 0;
    if (["unsafe", "harmful", "jailbreak", "attack_success"].includes(t)) return 1;
    const parsed = Number(t);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function normalizeScore(value) {
  const scalar = toNum(value);
  if (scalar !== null) {
    const scaled = scalar > 1 && scalar <= 100 ? scalar / 100 : scalar;
    return Math.max(0, Math.min(1, scaled));
  }

  if (Array.isArray(value)) {
    const subs = value.map(normalizeScore).filter((v) => v !== null);
    if (!subs.length) return null;
    return subs.reduce((a, b) => a + b, 0) / subs.length;
  }

  if (value && typeof value === "object") {
    const subs = Object.values(value)
      .map(normalizeScore)
      .filter((v) => v !== null);
    if (!subs.length) return null;
    return subs.reduce((a, b) => a + b, 0) / subs.length;
  }

  return null;
}

function weightedMean(items) {
  let numer = 0;
  let denom = 0;
  for (const [val, weight] of items) {
    if (val === null || val === undefined || Number.isNaN(val)) continue;
    if (weight === null || weight === undefined || weight <= 0 || Number.isNaN(weight)) continue;
    numer += val * weight;
    denom += weight;
  }
  if (denom <= 0) return null;
  return numer / denom;
}

function uniqueSorted(values) {
  return Array.from(new Set(values.filter(Boolean))).sort((a, b) => String(a).localeCompare(String(b)));
}

function setModelJudgerNote(judgers) {
  const node = document.getElementById("modelJudgerNote");
  if (!node) return;
  const parts = uniqueSorted(
    (judgers || [])
      .flatMap((j) => String(j || "").split("+"))
      .map((s) => s.trim())
      .filter(Boolean)
  );
  if (!parts.length) {
    node.textContent = "Default judgers: N/A.";
    return;
  }
  node.innerHTML = `Default judgers: ${parts.map((p) => `<code>${esc(p)}</code>`).join(", ")}.`;
}

function setMeta(meta) {
  document.getElementById("metaGeneratedAt").textContent = meta.generated_at || "-";
  document.getElementById("metaSource").textContent = meta.source || "-";
  document.getElementById("metaSchema").textContent = meta.schema_version || "-";
}

function renderOverview(overview, modelRows, defenseRows, attackRows) {
  const modelNames = Array.isArray(overview?.models) ? overview.models : modelRows.map((r) => r.model).filter(Boolean);
  const apiModelCount = modelNames.filter((m) => inferModelType(m) === "api").length;
  const localModelCount = modelNames.filter((m) => inferModelType(m) === "local").length;
  const topApiModel = modelRows.find((r) => inferModelType(r.model) === "api")?.model || "N/A";
  const topLocalModel = modelRows.find((r) => inferModelType(r.model) === "local")?.model || "N/A";

  const primaryCards = [
    { label: "Completed Runs", value: overview.run_count },
    { label: "Sample Completion", value: fmtPct(overview.sample_completion_rate) },
    { label: "Models (Total)", value: overview.model_count },
    { label: "Models (API)", value: apiModelCount },
    { label: "Models (Local)", value: localModelCount },
    { label: "Attacks", value: overview.attack_count },
    { label: "Defenses", value: overview.defense_count },
  ];
  const highlightCards = [
    { label: "Top API Model (Robustness)", value: topApiModel, text: true },
    { label: "Top Local Model (Robustness)", value: topLocalModel, text: true },
    { label: "Best Defense Gain", value: fmtPct(defenseRows[0]?.asr_gain_vs_no_defense) },
    { label: "Hardest Attack", value: attackRows[0]?.attack || "N/A", text: true },
  ];

  const renderCard = (card) => {
    const cls = card.text ? "card card-text" : "card";
    return `<article class="${cls}"><span class="label">${esc(card.label)}</span><span class="value">${esc(
      card.value
    )}</span></article>`;
  };

  const container = document.getElementById("overviewCards");
  container.innerHTML = `
    <div class="overview-row overview-row-primary">${primaryCards.map(renderCard).join("")}</div>
    <div class="overview-row overview-row-highlights">${highlightCards.map(renderCard).join("")}</div>
  `;
}

function renderTable(tableId, columns, rows, rowClick) {
  const table = document.getElementById(tableId);
  if (!rows.length) {
    table.innerHTML = `<tbody><tr><td class="empty" colspan="${columns.length || 1}">No data</td></tr></tbody>`;
    return;
  }

  const thead = `<thead><tr>${columns
    .map((col) => `<th>${esc(col.label)}</th>`)
    .join("")}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map((row, idx) => {
      const attrs = [];
      if (rowClick) {
        attrs.push(`data-row-index="${idx}"`);
        attrs.push('class="clickable-row"');
      }
      const tds = columns
        .map((col) => {
          const value = col.render ? col.render(row) : row[col.key];
          const cls = col.className ? ` class="${col.className}"` : "";
          return col.html ? `<td${cls}>${value}</td>` : `<td${cls}>${esc(value)}</td>`;
        })
        .join("");
      return `<tr ${attrs.join(" ")}>${tds}</tr>`;
    })
    .join("")}</tbody>`;
  table.innerHTML = `${thead}${tbody}`;

  if (rowClick) {
    table.querySelectorAll("tbody tr").forEach((tr) => {
      tr.addEventListener("click", () => {
        const idx = Number(tr.getAttribute("data-row-index"));
        rowClick(rows[idx]);
      });
    });
  }
}

function setMatrixLegend(legendEl) {
  if (!legendEl) return;
  legendEl.innerHTML =
    '<div class="swatch"></div><span>Lower ASR</span><span style="margin-left:auto">Higher ASR</span>';
}

function renderModelAttackMatrix(matrix) {
  const table = document.getElementById("matrixTable");
  const legend = document.getElementById("matrixLegend");
  if (!table || !legend || !matrix) return;

  const min = matrix?.range?.min_asr ?? 0;
  const max = matrix?.range?.max_asr ?? 1;
  const span = Math.max(max - min, 1e-6);
  setMatrixLegend(legend);

  const attacks = Array.isArray(matrix.attacks) ? matrix.attacks : [];
  const rows = Array.isArray(matrix.rows) ? matrix.rows : [];
  if (!attacks.length || !rows.length) {
    table.innerHTML = '<tbody><tr><td class="empty">No matrix data</td></tr></tbody>';
    return;
  }

  const head = `<thead><tr><th>Model</th>${attacks.map((a) => `<th>${esc(a)}</th>`).join("")}</tr></thead>`;

  const body = `<tbody>${rows
    .map((row) => {
      const cells = (row.cells || [])
        .map((cell) => {
          if (cell.asr === null || cell.asr === undefined) {
            return `<td title="No run">-</td>`;
          }
          const ratio = (cell.asr - min) / span;
          const hue = Math.max(0, 140 - ratio * 140);
          const bg = `hsl(${hue}deg 70% 38%)`;
          const sourceTag = cell.source === "no_defense" ? "" : " *";
          return `<td style="background:${bg}" title="ASR=${fmtPct(
            cell.asr
          )}, source=${esc(cell.source)}, runs=${cell.run_count}">${fmtPct(cell.asr, 1)}${sourceTag}</td>`;
        })
        .join("");
      return `<tr><td class="first-col">${esc(row.model)}</td>${cells}</tr>`;
    })
    .join("")}</tbody>`;

  table.innerHTML = `${head}${body}`;
}

function renderAttackDefenseMatrix(matrix) {
  const table = document.getElementById("attackDefenseMatrixTable");
  const legend = document.getElementById("attackDefenseMatrixLegend");
  if (!table || !legend || !matrix) return;

  const min = matrix?.range?.min_asr ?? 0;
  const max = matrix?.range?.max_asr ?? 1;
  const span = Math.max(max - min, 1e-6);
  setMatrixLegend(legend);

  const defenses = Array.isArray(matrix.defenses) ? matrix.defenses : [];
  const rows = Array.isArray(matrix.rows) ? matrix.rows : [];
  if (!defenses.length || !rows.length) {
    table.innerHTML = '<tbody><tr><td class="empty">No matrix data</td></tr></tbody>';
    return;
  }

  const head = `<thead><tr><th>Attack</th>${defenses.map((d) => `<th>${esc(d)}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${rows
    .map((row) => {
      const cells = (row.cells || [])
        .map((cell) => {
          if (cell.asr === null || cell.asr === undefined) {
            return '<td title="No run">-</td>';
          }
          const ratio = (cell.asr - min) / span;
          const hue = Math.max(0, 140 - ratio * 140);
          const bg = `hsl(${hue}deg 70% 38%)`;
          return `<td style="background:${bg}" title="ASR=${fmtPct(cell.asr)}, runs=${cell.run_count}, models=${
            cell.model_coverage ?? 0
          }, judged=${cell.judged_samples ?? 0}">${fmtPct(cell.asr, 1)}</td>`;
        })
        .join("");
      return `<tr><td class="first-col">${esc(row.attack)}</td>${cells}</tr>`;
    })
    .join("")}</tbody>`;

  table.innerHTML = `${head}${body}`;
}

function fillSelect(selectId, values) {
  const select = document.getElementById(selectId);
  select.innerHTML = `<option value="">All</option>${values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("")}`;
}

function pickFirst(sample, keys) {
  for (const key of keys) {
    if (!(key in sample)) continue;
    const value = sample[key];
    if (value !== null && value !== undefined && value !== "") return value;
  }
  for (const key of keys) {
    if (key in sample) return sample[key];
  }
  return null;
}

function shortText(value, maxLen = 140) {
  if (value === null || value === undefined) return "-";
  const text = String(value).replace(/\s+/g, " ").trim();
  if (!text) return "-";
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 3)}...`;
}

function judgerShortName(name) {
  const raw = String(name || "");
  const known = {
    harmbench_judger: "harmbench",
    gpt_judger_harmful_binary: "gpt-hb",
    rejection_prefix_judger: "reject-prefix",
  };
  if (known[raw]) return known[raw];
  return raw.replace(/_judger$/i, "").replace(/_/g, "-");
}

function formatIndividualCompact(individual) {
  if (!individual || typeof individual !== "object") return "-";
  const parts = Object.entries(individual).map(([k, v]) => `${judgerShortName(k)}:${fmtPct(normalizeScore(v), 0)}`);
  return parts.join(" | ");
}

function simpleValueSummary(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "string") return shortText(value, 80);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    const head = value.slice(0, 3).map((v) => simpleValueSummary(v)).join(", ");
    return value.length > 3 ? `[${value.length}] ${head}, ...` : `[${value.length}] ${head}`;
  }
  if (typeof value === "object") {
    const keys = Object.keys(value);
    return `{${keys.length} keys}`;
  }
  return String(value);
}

function extractSampleView(sample) {
  const cleanPrompt = pickFirst(sample, ["clean_prompt", "prompt_on_clean", "original_prompt"]);
  const attackedPrompt = pickFirst(sample, ["attacked_prompt", "prompt_on_attacked", "jailbreak_prompt"]);
  const defendedPrompt = pickFirst(sample, ["attacked_prompt_under_defense", "defended_prompt"]);

  const responseClean = pickFirst(sample, ["llm_response_on_clean"]);
  const responseAttacked = pickFirst(sample, ["llm_response_on_attacked"]);
  const responseCleanUnderDefense = pickFirst(sample, ["llm_response_on_clean_under_defense"]);
  const responseAttackedUnderDefense = pickFirst(sample, ["llm_response_on_attacked_under_defense"]);
  const responsePrimary = pickFirst(sample, ["llm_response", "response"]);

  const judgerOverall = pickFirst(sample, [
    "judger_result_on_attack_under_defense",
    "judger_result_on_attack",
    "judger_result",
    "judger_result_on_clean_under_defense",
    "judger_result_on_clean",
  ]);

  return {
    sampleIndex: sample.sample_index,
    status: sample.status,
    targetLlmType: sample.target_llm_type,
    cleanPrompt,
    attackedPrompt,
    defendedPrompt,
    responseClean,
    responseAttacked,
    responseCleanUnderDefense,
    responseAttackedUnderDefense,
    responsePrimary,
    responseType: sample.response_type,
    judgerOverall,
    judgerOnClean: sample.judger_result_on_clean,
    judgerOnAttack: sample.judger_result_on_attack,
    judgerOnCleanUnderDefense: sample.judger_result_on_clean_under_defense,
    judgerOnAttackUnderDefense: sample.judger_result_on_attack_under_defense,
    judgerIndividual: sample.judger_individual_results,
    judgerContext: sample.judger_context,
    attackQueries: sample.attack_query_count,
    attackRuntime: sample.attack_runtime,
    llmResponseTime: pickFirst(sample, [
      "llm_response_time_under_defense",
      "llm_response_time_attacked",
      "llm_response_time",
      "llm_response_time_clean",
    ]),
    defenseFallback: sample.defense_fallback,
    defenseFallbackToModel: sample.defense_fallback_to_model,
    judgerFallback: sample.judger_fallback,
    statusReason: sample.status_reason,
    sampleError: sample.error || sample.judger_error_reason || null,
  };
}

function renderModalSummaryCards(container, payload, sampleViews) {
  if (!container) return;

  const sampleCount = sampleViews.length;
  const judgedScores = sampleViews.map((s) => normalizeScore(s.judgerOverall)).filter((v) => v !== null);
  const avgOverall = judgedScores.length ? judgedScores.reduce((a, b) => a + b, 0) / judgedScores.length : null;
  const completion =
    toNum(payload?.total_samples) && Number(payload.total_samples) > 0
      ? Number(payload.successful_samples || 0) / Number(payload.total_samples)
      : null;
  const avgQueries =
    sampleViews.filter((s) => toNum(s.attackQueries) !== null).length > 0
      ? sampleViews.filter((s) => toNum(s.attackQueries) !== null).reduce((a, s) => a + Number(s.attackQueries), 0) /
        sampleViews.filter((s) => toNum(s.attackQueries) !== null).length
      : null;
  const avgAttackRuntime =
    sampleViews.filter((s) => toNum(s.attackRuntime) !== null).length > 0
      ? sampleViews.filter((s) => toNum(s.attackRuntime) !== null).reduce((a, s) => a + Number(s.attackRuntime), 0) /
        sampleViews.filter((s) => toNum(s.attackRuntime) !== null).length
      : null;
  const avgLatency =
    sampleViews.filter((s) => toNum(s.llmResponseTime) !== null).length > 0
      ? sampleViews.filter((s) => toNum(s.llmResponseTime) !== null).reduce((a, s) => a + Number(s.llmResponseTime), 0) /
        sampleViews.filter((s) => toNum(s.llmResponseTime) !== null).length
      : null;
  const defenseFallbackCount = sampleViews.filter((s) => s.defenseFallback === true).length;
  const judgerFallbackCount = sampleViews.filter((s) => s.judgerFallback === true).length;
  const errorCount = sampleViews.filter((s) => !!s.sampleError).length;

  const cards = [
    { label: "Samples", value: sampleCount },
    { label: "Overall ASR", value: fmtPct(avgOverall) },
    { label: "Completion", value: fmtPct(completion) },
    { label: "Status", value: payload?.status || "-" },
    { label: "Success / Total", value: `${payload?.successful_samples ?? "-"}/${payload?.total_samples ?? sampleCount}` },
    { label: "Attack Success Rate", value: fmtPct(toNum(payload?.attack_success_rate)) },
    { label: "Clean Safe Rate", value: fmtPct(toNum(payload?.clean_safe_rate)) },
    { label: "Avg Attack Queries", value: avgQueries === null ? "N/A" : avgQueries.toFixed(2) },
    { label: "Avg Attack Runtime (s)", value: avgAttackRuntime === null ? "N/A" : avgAttackRuntime.toFixed(2) },
    { label: "Avg LLM Latency (s)", value: avgLatency === null ? "N/A" : avgLatency.toFixed(2) },
    { label: "Defense Fallback", value: `${defenseFallbackCount}/${sampleCount}` },
    { label: "Judger Fallback", value: `${judgerFallbackCount}/${sampleCount}` },
    { label: "Sample Errors", value: `${errorCount}/${sampleCount}` },
  ];

  container.innerHTML = cards
    .map(
      (c) =>
        `<article class="card"><span class="label">${esc(c.label)}</span><span class="value">${esc(
          c.value
        )}</span></article>`
    )
    .join("");
}

function summarizeConfigRows(samples, fieldName, maxRows = 6) {
  const objects = samples
    .map((s) => (s && typeof s === "object" ? s[fieldName] : null))
    .filter((v) => v && typeof v === "object" && !Array.isArray(v));
  if (!objects.length) return [];

  const keys = Array.from(new Set(objects.flatMap((o) => Object.keys(o)))).sort();
  const rows = [];
  for (const key of keys) {
    const values = objects
      .map((o) => o[key])
      .filter((v) => v !== undefined)
      .map((v) => simpleValueSummary(v));
    if (!values.length) continue;
    const uniq = Array.from(new Set(values));
    const summary =
      uniq.length === 1 ? uniq[0] : `mixed (${uniq.length}): ${uniq.slice(0, 3).join(", ")}${uniq.length > 3 ? ", ..." : ""}`;
    rows.push([`${fieldName}.${key}`, summary]);
    if (rows.length >= maxRows) break;
  }
  return rows;
}

function renderModalMetaTable(table, payload, sampleViews, samples) {
  if (!table) return;

  const judgerText = Array.isArray(payload?.judger_name)
    ? payload.judger_name.join(", ")
    : payload?.judger_name || "-";
  const cfg = payload?.config && typeof payload.config === "object" ? payload.config : {};
  const responseTypes = uniqueSorted(sampleViews.map((s) => s.responseType).filter(Boolean));
  const statusReasons = uniqueSorted(sampleViews.map((s) => s.statusReason).filter(Boolean));
  const dependencies = Array.isArray(payload?.dependencies) ? payload.dependencies : [];

  const rows = [
    ["Run ID", payload?.experiment_id || "-"],
    ["Experiment Name", payload?.experiment_name || "-"],
    ["Model", payload?.target_llm_name || "-"],
    ["Model Type", sampleViews.find((s) => s.targetLlmType)?.targetLlmType || "-"],
    ["Attack", payload?.attack_method || "-"],
    ["Defense", payload?.defense_method || "-"],
    ["Dataset", payload?.dataset_name || "-"],
    ["Judger(s)", judgerText],
    ["Execution Time (s)", payload?.execution_time ?? "-"],
    ["Run Success Count", payload?.success_count ?? payload?.successful_samples ?? "-"],
    ["Run Failed Count", payload?.failed_count ?? payload?.failed_samples ?? "-"],
    ["Run Total Samples", payload?.total_samples ?? samples.length ?? "-"],
    ["Sample Limit", payload?.sample_limit ?? "-"],
    ["Seed", cfg.seed ?? "-"],
    ["Dependency Count", dependencies.length],
    ["Response Types", responseTypes.length ? responseTypes.join(", ") : "-"],
    ["Status Reasons", statusReasons.length ? statusReasons.join(", ") : "-"],
    ["Created Time", payload?.created_time || "-"],
    ["Last Updated", payload?.last_updated || "-"],
    ["config.model", cfg.model ?? "-"],
    ["config.attack", cfg.attack ?? "-"],
    ["config.defense", cfg.defense ?? "-"],
    ["config.dataset", cfg.dataset ?? "-"],
    ["config.judger", Array.isArray(cfg.judger) ? cfg.judger.join(", ") : cfg.judger ?? "-"],
    ...summarizeConfigRows(samples, "attack_config", 5),
    ...summarizeConfigRows(samples, "defense_config", 5),
  ];

  table.innerHTML = `<tbody>${rows
    .map(([k, v]) => `<tr><th>${esc(k)}</th><td>${v === null || v === undefined || v === "" ? "-" : esc(String(v))}</td></tr>`)
    .join("")}</tbody>`;
}

function renderModalMergedSamplesHeader(container) {
  if (!container) return;
  container.innerHTML = `<div class="sample-merged-grid sample-merged-grid--header">
    <span>Sample</span>
    <span>Status</span>
    <span>Clean Prompt</span>
    <span>Attacked Prompt</span>
    <span>Evaluated Response</span>
    <span>Overall Judger</span>
    <span>Individual Judgers</span>
  </div>`;
}

function renderSampleItem(item) {
  const overallScore = normalizeScore(item.judgerOverall);
  const scoreClass = asrClass(overallScore);
  const statusText = item.status || "unknown";
  const sampleId = item.sampleIndex ?? "?";
  const evaluatedResponse =
    item.responseAttackedUnderDefense || item.responseAttacked || item.responsePrimary || item.responseCleanUnderDefense || item.responseClean;
  const individualText =
    item.judgerIndividual && typeof item.judgerIndividual === "object"
      ? Object.entries(item.judgerIndividual)
          .map(([k, v]) => `${k}=${fmtPct(normalizeScore(v))}`)
          .join("; ")
      : "-";
  const individualCompact = formatIndividualCompact(item.judgerIndividual);

  const badges = [
    `<span class="sample-badge">sample=${esc(sampleId)}</span>`,
    `<span class="sample-badge">status=${esc(statusText)}</span>`,
    `<span class="sample-badge ${scoreClass}">overall=${esc(fmtPct(overallScore))}</span>`,
    item.attackQueries !== null && item.attackQueries !== undefined
      ? `<span class="sample-badge">queries=${esc(item.attackQueries)}</span>`
      : "",
    item.attackRuntime !== null && item.attackRuntime !== undefined
      ? `<span class="sample-badge">attack_rt=${esc(Number(item.attackRuntime).toFixed(2))}s</span>`
      : "",
  ]
    .filter(Boolean)
    .join("");

  const fields = [
    ["Prompt (Clean / pre-attack)", item.cleanPrompt],
    ["Prompt (Attacked / post-attack)", item.attackedPrompt],
    ["Prompt (Under Defense)", item.defendedPrompt],
    ["Response (Clean)", item.responseClean],
    ["Response (Attacked, no defense)", item.responseAttacked],
    ["Response (Clean + defense)", item.responseCleanUnderDefense],
    ["Response (Attacked + defense)", item.responseAttackedUnderDefense],
    ["Response (Primary Evaluated)", item.responsePrimary],
    ["Response Type", item.responseType],
    ["Judger (clean)", item.judgerOnClean],
    ["Judger (attack)", item.judgerOnAttack],
    ["Judger (clean + defense)", item.judgerOnCleanUnderDefense],
    ["Judger (attack + defense)", item.judgerOnAttackUnderDefense],
    [
      "Judger Individual Results",
      item.judgerIndividual !== null && item.judgerIndividual !== undefined
        ? JSON.stringify(item.judgerIndividual, null, 2)
        : null,
    ],
    [
      "Judger Context",
      item.judgerContext !== null && item.judgerContext !== undefined ? JSON.stringify(item.judgerContext, null, 2) : null,
    ],
  ]
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(
      ([label, value]) =>
        `<article class="sample-item"><span class="label">${esc(label)}</span><pre>${esc(
          typeof value === "string" ? value : JSON.stringify(value, null, 2)
        )}</pre></article>`
    )
    .join("");

  return `<details class="sample-card sample-card-merged" id="sample-${esc(sampleId)}">
    <summary class="sample-merged-summary">
      <div class="sample-merged-grid">
        <span class="mono">${esc(sampleId)}</span>
        <span>${esc(statusText)}</span>
        <span title="${esc(item.cleanPrompt || "-")}">${esc(shortText(item.cleanPrompt, 180))}</span>
        <span title="${esc(item.attackedPrompt || "-")}">${esc(shortText(item.attackedPrompt, 180))}</span>
        <span title="${esc(evaluatedResponse || "-")}">${esc(shortText(evaluatedResponse, 180))}</span>
        <span class="${scoreClass}">${esc(fmtPct(overallScore))}</span>
        <span title="${esc(individualText)}">${esc(shortText(individualCompact, 100))}</span>
      </div>
    </summary>
    <div class="sample-body">
      <div class="sample-badges">${badges}</div>
      <div class="sample-grid">${fields}</div>
    </div>
  </details>`;
}

function runPathCandidates(path) {
  if (!path) return [];
  const clean = String(path).replace(/^[./]+/, "").replace(/^\/+/, "");
  if (!clean) return [];
  const filename = clean.includes("/") ? clean.split("/").pop() : clean;

  const raw = [
    clean,
    clean.startsWith("leaderboard_site/") ? clean.replace(/^leaderboard_site\//, "") : null,
    filename ? `data/runs/${filename}` : null,
    clean.startsWith("data/") ? `./${clean}` : null,
    clean.startsWith("experiments/") ? `../${clean}` : null,
    clean.startsWith("placeholders/") ? `../experiments/${clean}` : null,
    clean.startsWith("experiments/placeholders/") ? clean.replace(/^experiments\//, "") : null,
    clean.startsWith("placeholders/") ? `experiments/${clean}` : null,
    !clean.startsWith("experiments/") ? `experiments/${clean}` : null,
    !clean.startsWith("../") ? `../${clean}` : null,
  ].filter(Boolean);

  return Array.from(new Set(raw));
}

function runPathCandidatesWithUrl(path) {
  const baseUrl = new URL(".", window.location.href);
  return runPathCandidates(path).map((p) => ({
    path: p,
    url: new URL(p, baseUrl).toString(),
  }));
}

function setupRunBreakdownModal() {
  const modal = document.getElementById("runBreakdownModal");
  const closeBtn = document.getElementById("runModalClose");
  const titleEl = document.getElementById("runModalTitle");
  const metaEl = document.getElementById("runModalMeta");
  const stateEl = document.getElementById("runModalState");
  const summaryCardsEl = document.getElementById("runModalSummaryCards");
  const metaTableEl = document.getElementById("runModalMetaTable");
  const samplesHeaderEl = document.getElementById("runModalSamplesHeader");
  const samplesEl = document.getElementById("runModalSamples");
  if (
    !modal ||
    !closeBtn ||
    !titleEl ||
    !metaEl ||
    !stateEl ||
    !summaryCardsEl ||
    !metaTableEl ||
    !samplesHeaderEl ||
    !samplesEl
  ) {
    return null;
  }

  function close() {
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
  }

  function open() {
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
  }

  modal.addEventListener("click", (evt) => {
    const target = evt.target;
    if (target && target instanceof Element && target.getAttribute("data-close-modal") === "true") {
      close();
    }
  });
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape" && !modal.classList.contains("hidden")) {
      close();
    }
  });

  return {
    showLoading(row) {
      open();
      titleEl.textContent = `Run Breakdown: ${row?.run_id || "-"}`;
      metaEl.innerHTML = `Model: <code>${esc(row?.model || "-")}</code> | Attack: <code>${esc(
        row?.attack || "-"
      )}</code> | Defense: <code>${esc(row?.defense || "-")}</code>`;
      stateEl.textContent = "Loading run payload and sample-level breakdown ...";
      summaryCardsEl.innerHTML = "";
      metaTableEl.innerHTML = '<tbody><tr><td class="empty">Loading run summary ...</td></tr></tbody>';
      samplesHeaderEl.innerHTML = "";
      samplesEl.innerHTML = "";
    },
    showData(row, payload, sampleViews) {
      open();
      titleEl.textContent = `Run Breakdown: ${row?.run_id || "-"}`;
      metaEl.innerHTML = `Model: <code>${esc(row?.model || "-")}</code> | Attack: <code>${esc(
        row?.attack || "-"
      )}</code> | Defense: <code>${esc(row?.defense || "-")}</code> | Samples: <code>${sampleViews.length}</code>`;
      stateEl.textContent = "";
      const samples = Array.isArray(payload?.sample_results) ? payload.sample_results : [];
      renderModalSummaryCards(summaryCardsEl, payload || row || {}, sampleViews || []);
      renderModalMetaTable(metaTableEl, payload || row || {}, sampleViews || [], samples);
      renderModalMergedSamplesHeader(samplesHeaderEl);
      if (!sampleViews.length) {
        samplesEl.innerHTML = `<div class="empty">No sample-level records in this run file.</div>`;
      } else {
        samplesEl.innerHTML = sampleViews.map(renderSampleItem).join("");
      }
    },
    showError(row, err) {
      open();
      titleEl.textContent = `Run Breakdown: ${row?.run_id || "-"}`;
      metaEl.innerHTML = `Model: <code>${esc(row?.model || "-")}</code> | Attack: <code>${esc(
        row?.attack || "-"
      )}</code> | Defense: <code>${esc(row?.defense || "-")}</code>`;
      stateEl.textContent = `Failed to load run payload: ${err?.message || err}`;
      summaryCardsEl.innerHTML = `<div class="empty">Failed to build run summary.</div>`;
      metaTableEl.innerHTML = `<tbody><tr><td class="empty">Could not load run metadata.</td></tr></tbody>`;
      samplesHeaderEl.innerHTML = "";
      samplesEl.innerHTML = `<div class="empty">Could not load sample-level data for this run.</div>`;
    },
    close,
  };
}

function buildRunScope(filteredRuns) {
  const modelSet = uniqueSorted(filteredRuns.map((r) => r.model));
  const attackSet = uniqueSorted(filteredRuns.map((r) => r.attack));
  const defenseSet = uniqueSorted(filteredRuns.map((r) => r.defense));

  const judgedSamples = filteredRuns.reduce((acc, r) => acc + (r.judged_samples || 0), 0);
  const totalSamples = filteredRuns.reduce((acc, r) => acc + (r.total_samples || 0), 0);
  const successfulSamples = filteredRuns.reduce((acc, r) => acc + (r.successful_samples || 0), 0);

  const weightedAsr = weightedMean(filteredRuns.map((r) => [r.asr, Math.max(r.judged_samples || 0, 1)]));
  const completion = totalSamples > 0 ? successfulSamples / totalSamples : null;

  return {
    runCount: filteredRuns.length,
    modelSet,
    attackSet,
    defenseSet,
    judgedSamples,
    totalSamples,
    completion,
    weightedAsr,
  };
}

function renderRunScope(scope) {
  const cards = [
    { label: "Filtered Runs", value: scope.runCount },
    { label: "Weighted ASR", value: fmtPct(scope.weightedAsr) },
    { label: "Judged Samples", value: scope.judgedSamples.toLocaleString() },
    { label: "Sample Completion", value: fmtPct(scope.completion) },
    { label: "Models", value: scope.modelSet.length },
    { label: "Attacks", value: scope.attackSet.length },
    { label: "Defenses", value: scope.defenseSet.length },
  ];

  const cardsEl = document.getElementById("runScopeCards");
  cardsEl.innerHTML = cards
    .map(
      (c) =>
        `<article class="card"><span class="label">${esc(c.label)}</span><span class="value">${esc(
          c.value
        )}</span></article>`
    )
    .join("");

  const breakdown = document.getElementById("runScopeBreakdown");
  const top = (arr) => (arr.length <= 8 ? arr.join(", ") : `${arr.slice(0, 8).join(", ")} ... (+${arr.length - 8})`);
  breakdown.innerHTML = `Models: <code>${esc(top(scope.modelSet) || "-")}</code> | Attacks: <code>${esc(
    top(scope.attackSet) || "-"
  )}</code> | Defenses: <code>${esc(top(scope.defenseSet) || "-")}</code>`;
}

async function loadRunPayload(run) {
  if (runPayloadCache.has(run.path)) {
    return runPayloadCache.get(run.path);
  }

  const candidates = runPathCandidatesWithUrl(run.path);
  if (!candidates.length) throw new Error(`Missing run path for ${run.run_id}`);

  let lastErr = null;
  for (const c of candidates) {
    try {
      const resp = await fetch(c.url);
      if (!resp.ok) {
        lastErr = `HTTP ${resp.status} @ ${c.url}`;
        continue;
      }
      const payload = await resp.json();
      runPayloadCache.set(run.path, payload);
      if (run.path !== c.path) {
        run.path = c.path;
      }
      return payload;
    } catch (err) {
      lastErr = `${err?.message || err} @ ${c.url}`;
    }
  }

  throw new Error(
    `Failed to fetch run payload. Last error: ${lastErr || "unknown"}. Tried: ${candidates
      .map((x) => x.path)
      .join(", ")}`
  );
}

function focusRunsByEntity(entityType, value) {
  if (!runsViewApi) return;
  runsViewApi.focusEntity(entityType, value);
}

function renderRunsSection(dataset) {
  fillSelect("filterModel", dataset.filters.models);
  fillSelect("filterAttack", dataset.filters.attacks);
  fillSelect("filterDefense", dataset.filters.defenses);
  fillSelect("filterDataset", dataset.filters.datasets);
  fillSelect("filterJudger", dataset.filters.judgers);

  const controls = [
    "filterModel",
    "filterAttack",
    "filterDefense",
    "filterDataset",
    "filterJudger",
    "filterSearch",
  ].map((id) => document.getElementById(id));

  let activeRunToken = 0;

  function currentFilters() {
    return {
      model: document.getElementById("filterModel").value,
      attack: document.getElementById("filterAttack").value,
      defense: document.getElementById("filterDefense").value,
      datasetName: document.getElementById("filterDataset").value,
      judger: document.getElementById("filterJudger").value,
      search: document.getElementById("filterSearch").value.trim().toLowerCase(),
    };
  }

  function filterRuns() {
    const f = currentFilters();
    return dataset.runs
      .filter((r) => !f.model || r.model === f.model)
      .filter((r) => !f.attack || r.attack === f.attack)
      .filter((r) => !f.defense || r.defense === f.defense)
      .filter((r) => !f.datasetName || r.dataset === f.datasetName)
      .filter((r) => !f.judger || r.judger === f.judger)
      .filter((r) => {
        if (!f.search) return true;
        return (
          r.model.toLowerCase().includes(f.search) ||
          r.attack.toLowerCase().includes(f.search) ||
          r.defense.toLowerCase().includes(f.search) ||
          r.dataset.toLowerCase().includes(f.search) ||
          r.judger.toLowerCase().includes(f.search)
        );
      })
      .sort((a, b) => {
        const asrA = a.asr ?? -1;
        const asrB = b.asr ?? -1;
        if (asrA !== asrB) return asrB - asrA;
        return (b.created_time || 0) - (a.created_time || 0);
      });
  }

  async function selectRun(row) {
    const token = ++activeRunToken;
    if (runBreakdownModalApi) {
      runBreakdownModalApi.showLoading(row);
    }

    try {
      const payload = await loadRunPayload(row);
      if (token !== activeRunToken) return;

      const samples = Array.isArray(payload.sample_results) ? payload.sample_results : [];
      const sampleViews = samples.map(extractSampleView);
      if (runBreakdownModalApi) {
        runBreakdownModalApi.showData(row, payload, sampleViews);
      }
    } catch (err) {
      if (token !== activeRunToken) return;
      if (runBreakdownModalApi) {
        runBreakdownModalApi.showError(row, err);
      }
    }
  }

  function paint() {
    const filtered = filterRuns();
    const scope = buildRunScope(filtered);
    renderRunScope(scope);

    renderTable(
      "runsTable",
      [
        { label: "Model", key: "model" },
        { label: "Attack", key: "attack" },
        { label: "Defense", key: "defense" },
        { label: "ASR", html: true, render: (r) => `<span class="${asrClass(r.asr)}">${fmtPct(r.asr)}</span>` },
        { label: "Judged", key: "judged_samples", className: "mono" },
        {
          label: "Sample Completion",
          html: true,
          render: (r) => `<span class="${asrClass(1 - (r.sample_success_rate ?? 0))}">${fmtPct(r.sample_success_rate)}</span>`,
        },
        { label: "Total", key: "total_samples", className: "mono" },
        { label: "Dataset", key: "dataset" },
        { label: "Judger", key: "judger" },
        { label: "Updated", key: "updated_at", className: "mono" },
      ],
      filtered,
      (row) => {
        void selectRun(row);
      }
    );

    if (!filtered.length) return;
  }

  controls.forEach((el) => el.addEventListener("input", paint));
  paint();

  runsViewApi = {
    focusEntity(entityType, value) {
      const modelEl = document.getElementById("filterModel");
      const attackEl = document.getElementById("filterAttack");
      const defenseEl = document.getElementById("filterDefense");
      const searchEl = document.getElementById("filterSearch");

      if (entityType === "model") {
        modelEl.value = value || "";
        attackEl.value = "";
        defenseEl.value = "";
      } else if (entityType === "attack") {
        modelEl.value = "";
        attackEl.value = value || "";
        defenseEl.value = "";
      } else if (entityType === "defense") {
        modelEl.value = "";
        attackEl.value = "";
        defenseEl.value = value || "";
      }

      searchEl.value = "";
      paint();

      const section = document.getElementById("runsSection");
      if (section) {
        section.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    },
  };
}

function renderLeaderboards(data) {
  const modelRows = sortByRobustnessDesc(data.leaderboards.models || []);
  const apiModels = withSubRank(modelRows.filter((row) => inferModelType(row.model) === "api"));
  const localModels = withSubRank(modelRows.filter((row) => inferModelType(row.model) === "local"));

  const modelColumns = [
    { label: "Rank", key: "sub_rank", className: "mono" },
    { label: "Model", key: "model" },
    {
      label: "Robustness",
      html: true,
      render: (r) => {
        const hint = `Robustness = 1 - Avg ASR. Avg ASR=${fmtPct(r.avg_asr)}; judged_samples=${r.judged_samples || 0}.`;
        return metricHint(fmtPct(r.robustness_score), hint, asrClass(r.avg_asr));
      },
    },
    {
      label: "No-Attack Robustness",
      html: true,
      render: (r) => {
        const robust =
          r.no_attack_robustness !== null && r.no_attack_robustness !== undefined
            ? r.no_attack_robustness
            : r.no_attack_unsafe_rate !== null && r.no_attack_unsafe_rate !== undefined
              ? 1 - r.no_attack_unsafe_rate
              : null;
        const hint = `No-attack robustness = 1 - no-attack ASR. baseline_runs=${r.no_attack_run_count || 0}; baseline_judged_samples=${
          r.no_attack_judged_samples || 0
        }.`;
        return metricHint(fmtPct(robust), hint, asrClass(1 - (robust ?? 1)));
      },
    },
    {
      label: "Attacks",
      html: true,
      render: (r) => metricHint(String(r.attack_coverage), `Unique attacks evaluated for this model: ${r.attack_coverage}.`, "mono"),
    },
    {
      label: "Defenses",
      html: true,
      render: (r) => metricHint(String(r.defense_coverage), `Unique defenses evaluated for this model: ${r.defense_coverage}.`, "mono"),
    },
    {
      label: "#Tests",
      html: true,
      render: (r) => metricHint(String(r.run_count), `Completed attack runs for ranking: ${r.run_count}.`, "mono"),
    },
    {
      label: "Test Completion",
      html: true,
      render: (r) =>
        metricHint(
          fmtPct(r.sample_completion_rate),
          `Weighted sample completion across runs. successful/total samples aggregated by run.`
        ),
    },
  ];

  renderTable("modelApiTable", modelColumns, apiModels, (row) => {
    focusRunsByEntity("model", row.model);
  });

  renderTable("modelLocalTable", modelColumns, localModels, (row) => {
    focusRunsByEntity("model", row.model);
  });

  renderTable(
    "defenseTable",
    [
      { label: "Rank", key: "rank", className: "mono" },
      { label: "Defense", key: "defense" },
      {
        label: "Avg ASR",
        html: true,
        render: (r) =>
          metricHint(
            fmtPct(r.avg_asr),
            `Defense-side ASR over matched pairs. matched_pairs=${r.matched_pair_count}; judged_samples=${r.judged_samples}.`,
            asrClass(r.avg_asr)
          ),
      },
      {
        label: "ASR Gain vs no_defense",
        html: true,
        render: (r) => {
          const cls = r.asr_gain_vs_no_defense === null ? "" : r.asr_gain_vs_no_defense >= 0 ? "good" : "bad";
          return metricHint(
            fmtPct(r.asr_gain_vs_no_defense),
            `ASR gain = baseline(no_defense) - defense ASR. Positive is better.`,
            cls
          );
        },
      },
      {
        label: "Matched Pairs",
        html: true,
        render: (r) => metricHint(String(r.matched_pair_count), `Matched (model, attack, dataset, judger) pairs: ${r.matched_pair_count}.`, "mono"),
      },
      {
        label: "Model Coverage",
        html: true,
        render: (r) => metricHint(String(r.model_coverage), `Unique models covered by this defense: ${r.model_coverage}.`, "mono"),
      },
      {
        label: "Attack Coverage",
        html: true,
        render: (r) => metricHint(String(r.attack_coverage), `Unique attacks covered by this defense: ${r.attack_coverage}.`, "mono"),
      },
      {
        label: "Judged Samples",
        html: true,
        render: (r) => metricHint(String(r.judged_samples), `Total judged samples contributing to this defense row.`, "mono"),
      },
    ],
    data.leaderboards.defenses,
    (row) => {
      focusRunsByEntity("defense", row.defense);
    }
  );

  renderTable(
    "attackTable",
    [
      { label: "Rank", key: "rank", className: "mono" },
      { label: "Attack", key: "attack" },
      {
        label: "Avg ASR",
        html: true,
        render: (r) =>
          metricHint(
            fmtPct(r.avg_asr),
            `Attack-side ASR over runs. Higher means stronger jailbreak success. judged_samples=${r.judged_samples}.`,
            asrClass(r.avg_asr)
          ),
      },
      {
        label: "Runs",
        html: true,
        render: (r) => metricHint(String(r.run_count), `Completed runs using this attack: ${r.run_count}.`, "mono"),
      },
      {
        label: "Model Coverage",
        html: true,
        render: (r) => metricHint(String(r.model_coverage), `Unique models attacked: ${r.model_coverage}.`, "mono"),
      },
      {
        label: "Defense Coverage",
        html: true,
        render: (r) => metricHint(String(r.defense_coverage), `Unique defenses paired with this attack: ${r.defense_coverage}.`, "mono"),
      },
      {
        label: "Judged Samples",
        html: true,
        render: (r) => metricHint(String(r.judged_samples), `Total judged samples contributing to this attack row.`, "mono"),
      },
    ],
    data.leaderboards.attacks,
    (row) => {
      focusRunsByEntity("attack", row.attack);
    }
  );
}

async function init() {
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) {
      throw new Error(`Failed to load ${DATA_URL}: ${response.status}`);
    }
    const dataset = await response.json();

    const sortedModelRows = sortByRobustnessDesc(dataset.leaderboards.models || []);
    setMeta(dataset.meta);
    runBreakdownModalApi = setupRunBreakdownModal();
    renderOverview(
      dataset.overview,
      sortedModelRows,
      dataset.leaderboards.defenses,
      dataset.leaderboards.attacks
    );
    renderRunsSection(dataset);
    renderLeaderboards(dataset);
    setModelJudgerNote(dataset?.filters?.judgers || []);
    renderModelAttackMatrix(dataset.model_attack_matrix || dataset.matrix);
    renderAttackDefenseMatrix(dataset.attack_defense_matrix);
  } catch (err) {
    document.body.innerHTML = `<main class="page"><section class="section"><h2>Failed to load leaderboard</h2><pre>${esc(
      err?.stack || err
    )}</pre></section></main>`;
  }
}

init();
