const DATA_URL = "./data/leaderboard.json?v=20260612-paper-clean-6";
const TRAFFIC_REFRESH_MS = 30_000;
const DEFAULT_JUDGER_KEY = "gpt_judger_harmful_binary";
const JUDGER_LABELS = {
  harmbench_judger: "HarmBench",
  gpt_judger_harmful_binary: "HarmBench-style Judger (GPT-4.1 mini)",
  rejection_prefix_judger: "Prefix Judger",
};

let runsViewApi = null;
let runBreakdownModalApi = null;
const runPayloadCache = new Map();
let trafficIntervalId = null;

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

function fmtPp(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  const scaled = value * 100;
  const sign = scaled > 0 ? "+" : "";
  return `${sign}${scaled.toFixed(digits)} pp`;
}

function fmtFloat(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return Number(value).toFixed(digits);
}

function fmtSec(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return `${Number(value).toFixed(1)}s`;
}

function asrClass(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  if (value <= 0.2) return "good";
  if (value <= 0.45) return "warn";
  return "bad";
}

function deltaClass(value, positiveIsGood = true) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  if (value === 0) return "warn";
  const good = positiveIsGood ? value > 0 : value < 0;
  return good ? "good" : "bad";
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

function fmtInt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toLocaleString();
}

function judgerLabel(key) {
  return JUDGER_LABELS[key] || String(key || "N/A");
}

function setTrafficStatus(message, kind = "info") {
  const node = document.getElementById("trafficStatus");
  if (!node) return;
  node.classList.remove("traffic-status-error");
  if (kind === "error") node.classList.add("traffic-status-error");
  node.textContent = message;
}

function setTrafficValue(id, value) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = value;
}

function resolveGoatcounterBase(rawCode) {
  const raw = String(rawCode || "").trim();
  if (!raw) return null;

  const withoutCount = raw.replace(/\/count\/?$/i, "").replace(/\/+$/, "");
  if (/^https?:\/\//i.test(withoutCount)) {
    return withoutCount;
  }
  if (/\.goatcounter\.com$/i.test(withoutCount)) {
    return `https://${withoutCount}`;
  }
  return `https://${withoutCount}.goatcounter.com`;
}

function getGoatcounterBase() {
  const bodyCode = document.body?.dataset?.goatcounterCode;
  const windowCode = window?.LEADERBOARD_ANALYTICS?.goatcounterCode;
  return resolveGoatcounterBase(windowCode || bodyCode || "");
}

function ensureGoatcounterScript(base) {
  return new Promise((resolve, reject) => {
    const endpoint = `${base}/count`;
    const existing = document.querySelector('script[src*="gc.zgo.at/count.js"]');
    if (existing) {
      if (!existing.dataset.goatcounter) {
        existing.dataset.goatcounter = endpoint;
      }
      resolve();
      return;
    }

    const script = document.createElement("script");
    script.async = true;
    script.src = "https://gc.zgo.at/count.js";
    script.dataset.goatcounter = endpoint;
    script.addEventListener("load", () => resolve());
    script.addEventListener("error", () => reject(new Error("Failed to load goatcounter count.js")));
    document.head.appendChild(script);
  });
}

function currentGoatPath() {
  try {
    const goatData = window?.goatcounter?.get_data?.();
    if (goatData && goatData.p) return String(goatData.p);
  } catch (err) {
    // Ignore and use pathname fallback.
  }
  return window.location.pathname || "/";
}

async function fetchGoatCount(base, path, params = {}) {
  const url = new URL(`${base}/counter/${encodeURIComponent(path)}.json`);
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined && v !== "") {
      url.searchParams.set(k, String(v));
    }
  }
  const response = await fetch(url.toString(), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const payload = await response.json();
  const count = Number(payload?.count);
  return Number.isFinite(count) ? count : null;
}

function isoDateDaysAgo(days) {
  const date = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
  return date.toISOString().slice(0, 10);
}

async function refreshTraffic(base) {
  const total = await fetchGoatCount(base, "TOTAL");
  const page = await fetchGoatCount(base, currentGoatPath());
  const last30 = await fetchGoatCount(base, "TOTAL", { start: isoDateDaysAgo(30) });

  setTrafficValue("trafficTotal", fmtInt(total));
  setTrafficValue("trafficPage", fmtInt(page));
  setTrafficValue("trafficMonth", fmtInt(last30));
  setTrafficStatus(`Last updated: ${new Date().toLocaleTimeString()}`);
}

async function initTraffic() {
  const base = getGoatcounterBase();
  if (!base) {
    setTrafficStatus("Analytics disabled. Set data-goatcounter-code on <body> to enable.");
    return;
  }

  setTrafficStatus("Loading live traffic ...");
  try {
    await ensureGoatcounterScript(base);
    await refreshTraffic(base);

    if (trafficIntervalId) clearInterval(trafficIntervalId);
    trafficIntervalId = window.setInterval(() => {
      void refreshTraffic(base).catch((err) => {
        setTrafficStatus(
          "Traffic refresh failed. Enable GoatCounter setting 'allow using the visitor counter' and check ad-blockers.",
          "error"
        );
      });
    }, TRAFFIC_REFRESH_MS);
  } catch (err) {
    setTrafficStatus(
      "Traffic init failed. Enable GoatCounter setting 'allow using the visitor counter' and check ad-blockers.",
      "error"
    );
  }
}

function setModelJudgerNote(primaryJudger) {
  const node = document.getElementById("modelJudgerNote");
  if (!node) return;
  if (!primaryJudger) {
    node.textContent = "Leaderboard ASR judger: N/A.";
    return;
  }
  node.innerHTML = `Leaderboard ASR judger: <code>${esc(judgerLabel(primaryJudger))}</code>.`;
}

function setMeta(meta) {
  const generated = document.getElementById("metaGeneratedAt");
  const source = document.getElementById("metaSource");
  const schema = document.getElementById("metaSchema");
  if (generated) generated.textContent = meta.generated_at || "-";
  if (source) source.textContent = meta.source || "-";
  if (schema) schema.textContent = meta.schema_version || "-";
}

function renderOverview(overview, modelRows, defenseRows, attackRows, paper) {
  const blackModel = modelRows.find((r) => r.access === "black-box" || (!r.access && inferModelType(r.model) === "api"));
  const whiteModel = modelRows.find((r) => r.access === "white-box" || (!r.access && inferModelType(r.model) === "local"));
  const blackAttack = paper?.attacks_by_access?.black_box?.[0] || attackRows.find((r) => r.access === "black-box");
  const whiteAttack = paper?.attacks_by_access?.white_box?.[0] || attackRows.find((r) => r.access === "white-box-only");
  const defense = defenseRows[0];

  const cards = [
    {
      label: "Best Black-Box Model",
      value: blackModel?.model || "N/A",
      metric: `No-defense ASR ${fmtPct(blackModel?.no_defense_asr ?? blackModel?.avg_asr)}; robustness ${fmtPct(
        blackModel?.robustness_score
      )}`,
    },
    {
      label: "Best White-Box Model",
      value: whiteModel?.model || "N/A",
      metric: `No-defense ASR ${fmtPct(whiteModel?.no_defense_asr ?? whiteModel?.avg_asr)}; robustness ${fmtPct(
        whiteModel?.robustness_score
      )}`,
    },
    {
      label: "Best Black-Box Attack",
      value: blackAttack?.attack || "N/A",
      metric: `Residual ASR ${fmtPct(blackAttack?.residual_defended_asr)}; no-defense ASR ${fmtPct(
        blackAttack?.no_defense_asr ?? blackAttack?.avg_asr
      )}`,
    },
    {
      label: "Best White-Box Attack",
      value: whiteAttack?.attack || "N/A",
      metric: `Residual ASR ${fmtPct(whiteAttack?.residual_defended_asr)}; no-defense ASR ${fmtPct(
        whiteAttack?.no_defense_asr ?? whiteAttack?.avg_asr
      )}`,
    },
    {
      label: "Best Defense",
      value: defense?.defense || "N/A",
      metric: `Defense gain ${fmtPp(defense?.gain ?? defense?.asr_gain_vs_no_defense)}; defended ASR ${fmtPct(
        defense?.defended_asr ?? defense?.avg_asr
      )}`,
    },
  ];

  const container = document.getElementById("overviewCards");
  if (container) {
    container.innerHTML = cards
      .map(
        (card) =>
          `<article class="card overview-card"><span class="label">${esc(card.label)}</span><span class="value">${esc(
            card.value
          )}</span><span class="metric">${esc(card.metric)}</span></article>`
      )
      .join("");
  }

  const footnote = document.getElementById("overviewFootnote");
  if (footnote) {
    footnote.innerHTML = `Footnote: leaderboard ASR metrics use only <code>${esc(
      paper?.policy?.primary_judger_label || judgerLabel(DEFAULT_JUDGER_KEY)
    )}</code>. HarmBench and Prefix Judger are exposed as independent comparison views below.`;
  }
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

  const blackAttackCount = matrix?.column_groups?.black_box_attacks ?? attacks.length;
  const blackModelCount = matrix?.row_groups?.black_box_models ?? rows.length;
  const head = `<thead><tr><th>Model</th>${attacks
    .map((a, idx) => `<th class="${idx === blackAttackCount ? "matrix-divider-left" : ""}">${esc(a)}</th>`)
    .join("")}</tr></thead>`;

  const body = `<tbody>${rows
    .map((row, rowIdx) => {
      const cells = (row.cells || [])
        .map((cell, cellIdx) => {
          const dividerClass = cellIdx === blackAttackCount ? "matrix-divider-left" : "";
          if (cell.asr === null || cell.asr === undefined) {
            return `<td class="${dividerClass}" title="No run">-</td>`;
          }
          const ratio = (cell.asr - min) / span;
          const hue = Math.max(0, 140 - ratio * 140);
          const bg = `hsl(${hue}deg 70% 38%)`;
          const sourceTag = cell.source === "no_defense" ? "" : " *";
          return `<td class="${dividerClass}" style="background:${bg}" title="ASR=${fmtPct(
            cell.asr
          )}, source=${esc(cell.source)}, runs=${cell.run_count}">${fmtPct(cell.asr, 1)}${sourceTag}</td>`;
        })
        .join("");
      return `<tr class="${rowIdx === blackModelCount ? "matrix-divider-top" : ""}"><td class="first-col">${esc(
        row.model
      )}</td>${cells}</tr>`;
    })
    .join("")}</tbody>`;

  table.innerHTML = `${head}${body}`;
}

function renderAttackDefenseMatrix(matrix) {
  const table = document.getElementById("attackDefenseMatrixTable");
  const legend = document.getElementById("attackDefenseMatrixLegend");
  if (!table || !legend || !matrix) return;

  const isGain = matrix.metric === "defense_gain";
  const min = isGain ? (matrix?.range?.min_gain ?? -0.2) : (matrix?.range?.min_asr ?? 0);
  const max = isGain ? (matrix?.range?.max_gain ?? 0.5) : (matrix?.range?.max_asr ?? 1);
  const span = Math.max(max - min, 1e-6);
  if (isGain) {
    legend.innerHTML =
      '<div class="swatch swatch-gain"></div><span>Backfire</span><span style="margin-left:auto">Higher ASR reduction</span>';
  } else {
    setMatrixLegend(legend);
  }

  const attacks = Array.isArray(matrix.attacks) ? matrix.attacks : [];
  const defenses = Array.isArray(matrix.defenses) ? matrix.defenses : [];
  const rows = Array.isArray(matrix.rows) ? matrix.rows : [];
  if (!attacks.length || !defenses.length || !rows.length) {
    table.innerHTML = '<tbody><tr><td class="empty">No matrix data</td></tr></tbody>';
    return;
  }

  const blackAttackCount = matrix?.column_groups?.black_box_attacks ?? attacks.length;
  const blackDefenseCount = matrix?.row_groups?.black_box_defenses ?? defenses.length;
  const head = `<thead><tr><th>Defense</th>${attacks
    .map((a, idx) => `<th class="${idx === blackAttackCount ? "matrix-divider-left" : ""}">${esc(a)}</th>`)
    .join("")}</tr></thead>`;
  const body = `<tbody>${rows
    .map((row, rowIdx) => {
      const cells = (row.cells || [])
        .map((cell, cellIdx) => {
          const value = isGain ? cell.gain : cell.asr;
          if (value === null || value === undefined) {
            return `<td class="${cellIdx === blackAttackCount ? "matrix-divider-left" : ""}" title="No run">-</td>`;
          }
          const ratio = (value - min) / span;
          const hue = isGain ? Math.max(8, Math.min(155, 8 + ratio * 147)) : Math.max(0, 140 - ratio * 140);
          const bg = `hsl(${hue}deg 70% 38%)`;
          const label = isGain ? fmtPp(value, 0) : fmtPct(value, 1);
          const title = isGain
            ? `Defense Gain=${fmtPp(value)}, residual ASR=${fmtPct(cell.asr)}, runs=${cell.run_count}, models=${
                cell.model_coverage ?? 0
              }, judged=${cell.judged_samples ?? 0}`
            : `ASR=${fmtPct(cell.asr)}, runs=${cell.run_count}, models=${cell.model_coverage ?? 0}, judged=${
                cell.judged_samples ?? 0
              }`;
          return `<td class="${cellIdx === blackAttackCount ? "matrix-divider-left" : ""}" style="background:${bg}" title="${esc(
            title
          )}">${esc(label)}</td>`;
        })
        .join("");
      return `<tr class="${rowIdx === blackDefenseCount ? "matrix-divider-top" : ""}"><td class="first-col">${esc(
        row.defense
      )}</td>${cells}</tr>`;
    })
    .join("")}</tbody>`;

  table.innerHTML = `${head}${body}`;
}

function fillSelect(selectId, values, options = {}) {
  const select = document.getElementById(selectId);
  if (!select) return;
  const cleanValues = Array.isArray(values) ? values : [];
  const includeAll = options.includeAll !== false;
  const optionHtml = cleanValues
    .map((v) => {
      const value = typeof v === "object" && v !== null ? v.key : v;
      const label = typeof v === "object" && v !== null ? v.label || v.key : v;
      return `<option value="${esc(value)}">${esc(label)}</option>`;
    })
    .join("");
  select.innerHTML = `${includeAll ? '<option value="">All</option>' : ""}${optionHtml}`;
  if (options.defaultValue && Array.from(select.options).some((opt) => opt.value === options.defaultValue)) {
    select.value = options.defaultValue;
  }
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

function extractSampleView(sample, selectedJudgerKey = DEFAULT_JUDGER_KEY) {
  const cleanPrompt = pickFirst(sample, ["clean_prompt", "prompt_on_clean", "original_prompt"]);
  const attackedPrompt = pickFirst(sample, ["attacked_prompt", "prompt_on_attacked", "jailbreak_prompt"]);
  const defendedPrompt = pickFirst(sample, ["attacked_prompt_under_defense", "defended_prompt"]);

  const responseClean = pickFirst(sample, ["llm_response_on_clean"]);
  const responseAttacked = pickFirst(sample, ["llm_response_on_attacked"]);
  const responseCleanUnderDefense = pickFirst(sample, ["llm_response_on_clean_under_defense"]);
  const responseAttackedUnderDefense = pickFirst(sample, ["llm_response_on_attacked_under_defense"]);
  const responsePrimary = pickFirst(sample, ["llm_response", "response"]);

  const judgerIndividual = sample.judger_individual_results;
  const selectedJudger =
    judgerIndividual && typeof judgerIndividual === "object" && selectedJudgerKey in judgerIndividual
      ? judgerIndividual[selectedJudgerKey]
      : null;
  const judgerOverall = selectedJudger ?? pickFirst(sample, [
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
    selectedJudgerKey,
    judgerOverall,
    judgerOnClean: sample.judger_result_on_clean,
    judgerOnAttack: sample.judger_result_on_attack,
    judgerOnCleanUnderDefense: sample.judger_result_on_clean_under_defense,
    judgerOnAttackUnderDefense: sample.judger_result_on_attack_under_defense,
    judgerIndividual,
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
  const selectedJudgerKey = sampleViews.find((s) => s.selectedJudgerKey)?.selectedJudgerKey || DEFAULT_JUDGER_KEY;

  const cards = [
    { label: "Samples", value: sampleCount },
    { label: "Selected Judger ASR", value: fmtPct(avgOverall) },
    { label: "Judger", value: judgerLabel(selectedJudgerKey) },
    { label: "Completion", value: fmtPct(completion) },
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
    <span>Clean Prompt</span>
    <span>Attacked Prompt</span>
    <span>Evaluated Response</span>
    <span>Selected Judger</span>
    <span>Individual Judgers</span>
  </div>`;
}

function renderSampleItem(item) {
  const overallScore = normalizeScore(item.judgerOverall);
  const scoreClass = asrClass(overallScore);
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
    `<span class="sample-badge ${scoreClass}">${esc(judgerLabel(item.selectedJudgerKey))}=${esc(fmtPct(overallScore))}</span>`,
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

function runJudgerMetric(row, judgerKey = DEFAULT_JUDGER_KEY) {
  const key = judgerKey || DEFAULT_JUDGER_KEY;
  const metric = row?.judger_scores?.[key];
  return {
    asr: metric?.asr ?? (key === DEFAULT_JUDGER_KEY ? row?.asr : null),
    judgedSamples: metric?.judged_samples ?? (key === DEFAULT_JUDGER_KEY ? row?.judged_samples : 0),
    label: metric?.label || judgerLabel(key),
  };
}

function buildRunScope(filteredRuns, judgerKey = DEFAULT_JUDGER_KEY) {
  const modelSet = uniqueSorted(filteredRuns.map((r) => r.model));
  const attackSet = uniqueSorted(filteredRuns.map((r) => r.attack));
  const defenseSet = uniqueSorted(filteredRuns.map((r) => r.defense));

  const metrics = filteredRuns.map((r) => runJudgerMetric(r, judgerKey));
  const judgedSamples = metrics.reduce((acc, m) => acc + (m.judgedSamples || 0), 0);
  const totalSamples = filteredRuns.reduce((acc, r) => acc + (r.total_samples || 0), 0);
  const successfulSamples = filteredRuns.reduce((acc, r) => acc + (r.successful_samples || 0), 0);

  const weightedAsr = weightedMean(metrics.map((m) => [m.asr, Math.max(m.judgedSamples || 0, 1)]));
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
    judgerLabel: judgerLabel(judgerKey),
  };
}

function renderRunScope(scope) {
  const cards = [
    { label: "Filtered Runs", value: scope.runCount },
    { label: "Judger", value: scope.judgerLabel },
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
  const query = dataset?.paper?.query || {};
  const queryRuns = Array.isArray(query.runs) ? query.runs : [];
  const filters = query.filters || {};
  const section = document.getElementById("runsSection");
  if (!section) return;

  fillSelect("filterModel", filters.models);
  fillSelect("filterAttack", filters.attacks);
  fillSelect("filterDefense", filters.defenses);
  fillSelect("filterJudger", filters.judgers, { includeAll: false, defaultValue: DEFAULT_JUDGER_KEY });

  const controls = [
    "filterModel",
    "filterAttack",
    "filterDefense",
    "filterSample",
    "filterJudger",
    "filterSearch",
  ].map((id) => document.getElementById(id)).filter(Boolean);

  let activeRunToken = 0;

  function currentFilters() {
    return {
      model: document.getElementById("filterModel").value,
      attack: document.getElementById("filterAttack").value,
      defense: document.getElementById("filterDefense").value,
      sample: document.getElementById("filterSample").value.trim(),
      judger: document.getElementById("filterJudger").value || DEFAULT_JUDGER_KEY,
      search: document.getElementById("filterSearch").value.trim().toLowerCase(),
    };
  }

  function filterRuns() {
    const f = currentFilters();
    return queryRuns
      .filter((r) => !f.model || r.model === f.model)
      .filter((r) => !f.attack || r.attack === f.attack)
      .filter((r) => !f.defense || r.defense === f.defense)
      .filter((r) => runJudgerMetric(r, f.judger).asr !== null)
      .filter((r) => {
        if (!f.search) return true;
        return (
          r.model.toLowerCase().includes(f.search) ||
          r.attack.toLowerCase().includes(f.search) ||
          r.defense.toLowerCase().includes(f.search) ||
          r.dataset.toLowerCase().includes(f.search) ||
          judgerLabel(f.judger).toLowerCase().includes(f.search)
        );
      })
      .sort((a, b) => {
        const asrA = runJudgerMetric(a, f.judger).asr ?? -1;
        const asrB = runJudgerMetric(b, f.judger).asr ?? -1;
        if (asrA !== asrB) return asrB - asrA;
        return String(b.updated_at || "").localeCompare(String(a.updated_at || ""));
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
      const sampleFilter = currentFilters().sample;
      const selectedJudger = currentFilters().judger || DEFAULT_JUDGER_KEY;
      const sampleViews = samples
        .map((sample) => extractSampleView(sample, selectedJudger))
        .filter((sample) => !sampleFilter || String(sample.sampleIndex) === sampleFilter);
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
    const f = currentFilters();
    const filtered = filterRuns();
    const scope = buildRunScope(filtered, f.judger);
    renderRunScope(scope);

    renderTable(
      "runsTable",
      [
        { label: "Model", key: "model" },
        { label: "Attack", key: "attack" },
        { label: "Defense", key: "defense" },
        {
          label: "ASR",
          html: true,
          render: (r) => {
            const value = runJudgerMetric(r, f.judger).asr;
            return `<span class="${asrClass(value)}">${fmtPct(value)}</span>`;
          },
        },
        {
          label: "Judged",
          html: true,
          render: (r) => `<span class="mono">${esc(runJudgerMetric(r, f.judger).judgedSamples)}</span>`,
        },
        {
          label: "Sample Completion",
          html: true,
          render: (r) => `<span class="${asrClass(1 - (r.sample_success_rate ?? 0))}">${fmtPct(r.sample_success_rate)}</span>`,
        },
        { label: "Total", key: "total_samples", className: "mono" },
        { label: "Dataset", key: "dataset" },
        { label: "Judger", html: true, render: () => esc(judgerLabel(f.judger)) },
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
      const sampleEl = document.getElementById("filterSample");
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

      sampleEl.value = "";
      searchEl.value = "";
      paint();
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    },
  };
}

function renderLeaderboards(data) {
  const paper = data.paper || {};
  const modelRows = [...(paper.models || [])].sort((a, b) => {
    const av = a.no_defense_asr ?? a.avg_asr;
    const bv = b.no_defense_asr ?? b.avg_asr;
    if ((av === null || av === undefined) !== (bv === null || bv === undefined)) return av === null || av === undefined ? 1 : -1;
    if (av !== bv) return av - bv;
    return String(a.model || "").localeCompare(String(b.model || ""));
  });
  const apiModels = withSubRank(
    modelRows.filter((row) => row.access === "black-box" || (!row.access && inferModelType(row.model) === "api"))
  );
  const localModels = withSubRank(
    modelRows.filter((row) => row.access === "white-box" || (!row.access && inferModelType(row.model) === "local"))
  );

  const modelColumns = [
    { label: "Rank", key: "sub_rank", className: "mono" },
    { label: "Model", key: "model" },
    {
      label: "No-Def ASR",
      html: true,
      render: (r) => {
        const value = r.no_defense_asr ?? r.avg_asr;
        return metricHint(fmtPct(value), `Attack success rate against the undefended target model. judged_samples=${fmtInt(r.judged_samples)}.`, asrClass(value));
      },
    },
    {
      label: "Clean ASR",
      html: true,
      render: (r) => {
        const value = r.clean_asr ?? r.no_attack_unsafe_rate;
        return metricHint(fmtPct(value), "No-attack harmful-query baseline.", asrClass(value));
      },
    },
    {
      label: "ASR Lift",
      html: true,
      render: (r) => metricHint(fmtPp(r.asr_lift), "No-defense ASR minus clean ASR.", deltaClass(-(r.asr_lift ?? 0), true)),
    },
    {
      label: "Induced Harmfulness",
      html: true,
      render: (r) =>
        metricHint(
          fmtPct(r.induced_harmfulness),
          `Clean-safe to attacked-harmful transition on matched sample identifiers. samples=${fmtInt(r.transition_samples)}.`,
          asrClass(r.induced_harmfulness)
        ),
    },
    {
      label: "Suppression",
      html: true,
      render: (r) =>
        metricHint(fmtPct(r.attack_suppression), "Clean-harmful to attacked-safe transition; high values can indicate failed attack execution."),
    },
    {
      label: "Strongest Attack",
      html: true,
      render: (r) =>
        r.strongest_attack
          ? `<span title="No-defense ASR=${esc(fmtPct(r.strongest_attack_asr))}">${esc(r.strongest_attack)} <span class="${asrClass(
              r.strongest_attack_asr
            )}">${esc(fmtPct(r.strongest_attack_asr, 0))}</span></span>`
          : "N/A",
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
      label: "Completion",
      html: true,
      render: (r) => metricHint(fmtPct(r.sample_completion_rate), `Completed no-defense samples across ${r.run_count} runs.`),
    },
    {
      label: "Latency",
      html: true,
      render: (r) => metricHint(fmtSec(r.avg_latency_s), "Average target-model latency on no-defense attack runs."),
    },
  ];

  renderTable("modelApiTable", modelColumns, apiModels, (row) => {
    focusRunsByEntity("model", row.model);
  });

  renderTable("modelLocalTable", modelColumns, localModels, (row) => {
    focusRunsByEntity("model", row.model);
  });

  const defenseColumns = [
    { label: "Rank", key: "rank", className: "mono" },
    { label: "Defense", key: "defense" },
    {
      label: "Defended ASR",
      html: true,
      render: (r) =>
        metricHint(
          fmtPct(r.defended_asr ?? r.avg_asr),
          `ASR of the defended endpoint over matched pairs. matched_pairs=${r.matched_pair_count}; judged_samples=${fmtInt(r.judged_samples)}.`,
          asrClass(r.defended_asr ?? r.avg_asr)
        ),
    },
    {
      label: "Defense Gain",
      html: true,
      render: (r) => {
        const value = r.gain ?? r.asr_gain_vs_no_defense;
        return metricHint(fmtPp(value), `ASR gain = baseline(no_defense) - defense ASR. Positive is better.`, deltaClass(value));
      },
    },
    {
      label: "Backfire",
      html: true,
      render: (r) =>
        metricHint(fmtPct(r.defense_backfire), "Attacked-safe to defended-harmful transition on matched samples.", asrClass(r.defense_backfire)),
    },
    {
      label: "Clean Backfire",
      html: true,
      render: (r) =>
        metricHint(fmtPct(r.clean_input_backfire), "Clean-safe to defended-harmful transition under no attack.", asrClass(r.clean_input_backfire)),
    },
    {
      label: "Win / Worse",
      html: true,
      render: (r) => `${esc(fmtPct(r.win_rate, 0))} / <span class="${asrClass(r.worse_rate)}">${esc(fmtPct(r.worse_rate, 0))}</span>`,
    },
    {
      label: "Utility ΔAcc",
      html: true,
      render: (r) =>
        metricHint(fmtPp(r.utility_accuracy_delta), `Benign-question accuracy change. utility_pairs=${r.utility_pairs || 0}.`, deltaClass(r.utility_accuracy_delta)),
    },
    {
      label: "Utility ΔRefusal",
      html: true,
      render: (r) =>
        metricHint(fmtPp(r.utility_refusal_delta), `Benign-question refusal-rate change. Lower is better.`, deltaClass(r.utility_refusal_delta, false)),
    },
    {
      label: "Matched Pairs",
      html: true,
      render: (r) => metricHint(String(r.matched_pair_count), `Matched (model, attack, dataset, judger) pairs: ${r.matched_pair_count}.`, "mono"),
    },
    {
      label: "Coverage",
      html: true,
      render: (r) => metricHint(r.coverage || `${r.model_coverage}M/${r.attack_coverage}A`, `Models=${r.model_coverage}; attacks=${r.attack_coverage}.`, "mono"),
    },
    {
      label: "Latency",
      html: true,
      render: (r) => metricHint(fmtSec(r.avg_latency_s), "Average defended endpoint latency."),
    },
  ];

  renderTable("defenseBlackBoxTable", defenseColumns, paper.defenses_by_access?.black_box || [], (row) => {
    focusRunsByEntity("defense", row.defense);
  });
  renderTable("defenseWhiteBoxTable", defenseColumns, paper.defenses_by_access?.white_box || [], (row) => {
    focusRunsByEntity("defense", row.defense);
  });

  const attackColumns = [
    { label: "Rank", key: "rank", className: "mono" },
    { label: "Attack", key: "attack" },
    {
      label: "Residual ASR",
      html: true,
      render: (r) =>
        metricHint(
          fmtPct(r.residual_defended_asr),
          `ASR after defenses are integrated with the target model. defended_runs=${r.defended_run_count}.`,
          asrClass(r.residual_defended_asr)
        ),
    },
    {
      label: "No-Def ASR",
      html: true,
      render: (r) =>
        metricHint(fmtPct(r.no_defense_asr ?? r.avg_asr), `Undefended target-model ASR. runs=${r.run_count}.`, asrClass(r.no_defense_asr ?? r.avg_asr)),
    },
    {
      label: "Retention",
      html: true,
      render: (r) => metricHint(fmtPct(r.asr_retention), "Residual defended ASR divided by no-defense ASR."),
    },
    {
      label: "Induced",
      html: true,
      render: (r) =>
        metricHint(
          fmtPct(r.induced_harmfulness),
          `Clean-safe to attacked-harmful transition. samples=${fmtInt(r.transition_samples)}.`,
          asrClass(r.induced_harmfulness)
        ),
    },
    {
      label: "Suppression",
      html: true,
      render: (r) => metricHint(fmtPct(r.attack_suppression), "Clean-harmful to attacked-safe transition."),
    },
    {
      label: "Spread",
      html: true,
      render: (r) => metricHint(fmtPp(r.model_spread), `Cross-model no-defense ASR range: ${fmtPct(r.min_model_asr)} to ${fmtPct(r.max_model_asr)}.`),
    },
    {
      label: "Queries",
      html: true,
      render: (r) => metricHint(fmtFloat(r.avg_target_queries, 1), "Average target-model queries per sample.", "mono"),
    },
    {
      label: "Assistant Refusal",
      html: true,
      render: (r) => metricHint(fmtPct(r.assistant_refusal_rate), `Samples with assistant-refusal instrumentation: ${fmtInt(r.assistant_refusal_samples)}.`),
    },
    {
      label: "Coverage",
      html: true,
      render: (r) => metricHint(`${r.model_coverage}M/${r.defense_coverage}D`, `No-defense models=${r.model_coverage}; defended defense states=${r.defense_coverage}.`, "mono"),
    },
  ];

  renderTable("attackBlackBoxTable", attackColumns, paper.attacks_by_access?.black_box || [], (row) => {
    focusRunsByEntity("attack", row.attack);
  });
  renderTable("attackWhiteBoxTable", attackColumns, paper.attacks_by_access?.white_box || [], (row) => {
    focusRunsByEntity("attack", row.attack);
  });
}

async function init() {
  void initTraffic();
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) {
      throw new Error(`Failed to load ${DATA_URL}: ${response.status}`);
    }
    const dataset = await response.json();

    const paper = dataset.paper || {};
    const sortedModelRows = [...(paper.models || [])].sort((a, b) => {
      const av = a.no_defense_asr ?? a.avg_asr;
      const bv = b.no_defense_asr ?? b.avg_asr;
      if ((av === null || av === undefined) !== (bv === null || bv === undefined)) return av === null || av === undefined ? 1 : -1;
      if (av !== bv) return av - bv;
      return String(a.model || "").localeCompare(String(b.model || ""));
    });
    setMeta(dataset.meta);
    runBreakdownModalApi = setupRunBreakdownModal();
    renderOverview(
      dataset.overview,
      sortedModelRows,
      paper.defenses || [],
      paper.attacks || [],
      paper
    );
    renderRunsSection(dataset);
    renderLeaderboards(dataset);
    setModelJudgerNote(paper?.policy?.primary_judger);
    renderModelAttackMatrix(paper.model_attack_matrix || dataset.model_attack_matrix || dataset.matrix);
    renderAttackDefenseMatrix(paper.attack_defense_matrix || dataset.attack_defense_matrix);
  } catch (err) {
    document.body.innerHTML = `<main class="page"><section class="section"><h2>Failed to load leaderboard</h2><pre>${esc(
      err?.stack || err
    )}</pre></section></main>`;
  }
}

init();
