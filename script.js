/**
 * script.js
 * AI-Powered 6G RIS Platform — Frontend Logic
 * Connects dashboard UI to FastAPI backend
 * JCAS Dashboard v2.4.1
 */

'use strict';

// ── Config ─────────────────────────────────────────────────────────────────
const API_BASE = window.location.origin;   // same-origin; change if backend is separate

// ── Utility ────────────────────────────────────────────────────────────────

/** Show a brief toast notification */
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

/** Generic API call wrapper with error handling */
async function apiFetch(path, options = {}) {
  try {
    const resp = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return await resp.json();
  } catch (e) {
    console.error(`[API] ${path}`, e);
    throw e;
  }
}

// ── Read current slider values from the dashboard UI ──────────────────────

function getSimParams() {
  const get = (id, fallback) => {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) : fallback;
  };
  return {
    velocity:       get('sl-vel',   45),
    ris_elements:   Math.round(get('sl-ris', 64)),
    tx_power_dbm:   get('sl-pow',  30),
    alpha:          +(get('sl-alpha', 60) / 100).toFixed(2),
    frequency_thz:  +(get('sl-freq', 3) * 0.1).toFixed(2),
    num_users:      Math.round(get('sl-usr', 4)),
  };
}

// ── Simulation ─────────────────────────────────────────────────────────────

/**
 * POST /api/simulate — run physics + AI + optimiser on backend
 * Updates dashboard metrics live with API response.
 */
async function apiSimulate() {
  const params = getSimParams();
  toast('Running backend simulation…', 'info');
  try {
    const data = await apiFetch('/api/simulate', {
      method: 'POST',
      body: JSON.stringify(params),
    });
    applySimResult(data);
    toast(`Sim #${data.id} complete — SNR ${data.snr_db} dB`, 'success');
    return data;
  } catch (e) {
    toast(`Simulation failed: ${e.message}`, 'error');
  }
}

/** Patch live dashboard metric elements with backend values */
function applySimResult(data) {
  const patch = (id, val, unit = '') => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = val + (unit ? `<span class="munit">${unit}</span>` : '');
  };
  if (data.snr_db    !== undefined) patch('m-snr',    data.snr_db.toFixed(1),  'dB');
  if (data.wsr_mbps  !== undefined) patch('m-wsr',    data.wsr_mbps.toFixed(1),'M');
  if (data.outage_pct!== undefined) patch('m-out',    data.outage_pct.toFixed(1),'%');
  if (data.doppler_hz!== undefined) patch('m-dop',    Math.round(data.doppler_hz),'Hz');
  if (data.rl_reward !== undefined) patch('m-reward', data.rl_reward.toFixed(2));

  // Update predicted SNR overlay if element exists
  const predEl = document.getElementById('m-predicted-snr');
  if (predEl && data.predicted_snr !== undefined) {
    predEl.textContent = `AI: ${data.predicted_snr.toFixed(1)} dB`;
  }

  // Append to simulation history table if it exists
  appendHistoryRow(data);
}

// ── AI Prediction ───────────────────────────────────────────────────────────

/**
 * POST /api/predict — ML-based SNR prediction
 */
async function apiPredict() {
  const params = getSimParams();
  toast('Querying AI predictor…', 'info');
  try {
    const data = await apiFetch('/api/predict', {
      method: 'POST',
      body: JSON.stringify({
        velocity:      params.velocity,
        ris_elements:  params.ris_elements,
        tx_power_dbm:  params.tx_power_dbm,
        frequency_thz: params.frequency_thz,
        num_users:     params.num_users,
      }),
    });
    toast(`AI SNR Prediction: ${data.predicted_snr_db} dB (conf ${(data.confidence * 100).toFixed(0)}%)`, 'success');
    addChatMessage(`AI Prediction\n▸ SNR: ${data.predicted_snr_db} dB\n▸ Confidence: ${(data.confidence*100).toFixed(1)}%\n▸ Model: ${data.model}`, 'ai');
    return data;
  } catch (e) {
    toast(`Prediction failed: ${e.message}`, 'error');
  }
}

// ── RIS Optimization ────────────────────────────────────────────────────────

/**
 * POST /api/optimize — run RIS phase optimization
 */
async function apiOptimize() {
  const params = getSimParams();
  toast('Optimizing RIS phases…', 'info');
  try {
    const data = await apiFetch('/api/optimize', {
      method: 'POST',
      body: JSON.stringify({
        ris_elements:  params.ris_elements,
        target_snr_db: 20.0,
        phase_bits:    4,
      }),
    });
    toast(`RIS Optimized — gain +${data.estimated_snr_gain.toFixed(2)} dB (${data.strategy})`, 'success');
    addChatMessage(
      `RIS Optimization Complete\n▸ Elements: ${data.ris_elements}\n▸ Strategy: ${data.strategy}\n▸ SNR Gain: +${data.estimated_snr_gain} dB\n▸ Phase bits: ${data.phase_bits}`,
      'ai'
    );
    return data;
  } catch (e) {
    toast(`Optimization failed: ${e.message}`, 'error');
  }
}

// ── Analytics ───────────────────────────────────────────────────────────────

/**
 * GET /api/analytics — fetch aggregate stats and display
 */
async function apiAnalytics() {
  toast('Fetching analytics…', 'info');
  try {
    const data = await apiFetch('/api/analytics');
    const msg = [
      `Dashboard Analytics`,
      `▸ Total Simulations: ${data.total_simulations}`,
      `▸ Avg SNR: ${data.avg_snr_db} dB`,
      `▸ Avg WSR: ${data.avg_wsr_mbps} Mbps`,
      `▸ Avg Outage: ${data.avg_outage_pct}%`,
      `▸ Best SNR: ${data.best_snr_db} dB`,
      `▸ Best WSR: ${data.best_wsr_mbps} Mbps`,
    ].join('\n');
    addChatMessage(msg, 'ai');
    toast(`Analytics loaded — ${data.total_simulations} simulations`, 'success');
    return data;
  } catch (e) {
    toast(`Analytics failed: ${e.message}`, 'error');
  }
}

// ── Simulation History ──────────────────────────────────────────────────────

async function apiLoadHistory(limit = 20) {
  try {
    const data = await apiFetch(`/api/simulations?limit=${limit}`);
    renderHistoryTable(data);
    return data;
  } catch (e) {
    toast(`History load failed: ${e.message}`, 'error');
  }
}

function renderHistoryTable(rows) {
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  rows.forEach(r => appendHistoryRow(r));
}

function appendHistoryRow(r) {
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;
  // Keep max 30 rows
  while (tbody.children.length >= 30) tbody.removeChild(tbody.firstChild);
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${r.id}</td>
    <td>${r.velocity}</td>
    <td>${r.ris_elements}</td>
    <td>${r.snr_db !== null ? r.snr_db.toFixed(2) : '—'}</td>
    <td>${r.wsr_mbps !== null ? r.wsr_mbps.toFixed(2) : '—'}</td>
    <td>${r.outage_pct !== null ? r.outage_pct.toFixed(2) : '—'}</td>
    <td>${r.rl_reward !== null ? r.rl_reward.toFixed(2) : '—'}</td>
    <td><button onclick="apiDeleteSim(${r.id})" style="font-size:9px;padding:2px 6px;background:rgba(255,60,95,.08);border:1px solid rgba(255,60,95,.3);color:#ff3c5f;cursor:pointer;border-radius:2px;font-family:var(--mono)">DEL</button></td>
  `;
  tbody.appendChild(tr);
}

async function apiDeleteSim(id) {
  try {
    await apiFetch(`/api/simulations/${id}`, { method: 'DELETE' });
    toast(`Simulation #${id} deleted`, 'warn');
    apiLoadHistory();
  } catch (e) {
    toast(`Delete failed: ${e.message}`, 'error');
  }
}

// ── Model Retraining ────────────────────────────────────────────────────────

async function apiRetrain(nSamples = 2000) {
  toast(`Retraining ML model on ${nSamples} samples…`, 'info');
  try {
    const data = await apiFetch(`/api/model/retrain?n_samples=${nSamples}`, { method: 'POST' });
    const m = data.metrics;
    addChatMessage(
      `Model Retrain Complete\n▸ Samples: ${m.train_samples + m.test_samples}\n▸ MAE: ${m.mae_db} dB\n▸ R²: ${m.r2_score}\n▸ Model: ${m.model}`,
      'ai'
    );
    toast(`Retrain done — MAE ${m.mae_db} dB, R² ${m.r2_score}`, 'success');
  } catch (e) {
    toast(`Retrain failed: ${e.message}`, 'error');
  }
}

// ── Chat helpers ────────────────────────────────────────────────────────────

function addChatMessage(text, type = 'ai') {
  const box = document.getElementById('chatbox');
  if (!box) return;
  const div = document.createElement('div');
  div.className = `cmsg c${type}`;
  div.style.whiteSpace = 'pre-line';
  div.textContent = '▸ ' + text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// ── Backend API Panel injection ─────────────────────────────────────────────

function injectAPIPanel() {
  if (document.getElementById('api-panel')) return;   // already injected
  const panel = document.createElement('div');
  panel.id = 'api-panel';
  panel.className = 'api-overlay';
  panel.innerHTML = `
    <div class="title"><span class="dot"></span>BACKEND API</div>
    <div class="api-row"><span class="api-label">STATUS</span><span class="api-value" id="api-status">checking…</span></div>
    <div class="api-row"><span class="api-label">ENDPOINT</span><span class="api-value" style="font-size:9px">${API_BASE}/api</span></div>
    <div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:10px">
      <button class="api-btn" onclick="apiSimulate()">▶ SIMULATE</button>
      <button class="api-btn" onclick="apiPredict()">🧠 PREDICT</button>
      <button class="api-btn" onclick="apiOptimize()">⚡ OPTIMIZE</button>
      <button class="api-btn" onclick="apiAnalytics()">📊 ANALYTICS</button>
      <button class="api-btn green-btn" onclick="apiRetrain(2000)">🔁 RETRAIN</button>
      <button class="api-btn" onclick="apiLoadHistory()">📋 HISTORY</button>
    </div>
    <div style="font-size:8px;font-family:var(--mono);color:var(--text3);margin-top:8px">
      Docs: <a href="/api/docs" target="_blank" style="color:var(--cyan)">/api/docs</a> &nbsp;|&nbsp;
      <a href="/api/redoc" target="_blank" style="color:var(--cyan)">/api/redoc</a>
    </div>
  `;
  document.body.appendChild(panel);
}

// ── Health check ────────────────────────────────────────────────────────────

async function checkHealth() {
  const statusEl = document.getElementById('api-status');
  try {
    const data = await apiFetch('/api/health');
    if (statusEl) {
      statusEl.textContent = `ONLINE v${data.version}`;
      statusEl.style.color = 'var(--green)';
    }
    console.log('[API] Health:', data);
  } catch {
    if (statusEl) {
      statusEl.textContent = 'OFFLINE — run main.py';
      statusEl.style.color = 'var(--red)';
    }
  }
}

// ── Auto-wire "Run Simulation" button in the Simulate view ─────────────────

function wireSimButton() {
  // The original dashboard has a "RUN" button calling runSim()
  // We extend it to also call the backend
  const origRunSim = window.runSim;
  window.runSim = function () {
    if (origRunSim) origRunSim();   // keep original frontend animation
    apiSimulate();                   // also hit the backend
  };
}

// ── Init ────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  injectAPIPanel();
  checkHealth();
  wireSimButton();

  // Re-check health every 30 s
  setInterval(checkHealth, 30000);
});

// Expose to global scope so inline onclick handlers can call them
Object.assign(window, {
  apiSimulate,
  apiPredict,
  apiOptimize,
  apiAnalytics,
  apiRetrain,
  apiLoadHistory,
  apiDeleteSim,
});
