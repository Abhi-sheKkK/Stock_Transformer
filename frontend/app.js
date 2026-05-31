/**
 * AIFIS — AI Financial Intelligence System
 * Frontend Application Logic
 */

const API_BASE = '';

// ── State ──
const state = {
  ticker: 'AAPL',
  market: null,
  news: null,
  forecast: null,
  loading: false,
};

// ── DOM Refs ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSearch();
  initActions();
  checkHealth();
});

// ── Navigation ──
function initNavigation() {
  $$('.nav-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const view = tab.dataset.view;
      $$('.nav-tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      $$('.view').forEach((v) => v.classList.remove('active'));
      $(`#view-${view}`).classList.add('active');
    });
  });
}

// ── Search ──
function initSearch() {
  const input = $('#ticker-input');
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      state.ticker = input.value.trim().toUpperCase();
      if (state.ticker) runAnalysis();
    }
  });
  // Global "/" shortcut
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    }
  });
}

// ── Actions ──
function initActions() {
  $('#btn-analyze').addEventListener('click', () => {
    state.ticker = $('#ticker-input').value.trim().toUpperCase();
    if (state.ticker) runAnalysis();
  });
  $('#btn-run-forecast').addEventListener('click', runForecast);
}

// ── Health Check ──
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      $('#status-dot').title = 'API Connected';
      $('.dot').style.background = 'var(--accent)';
    }
  } catch {
    $('#status-dot').title = 'API Offline';
    $('.dot').style.background = 'var(--red)';
  }
}

// ── Full Analysis Pipeline ──
async function runAnalysis() {
  showLoading('Fetching market data...');
  try {
    const [marketRes, newsRes] = await Promise.allSettled([
      fetch(`${API_BASE}/market/${state.ticker}`),
      fetch(`${API_BASE}/news/${state.ticker}?max_articles=10&include_sentiment=true`),
    ]);

    if (marketRes.status === 'fulfilled' && marketRes.value.ok) {
      state.market = await marketRes.value.json();
      renderMarket(state.market);
    } else {
      toast('Market data unavailable', 'error');
    }

    if (newsRes.status === 'fulfilled' && newsRes.value.ok) {
      state.news = await newsRes.value.json();
      renderNews(state.news);
      renderSentiment(state.news.sentiment);
    } else {
      toast('News data unavailable', 'error');
    }

    renderStrip();
    toast(`Analysis complete for ${state.ticker}`, 'success');
  } catch (err) {
    toast(`Analysis failed: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

// ── Forecast ──
async function runForecast() {
  const days = parseInt($('#forecast-days').value);
  const ticker = $('#ticker-input').value.trim().toUpperCase() || state.ticker;
  showLoading('Running Transformer forecast...');
  try {
    const res = await fetch(`${API_BASE}/predict/${ticker}?days=${days}`, { method: 'POST' });
    if (!res.ok) throw new Error(await res.text());
    state.forecast = await res.json();
    renderForecast(state.forecast);
    toast(`${days}-day forecast generated`, 'success');
  } catch (err) {
    toast(`Forecast failed: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

// ── Render: Summary Strip ──
function renderStrip() {
  const m = state.market;
  const s = state.news?.sentiment;
  if (!m) return;
  const ind = m.indicators;
  const sig = m.signals;
  const cur = m.currency || '$';

  $('#strip-ticker-val').textContent = m.ticker;
  $('#strip-price-val').textContent = `${cur}${ind.price.toLocaleString()}`;
  const changeEl = $('#strip-change-val');
  changeEl.textContent = `${ind.price_change_pct >= 0 ? '+' : ''}${ind.price_change_pct.toFixed(2)}%`;
  changeEl.className = `strip-value ${ind.price_change_pct >= 0 ? 'positive' : 'negative'}`;
  $('#strip-rsi-val').textContent = ind.rsi.toFixed(1);
  $('#strip-trend-val').textContent = sig.trend;
  $('#strip-volume-val').textContent = formatNumber(ind.volume);
  if (s) {
    const sentEl = $('#strip-sentiment-val');
    sentEl.textContent = s.overall_label.toUpperCase();
    sentEl.className = `strip-value ${s.overall_label === 'bullish' ? 'positive' : s.overall_label === 'bearish' ? 'negative' : 'neutral-c'}`;
  }
}

// ── Render: Market Snapshot ──
function renderMarket(data) {
  const ind = data.indicators;
  const sig = data.signals;
  const cur = data.currency || '$';

  $('#market-timestamp').textContent = new Date().toLocaleString();
  $('#hero-ticker').textContent = data.ticker;
  $('#hero-price').textContent = `${cur}${ind.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;
  const changeEl = $('#hero-change');
  changeEl.textContent = `${ind.price_change >= 0 ? '+' : ''}${cur}${Math.abs(ind.price_change).toFixed(2)} (${ind.price_change_pct >= 0 ? '+' : ''}${ind.price_change_pct.toFixed(2)}%)`;
  changeEl.className = `hero-change ${ind.price_change_pct >= 0 ? 'positive' : 'negative'}`;

  // Indicator grid
  const indicators = [
    { label: 'MACD', value: ind.macd.toFixed(4), sub: `Signal: ${ind.macd_signal.toFixed(4)}` },
    { label: 'RSI (14)', value: ind.rsi.toFixed(1), sub: sig.rsi },
    { label: 'ATR (14)', value: `${cur}${ind.atr.toFixed(2)}`, sub: 'Volatility' },
    { label: 'VWAP', value: `${cur}${ind.vwap.toFixed(2)}`, sub: sig.vwap_position },
    { label: 'BB Upper', value: `${cur}${ind.bb_upper.toFixed(2)}`, sub: '' },
    { label: 'BB Middle', value: `${cur}${ind.bb_middle.toFixed(2)}`, sub: sig.bollinger },
    { label: 'BB Lower', value: `${cur}${ind.bb_lower.toFixed(2)}`, sub: '' },
    { label: 'Vol Ratio', value: `${ind.volume_ratio.toFixed(2)}x`, sub: sig.volume },
  ];

  $('#indicator-grid').innerHTML = indicators.map((i) => `
    <div class="indicator-item">
      <div class="indicator-label">${i.label}</div>
      <div class="indicator-value">${i.value}</div>
      ${i.sub ? `<div class="indicator-sub">${i.sub}</div>` : ''}
    </div>
  `).join('');

  // Signal grid
  const signals = [
    { name: 'RSI', value: sig.rsi, type: classifySignal(sig.rsi) },
    { name: 'MACD', value: sig.macd, type: classifySignal(sig.macd) },
    { name: 'Bollinger', value: sig.bollinger, type: classifySignal(sig.bollinger) },
    { name: 'Volume', value: sig.volume, type: 'neutral-s' },
    { name: 'Trend', value: sig.trend, type: sig.trend === 'Uptrend' ? 'bullish' : 'bearish' },
    { name: 'VWAP', value: sig.vwap_position, type: classifySignal(sig.vwap_position) },
  ];

  $('#signal-grid').innerHTML = signals.map((s) => `
    <div class="signal-item ${s.type}">
      <div class="signal-name">${s.name}</div>
      <div class="signal-value">${s.value}</div>
    </div>
  `).join('');

  // Technicals view
  renderTechnicals(data);
}

function classifySignal(text) {
  const t = text.toLowerCase();
  if (t.includes('bullish') || t.includes('overbought') || t.includes('uptrend') || t.includes('upper')) return 'bullish';
  if (t.includes('bearish') || t.includes('oversold') || t.includes('downtrend') || t.includes('lower')) return 'bearish';
  return 'neutral-s';
}

// ── Render: News ──
function renderNews(data) {
  const articles = data.articles || [];
  $('#news-count').textContent = `${articles.length} articles`;

  // Overview news list (compact)
  $('#news-list').innerHTML = articles.slice(0, 6).map((a) => `
    <a href="${a.url}" target="_blank" rel="noopener" class="news-item">
      <div class="news-item-title">${escHtml(a.title)}</div>
      <div class="news-item-meta">
        <span class="news-source">${escHtml(a.source)}</span>
        <span>${formatDate(a.published_date)}</span>
        <span class="news-confidence ${confidenceClass(a.truth_score)}">${(a.truth_score * 100).toFixed(0)}%</span>
      </div>
    </a>
  `).join('') || '<p class="table-empty">No news available</p>';

  // Detailed news view
  $('#news-detailed-list').innerHTML = articles.map((a) => `
    <a href="${a.url}" target="_blank" rel="noopener" class="news-detailed-item">
      <div class="news-d-title">${escHtml(a.title)}</div>
      ${a.summary && a.summary !== a.title ? `<div class="news-d-summary">${escHtml(a.summary.slice(0, 200))}</div>` : ''}
      <div class="news-d-meta">
        <span class="news-d-source">${escHtml(a.source)}</span>
        <span class="news-d-date">${formatDate(a.published_date)}</span>
        <span class="news-d-truth ${confidenceClass(a.truth_score)}">${(a.truth_score * 100).toFixed(0)}% truth</span>
        ${a.source_count > 1 ? `<span class="news-d-source">${a.source_count} sources</span>` : ''}
      </div>
    </a>
  `).join('') || '<p class="table-empty">No news available</p>';

  // Source badges
  const breakdown = data.source_breakdown || {};
  $('#source-badges').innerHTML = Object.entries(breakdown).map(([src, count]) =>
    `<span class="source-badge active">${src} (${count})</span>`
  ).join('');

  // Truth pillars (average from first article)
  if (articles.length > 0 && articles[0].pillars) {
    renderTruthPillars(articles[0].pillars);
  }
}

function renderTruthPillars(pillars) {
  const items = [
    { name: 'Consistency', val: pillars.consistency || 0 },
    { name: 'Credibility', val: pillars.credibility || 0 },
    { name: 'Temporal', val: pillars.temporal || 0 },
    { name: 'Contradiction', val: pillars.contradiction_penalty || 0 },
  ];
  $('#truth-pillars').innerHTML = items.map((p) => `
    <div class="pillar-item">
      <span class="pillar-name">${p.name}</span>
      <div class="pillar-bar-wrap"><div class="pillar-bar" style="width:${Math.min(p.val * 100, 100)}%; background:${p.name === 'Contradiction' ? 'var(--red)' : 'var(--accent)'}"></div></div>
      <span class="pillar-val">${p.val.toFixed(2)}</span>
    </div>
  `).join('');
}

// ── Render: Sentiment ──
function renderSentiment(data) {
  if (!data) return;
  const label = data.overall_label || 'neutral';
  const score = data.overall_score || 0;

  $('#sentiment-method').textContent = data.method || '—';
  const gaugeLabel = $('#gauge-label');
  gaugeLabel.textContent = label.toUpperCase();
  gaugeLabel.className = `gauge-label ${label === 'bullish' ? 'positive' : label === 'bearish' ? 'negative' : 'neutral-c'}`;

  const pct = ((score + 1) / 2) * 100;
  const bar = $('#gauge-bar');
  bar.style.width = `${pct}%`;
  bar.style.background = label === 'bullish' ? 'var(--accent)' : label === 'bearish' ? 'var(--red)' : 'var(--text-tertiary)';
  $('#gauge-score').textContent = `Score: ${score >= 0 ? '+' : ''}${score.toFixed(3)}`;

  const dist = data.distribution || {};
  $('#sentiment-dist').innerHTML = `
    <div class="dist-item"><div class="dist-count positive">${dist.bullish || 0}</div><div class="dist-label">Bullish</div></div>
    <div class="dist-item"><div class="dist-count neutral-c">${dist.neutral || 0}</div><div class="dist-label">Neutral</div></div>
    <div class="dist-item"><div class="dist-count negative">${dist.bearish || 0}</div><div class="dist-label">Bearish</div></div>
  `;

  // Sentiment breakdown for news view
  const details = data.details || [];
  $('#sentiment-breakdown').innerHTML = details.slice(0, 10).map((d) => `
    <div class="sb-item">
      <span class="sb-text" title="${escHtml(d.text)}">${escHtml(d.text)}</span>
      <span class="sb-label ${d.label === 'bullish' ? 'sb-bullish' : d.label === 'bearish' ? 'sb-bearish' : 'sb-neutral'}">${d.label}</span>
    </div>
  `).join('') || '<p class="table-empty">No sentiment data</p>';
}

// ── Render: Forecast ──
function renderForecast(data) {
  if (!data || !data.predictions) return;
  const preds = data.predictions;
  const cur = data.currency || '$';

  // Model info
  const mi = data.model_info;
  if (mi) {
    $('#forecast-model-info').textContent = `${mi.architecture} | Window: ${mi.input_window}d | Horizon: ${mi.prediction_horizon}d`;
  }

  // Table
  let prevPrice = null;
  $('#forecast-tbody').innerHTML = preds.map((p, i) => {
    const change = prevPrice !== null ? ((p.price - prevPrice) / prevPrice * 100) : 0;
    const changeStr = i === 0 ? '—' : `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
    const changeClass = i === 0 ? '' : (change >= 0 ? 'positive' : 'negative');
    prevPrice = p.price;
    return `<tr>
      <td>${p.date}</td>
      <td>${cur}${p.price.toFixed(2)}</td>
      <td class="${changeClass}">${changeStr}</td>
    </tr>`;
  }).join('');

  // Chart
  renderForecastChart(preds, cur);
}

function renderForecastChart(predictions, currency) {
  const canvas = $('#forecast-canvas');
  const empty = $('#forecast-empty');
  empty.style.display = 'none';
  canvas.style.display = 'block';

  const container = $('#forecast-chart-container');
  canvas.width = container.clientWidth;
  canvas.height = 340;

  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const pad = { top: 30, right: 60, bottom: 40, left: 10 };

  ctx.clearRect(0, 0, W, H);

  const prices = predictions.map((p) => p.price);
  const dates = predictions.map((p) => p.date);
  const minP = Math.min(...prices) * 0.998;
  const maxP = Math.max(...prices) * 1.002;
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;

  const xScale = (i) => pad.left + (i / (prices.length - 1)) * chartW;
  const yScale = (v) => pad.top + chartH - ((v - minP) / (maxP - minP)) * chartH;

  // Grid lines
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const val = maxP - ((maxP - minP) / 4) * i;
    ctx.fillStyle = '#555';
    ctx.font = '10px "JetBrains Mono"';
    ctx.textAlign = 'left';
    ctx.fillText(`${currency}${val.toFixed(2)}`, W - pad.right + 6, y + 3);
  }

  // Date labels
  ctx.fillStyle = '#444';
  ctx.font = '10px "JetBrains Mono"';
  ctx.textAlign = 'center';
  dates.forEach((d, i) => {
    if (i % Math.ceil(dates.length / 6) === 0 || i === dates.length - 1) {
      ctx.fillText(d.slice(5), xScale(i), H - 10);
    }
  });

  // Area fill
  const gradient = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
  const isUp = prices[prices.length - 1] >= prices[0];
  gradient.addColorStop(0, isUp ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)');
  gradient.addColorStop(1, 'rgba(10,10,10,0)');
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(prices[0]));
  prices.forEach((p, i) => ctx.lineTo(xScale(i), yScale(p)));
  ctx.lineTo(xScale(prices.length - 1), H - pad.bottom);
  ctx.lineTo(xScale(0), H - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(prices[0]));
  prices.forEach((p, i) => ctx.lineTo(xScale(i), yScale(p)));
  ctx.strokeStyle = isUp ? '#10b981' : '#ef4444';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Dots
  prices.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(xScale(i), yScale(p), 3, 0, Math.PI * 2);
    ctx.fillStyle = isUp ? '#10b981' : '#ef4444';
    ctx.fill();
    ctx.strokeStyle = '#0a0a0a';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

// ── Render: Technicals ──
function renderTechnicals(data) {
  const ind = data.indicators;
  const sig = data.signals;
  const cur = data.currency || '$';

  const cards = [
    { title: 'RSI (14)', value: ind.rsi.toFixed(1), signal: sig.rsi, type: classifySignal(sig.rsi) },
    { title: 'MACD', value: ind.macd.toFixed(4), signal: sig.macd, type: classifySignal(sig.macd) },
    { title: 'MACD Signal', value: ind.macd_signal.toFixed(4), signal: `Histogram: ${ind.macd_histogram.toFixed(4)}`, type: ind.macd_histogram > 0 ? 'bullish' : 'bearish' },
    { title: 'Bollinger Upper', value: `${cur}${ind.bb_upper.toFixed(2)}`, signal: sig.bollinger, type: classifySignal(sig.bollinger) },
    { title: 'Bollinger Middle', value: `${cur}${ind.bb_middle.toFixed(2)}`, signal: `Position: ${(ind.bb_position * 100).toFixed(1)}%`, type: 'neutral-s' },
    { title: 'Bollinger Lower', value: `${cur}${ind.bb_lower.toFixed(2)}`, signal: '', type: 'neutral-s' },
    { title: 'ATR (14)', value: `${cur}${ind.atr.toFixed(2)}`, signal: 'Average True Range', type: 'neutral-s' },
    { title: 'VWAP', value: `${cur}${ind.vwap.toFixed(2)}`, signal: sig.vwap_position, type: classifySignal(sig.vwap_position) },
    { title: 'Volume', value: formatNumber(ind.volume), signal: `${sig.volume} (${ind.volume_ratio}x avg)`, type: 'neutral-s' },
    { title: 'Avg Volume', value: formatNumber(ind.avg_volume), signal: '20-day average', type: 'neutral-s' },
    { title: 'Trend', value: sig.trend, signal: `Price vs BB Middle`, type: sig.trend === 'Uptrend' ? 'bullish' : 'bearish' },
    { title: 'Last Price', value: `${cur}${ind.price.toFixed(2)}`, signal: `${ind.price_change_pct >= 0 ? '+' : ''}${ind.price_change_pct.toFixed(2)}%`, type: ind.price_change_pct >= 0 ? 'bullish' : 'bearish' },
  ];

  const colorMap = { bullish: { bg: 'var(--accent-dim)', color: 'var(--accent-text)' }, bearish: { bg: 'var(--red-dim)', color: 'var(--red-text)' }, 'neutral-s': { bg: 'rgba(136,136,136,0.1)', color: 'var(--text-secondary)' } };

  $('#technicals-grid').innerHTML = cards.map((c) => {
    const cm = colorMap[c.type] || colorMap['neutral-s'];
    return `<div class="tech-card">
      <div class="tech-card-title">${c.title}</div>
      <div class="tech-card-value">${c.value}</div>
      ${c.signal ? `<span class="tech-card-signal" style="background:${cm.bg};color:${cm.color}">${c.signal}</span>` : ''}
    </div>`;
  }).join('');
}

// ── Utilities ──
function formatNumber(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatDate(dateStr) {
  if (!dateStr) return '—';
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch { return dateStr.slice(0, 10); }
}

function confidenceClass(score) {
  if (score >= 0.7) return 'confidence-high';
  if (score >= 0.4) return 'confidence-med';
  return 'confidence-low';
}

function escHtml(str) {
  if (!str) return '';
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
  return str.replace(/[&<>"]/g, (c) => map[c]);
}

function showLoading(text) {
  $('#loading-text').textContent = text || 'Loading...';
  $('#loading-overlay').classList.add('active');
}

function hideLoading() {
  $('#loading-overlay').classList.remove('active');
}

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('#toast-container').appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 200); }, 3500);
}

// Handle window resize for chart
window.addEventListener('resize', () => {
  if (state.forecast?.predictions) {
    renderForecastChart(state.forecast.predictions, state.forecast.currency || '$');
  }
});
