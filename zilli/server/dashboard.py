_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zilli Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}
h1{font-size:24px;margin-bottom:24px;color:#f8fafc}
h2{font-size:16px;margin:20px 0 12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px}
.card{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:16px}
.row{display:flex;gap:20px;flex-wrap:wrap}
.stat{padding:12px 16px;background:#334155;border-radius:8px;min-width:140px}
.stat-label{font-size:12px;color:#94a3b8}
.stat-value{font-size:22px;font-weight:600;margin-top:4px;color:#f8fafc}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.badge-ok{background:#065f46;color:#6ee7b7}
.badge-warn{background:#92400e;color:#fcd34d}
.badge-err{background:#7f1d1d;color:#fca5a5}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;padding:8px 12px;color:#94a3b8;border-bottom:1px solid #334155;font-weight:500}
td{padding:8px 12px;border-bottom:1px solid #1e293b}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.status-dot.ok{background:#22c55e}
.status-dot.err{background:#ef4444}
.refresh{float:right;font-size:12px;color:#64748b;cursor:pointer}
.refresh:hover{color:#f8fafc}
</style>
</head>
<body>
<h1>Zilli Dashboard <span class="refresh" onclick="fetchAll()">↻ refresh</span></h1>

<div class="row" id="stats"></div>

<h2>Models</h2>
<div class="card"><table><thead><tr>
<th>Name</th><th>Model ID</th><th>Role</th><th>Backend</th><th>Status</th>
</tr></thead><tbody id="models-body"></tbody></table></div>

<h2>Cache</h2>
<div class="card" id="cache"></div>

<h2>Active Endpoints</h2>
<div class="card"><table><thead><tr>
<th>Endpoint</th><th>Method</th><th>Description</th>
</tr></thead><tbody>
<tr><td><code>/v1/health</code></td><td>GET</td><td>Health check</td></tr>
<tr><td><code>/v1/route</code></td><td>POST</td><td>Hybrid routing (plan→execute→review)</td></tr>
<tr><td><code>/v1/industry/{type}</code></td><td>POST</td><td>Industry workflow</td></tr>
<tr><td><code>/v1/chat/completions</code></td><td>POST</td><td>OpenAI-compatible chat (streaming support)</td></tr>
<tr><td><code>/v1/models</code></td><td>GET</td><td>OpenAI-compatible model list</td></tr>
<tr><td><code>/v1/cost/status</code></td><td>GET</td><td>Budget status</td></tr>
<tr><td><code>/v1/metrics</code></td><td>GET</td><td>Server metrics</td></tr>
<tr><td><code>/v1/cache/stats</code></td><td>GET</td><td>Cache statistics</td></tr>
<tr><td><code>/v1/cache/clear</code></td><td>POST</td><td>Clear cache</td></tr>
<tr><td><code>/docs</code></td><td>GET</td><td>Swagger UI</td></tr>
</tbody></table></div>

<script>
async function fetchAll() {
  await Promise.all([fetchStats(), fetchModels(), fetchCache()]);
}

async function fetchStats() {
  try {
    const [health, metrics] = await Promise.all([
      fetch('/v1/health').then(r => r.json()),
      fetch('/v1/metrics').then(r => r.json()).catch(() => ({})),
    ]);
    document.getElementById('stats').innerHTML =
      `<div class="stat"><div class="stat-label">Version</div><div class="stat-value">${health.version}</div></div>
       <div class="stat"><div class="stat-label">Models</div><div class="stat-value">${health.models_configured}</div></div>
       <div class="stat"><div class="stat-label">Alive</div><div class="stat-value">${health.models_alive}</div></div>
       <div class="stat"><div class="stat-label">Requests</div><div class="stat-value">${metrics.requests_total || '-'}</div></div>
       <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value">${metrics.tokens_total || '-'}</div></div>
       <div class="stat"><div class="stat-label">Errors</div><div class="stat-value">${metrics.errors_total || '0'}</div></div>`;
  } catch(e) {
    document.getElementById('stats').innerHTML = '<div class="stat"><div class="stat-label">Error</div><div class="stat-value">-</div></div>';
  }
}

async function fetchModels() {
  try {
    const r = await fetch('/v1/models/health');
    const d = await r.json();
    document.getElementById('models-body').innerHTML = d.map(m =>
      `<tr>
        <td>${m.name}</td>
        <td>${m.model_id}</td>
        <td>${m.name}</td>
        <td>ollama</td>
        <td><span class="status-dot ${m.status === 'healthy' ? 'ok' : 'err'}"></span>${m.status}</td>
      </tr>`
    ).join('');
  } catch(e) {}
}

async function fetchCache() {
  document.getElementById('cache').innerHTML = '<p style="color:#64748b">Cache stats available via /v1/cache/stats</p>';
}

fetchAll();
</script>
</body>
</html>"""
