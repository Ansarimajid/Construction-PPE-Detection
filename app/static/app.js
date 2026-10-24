/* PPE Detection Dashboard */
'use strict';

const API = '/api/v1';
let selectedCameraId = null;
let ws = null;
const frameModal = new bootstrap.Modal(document.getElementById('frameModal'));

// ── Health polling ───────────────────────────────────────────────────────────
async function pollHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const badge = document.getElementById('health-badge');
    badge.textContent = d.status === 'ok'
      ? `Online · ${d.cameras_active} camera(s) active`
      : 'Degraded';
    badge.className = d.status === 'ok' ? 'badge bg-success' : 'badge bg-warning';
  } catch {
    const badge = document.getElementById('health-badge');
    badge.textContent = 'Offline';
    badge.className = 'badge bg-danger';
  }
}
setInterval(pollHealth, 5000);
pollHealth();

// ── Camera list ──────────────────────────────────────────────────────────────
async function loadCameras() {
  const r = await fetch(`${API}/cameras`);
  const cameras = await r.json();
  const sel = document.getElementById('camera-select');
  const current = sel.value;
  sel.innerHTML = '<option value="">— Select camera —</option>';
  cameras.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = `${c.name} (${c.source_type}) ${c.is_running ? '🟢' : '⚪'}`;
    sel.appendChild(opt);
  });
  if (current) sel.value = current;
  updateCameraButtons(cameras);
}

function updateCameraButtons(cameras) {
  const sel = document.getElementById('camera-select');
  const id = parseInt(sel.value);
  const cam = cameras.find(c => c.id === id);
  document.getElementById('btn-start').disabled = !cam || cam.is_running;
  document.getElementById('btn-stop').disabled = !cam || !cam.is_running;
}

document.getElementById('camera-select').addEventListener('change', async () => {
  const id = parseInt(document.getElementById('camera-select').value);
  if (!id) return;
  const r = await fetch(`${API}/cameras`);
  const cameras = await r.json();
  updateCameraButtons(cameras);
});

document.getElementById('btn-start').addEventListener('click', async () => {
  const id = parseInt(document.getElementById('camera-select').value);
  if (!id) return;
  await fetch(`${API}/cameras/${id}/start`, { method: 'POST' });
  await loadCameras();
  selectCamera(id);
});

document.getElementById('btn-stop').addEventListener('click', async () => {
  const id = parseInt(document.getElementById('camera-select').value);
  if (!id) return;
  await fetch(`${API}/cameras/${id}/stop`, { method: 'POST' });
  stopStream();
  await loadCameras();
});

// ── Live stream ──────────────────────────────────────────────────────────────
function selectCamera(id) {
  if (ws) { ws.close(); ws = null; }
  selectedCameraId = id;

  const img = document.getElementById('stream-img');
  const placeholder = document.getElementById('stream-placeholder');
  img.src = `${API}/stream/${id}`;
  img.classList.remove('d-none');
  placeholder.classList.add('d-none');

  openWebSocket(id);
}

function stopStream() {
  if (ws) { ws.close(); ws = null; }
  const img = document.getElementById('stream-img');
  img.src = '';
  img.classList.add('d-none');
  document.getElementById('stream-placeholder').classList.remove('d-none');
  ['cnt-hardhat', 'cnt-vest', 'cnt-person', 'cnt-tracked', 'cnt-noncompliant'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });
}

// ── WebSocket for live counts ────────────────────────────────────────────────
function openWebSocket(cameraId) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/api/v1/ws/${cameraId}`);
  ws.onmessage = (event) => {
    try {
      const d = JSON.parse(event.data);
      if (d.ping) return;
      document.getElementById('cnt-hardhat').textContent = d.hardhat_count ?? '—';
      document.getElementById('cnt-vest').textContent = d.vest_count ?? '—';
      document.getElementById('cnt-person').textContent = d.person_count ?? '—';
      document.getElementById('cnt-tracked').textContent = d.unique_person_count ?? '—';
      document.getElementById('cnt-noncompliant').textContent = d.non_compliant_count ?? '—';
    } catch {}
  };
  ws.onerror = () => {};
  ws.onclose = () => {};
}

// ── Add camera form ──────────────────────────────────────────────────────────
document.getElementById('add-camera-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = {
    name: document.getElementById('cam-name').value.trim(),
    source_type: document.getElementById('cam-type').value,
    source_uri: document.getElementById('cam-uri').value.trim(),
  };
  const r = await fetch(`${API}/cameras`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (r.ok) {
    e.target.reset();
    await loadCameras();
  } else {
    alert('Failed to add camera');
  }
});

// ── Violations table ─────────────────────────────────────────────────────────
async function loadViolations() {
  const r = await fetch(`${API}/violations?page_size=50`);
  const data = await r.json();
  const tbody = document.getElementById('violations-tbody');

  if (data.items.length === 0) {
    tbody.innerHTML = '<tr id="no-violations-row"><td colspan="6" class="text-center text-muted py-4">No violations recorded</td></tr>';
    return;
  }

  tbody.innerHTML = data.items.map(v => {
    const ts = new Date(v.timestamp).toLocaleString();
    const frameBtn = v.frame_url
      ? `<button class="btn btn-xs btn-outline-secondary py-0 px-1" onclick="showFrame('${v.frame_url}')">Frame</button>`
      : '';
    const conf = (v.confidence * 100).toFixed(0) + '%';
    const worker = v.track_id != null ? `#${v.track_id}` : '—';
    return `<tr>
      <td class="text-nowrap small">${ts}</td>
      <td>${v.camera_id}</td>
      <td>${worker}</td>
      <td><span class="badge badge-violation">${v.violation_type}</span></td>
      <td>${conf}</td>
      <td>${frameBtn}</td>
    </tr>`;
  }).join('');
}

function showFrame(url) {
  document.getElementById('modal-frame-img').src = url;
  frameModal.show();
}

setInterval(loadViolations, 5000);

// ── Init ─────────────────────────────────────────────────────────────────────
(async () => {
  await loadCameras();
  await loadViolations();
})();
