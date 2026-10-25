// app.js

const $ = id => document.getElementById(id);
const pages = ['overview', 'monitoring', 'twin', 'predictions', 'simulation', 'settings'];
const titles = {
  overview: 'Overview',
  monitoring: 'Live Monitoring',
  twin: 'Digital Twin',
  predictions: 'Predictions & MADTP',
  simulation: 'Simulation Lab',
  settings: 'Settings'
};

let chart;
let liveTimer = null;
let liveHistory = [];
const defaultApi = 'http://127.0.0.1:8000';

// API URL Resolution
// API URL Resolution
function apiBase() {
  return ($('apiUrl')?.value || localStorage.getItem('madtpApiUrl') || defaultApi).trim().replace(/\/$/, '');
}

// Safe DOM text updater
function setText(id, value) {
  const el = $(id);
  if (el && value !== null && value !== undefined && value !== '') {
    el.textContent = value;
  }
}

// Number formatter
function format(value, digits = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

// Main live data mapper
function updateLiveView(d) {
  if (!d) return;

  // 1. Overview Tab
  setText('heartRate', format(d.heart_rate, 0));
  setText('spo2', format(d.spo2, 0));
  setText('bodyTemp', format(d.ds18b20_surface_temperature_c, 1));
  setText('ambientTemp', format(d.dht22_ambient_temperature_c, 1));
  setText('humidity', `Humidity ${format(d.dht22_humidity_percent, 1)}%`);
  setText('riskScore', format(d.current_risk_score, 1));
  
  if (typeof d.current_risk_score === 'number' && $('riskProgress')) {$('riskProgress').style.width = `${Math.max(0, Math.min(100, d.current_risk_score))}%`;
  }
  
  setText('riskBadge', d.risk_category || 'UNKNOWN');
  setText('divergence', format(d.twin_divergence_index, 4));
  setText('clock', d.received_utc ? new Date(d.received_utc).toLocaleString() : 'Live data');
  setText('overviewBaseline', 'Generated');
  setText('twinSyncText', 'Synchronised to Python ML Engine');

  if (d.risk_category === 'HIGH' || d.risk_category === 'CRITICAL') {
    setText('recommendation', 'High prototype risk score: follow your established safety procedure.');
  } else {
    setText('recommendation', 'Continue monitoring; this prototype score is not a medical diagnosis.');
  }

  // 2. Live Monitoring Tab (Includes MPU6050 Mapping)
  setText('tableHr', `${format(d.heart_rate, 0)} BPM`);
  setText('tableSpo2', `${format(d.spo2, 0)}%`);
  setText('tableBodyTemp', `${format(d.ds18b20_surface_temperature_c, 1)} °C`);
  setText('tableAmbient', `${format(d.dht22_ambient_temperature_c, 1)} °C`);
  setText('tableHumidity', `${format(d.dht22_humidity_percent, 1)}%`);
  setText('tableMotion', `Acc [${format(d.mpu6050_accel_x, 2)}, ${format(d.mpu6050_accel_y, 2)}, ${format(d.mpu6050_accel_z, 2)}]`);

  // 3. Digital Twin Tab
  setText('twinHr', `${format(d.heart_rate, 0)} BPM`);
  setText('twinExpected', 'Active Profile');
  setText('twinDivergenceVal', format(d.twin_divergence_index, 4));
  setText('twinHeroSync', 'Synchronised to live hardware stream');

  // 4. Predictions & MADTP Tab (Random Forest integration)
  setText('predScore', format(d.current_risk_score, 1));
  
  let rfProb = typeof d.rf_deterioration_probability === 'number' ? (d.rf_deterioration_probability * 100) : null;
  setText('predRisk', format(rfProb, 1));
  
  if (d.rf_prediction == 0) {
    setText('predStatus', 'Normal (Class 0)');
  } else if (d.rf_prediction == 1) {
    setText('predStatus', 'Deteriorating (Class 1)');
  }

  // 5. Global UI Polish
  document.querySelectorAll('.state.pending').forEach(el => {
    el.textContent = 'Live';
    el.style.color = '#34d399'; // Optional: makes the text green when live
  });
}

// API Poller
async function pollLive() {
  try {
    const [latestResponse, historyResponse] = await Promise.all([
      fetch(`${apiBase()}/api/latest`, { cache: 'no-store' }),
      fetch(`${apiBase()}/api/history`, { cache: 'no-store' })
    ]);
    
    if (!latestResponse.ok || !historyResponse.ok) {
      throw new Error('API request failed');
    }
    
    const latest = await latestResponse.json();
    const history = await historyResponse.json();
    
    if (!latest.data) {
      if ($('settingsMessage'))$('settingsMessage').textContent = 'API connected; waiting for live prediction rows.';
      return;
    }
    
    liveHistory = Array.isArray(history.data) ? history.data : [];
    updateLiveView(latest.data);
    renderChart();
    
    if ($('settingsMessage'))$('settingsMessage').textContent = 'Connected to Python prediction results.';
  } catch (err) {
    if ($('settingsMessage'))$('settingsMessage').textContent = `API unavailable: ${err.message}. Start api_server.py and live_madtp.py.`;
  }
}

// Polling Controller
function setLiveMode(enabled) {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
  }
  if (enabled) {
    pollLive();
    liveTimer = setInterval(pollLive, 2500);
  }
}

// Navigation
function showPage(page) {
  if (!pages.includes(page)) return;
  document.querySelectorAll('.page').forEach(x => x.classList.toggle('active', x.id === `page-${page}`));
  document.querySelectorAll('[data-page]').forEach(x => x.classList.toggle('active', x.dataset.page === page));
  if ($('pageTitle'))$('pageTitle').textContent = titles[page];
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// App Initialization
function openApp() {
  if ($('loginScreen'))$('loginScreen').classList.add('hidden');
  if ($('app'))$('app').classList.remove('hidden');
  if ($('clock'))$('clock').textContent = new Date().toLocaleString();
  
  // Instantly start pulling live data from the API
  setLiveMode(true);
  renderChart();
}

// Charting Logic
function renderChart() {
  if (!window.Chart || !$('trendChart')) return;
  if (chart) chart.destroy();
  
  const hasLive = liveHistory.length > 0;
  const rows = hasLive ? liveHistory.slice(-30) : [];
  
  const labels = hasLive 
    ? rows.map((r, i) => r.received_utc ? new Date(r.received_utc).toLocaleTimeString() : String(i + 1)) 
    : [];
    
  const hr = hasLive ? rows.map(r => r.heart_rate) : [];
  const temp = hasLive ? rows.map(r => r.ds18b20_surface_temperature_c) : [];
    
  chart = new Chart($('trendChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { 
          label: 'Heart rate (BPM)', 
          data: hr, 
          borderColor: '#607cf2', 
          backgroundColor: '#607cf21a', 
          fill: true, 
          tension: .35, 
          pointRadius: 0, 
          borderWidth: 2 
        },
        { 
          label: 'Temperature (°C)', 
          data: temp, 
          borderColor: '#eea45c', 
          tension: .35, 
          pointRadius: 0, 
          borderWidth: 2 
        }
      ]
    },
    options: { 
      responsive: true, 
      maintainAspectRatio: false, 
      plugins: { 
        legend: { labels: { usePointStyle: true, boxWidth: 7, font: { size: 10 } } } 
      }, 
      scales: { 
        x: { grid: { display: false } }, 
        y: { grid: { color: '#eef1f6' } } 
      } 
    }
  });
}

// Event Listeners
if ($('loginForm')) {$('loginForm').addEventListener('submit', e => { 
    e.preventDefault(); 
    openApp(); 
  });
}

if ($('mainNav')) {$('mainNav').addEventListener('click', e => { 
    const b = e.target.closest('[data-page]'); 
    if (b) showPage(b.dataset.page); 
  });
}

document.querySelectorAll('[data-goto]').forEach(b => {
  b.addEventListener('click', () => showPage(b.dataset.goto));
});

if ($('signOutBtn')) {$('signOutBtn').addEventListener('click', () => { 
    $('app').classList.add('hidden');$('loginScreen').classList.remove('hidden'); 
    
    // Stop polling when signed out
    setLiveMode(false);
  });
}

if ($('searchBtn')) {
  $('searchBtn').addEventListener('click', () => {$('searchRow').classList.toggle('hidden'); 
    if (!$('searchRow').classList.contains('hidden'))$('globalSearch').focus(); 
  });
}

if ($('closeSearch')) {
  $('closeSearch').addEventListener('click', () =>$('searchRow').classList.add('hidden'));
}

if ($('globalSearch')) {$('globalSearch').addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll('.search-highlight').forEach(x => x.classList.remove('search-highlight'));
    const found = [...document.querySelectorAll('.page.active h1,.page.active h2,.page.active h3,.page.active td,.page.active .metric-card,.page.active .panel')].find(x => x.textContent.toLowerCase().includes(q));
    if (found) { 
      found.classList.add('search-highlight'); 
      found.scrollIntoView({ behavior: 'smooth', block: 'center' }); 
    } else {
      alert('No matching item found on this page.');
    }
  });
}

// Settings initialization
if ($('apiUrl')) {$('apiUrl').value = localStorage.getItem('madtpApiUrl') || defaultApi;
}

if ($('saveSettings')) {$('saveSettings').addEventListener('click', () => {
    const url = $('apiUrl').value.trim();
    if (!url) { 
      $('settingsMessage').textContent = 'Enter an API URL first.'; 
      return; 
    }
    localStorage.setItem('madtpApiUrl', url);
    $('settingsMessage').textContent = 'API URL saved.';
    
    // Always enforce live mode since there's no demo toggle anymore
    setLiveMode(true);
  });
}

// Cleanup on unload
window.addEventListener('beforeunload', () => { 
  if (liveTimer) clearInterval(liveTimer); 
});