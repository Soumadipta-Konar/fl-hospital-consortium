// Hospital Consortium Interactive Simulator App
let baselinesChart, byzantineChart, systemsChart, privacyChart, pflChart;

function switchTab(arg1, arg2) {
  let tabId = null;
  let clickedBtn = null;

  if (typeof arg1 === 'string') {
    tabId = arg1;
    if (arg2 instanceof HTMLElement) {
      clickedBtn = arg2;
    }
  } else if (typeof arg2 === 'string') {
    tabId = arg2;
    if (arg1 && (arg1.currentTarget || arg1.target)) {
      clickedBtn = arg1.currentTarget || arg1.target;
    }
  }

  if (!tabId) return;

  // Deactivate all tabs and contents
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  // Activate clicked button or matching button
  if (clickedBtn && clickedBtn.classList) {
    clickedBtn.classList.add('active');
  } else {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      const oc = btn.getAttribute('onclick') || '';
      if (oc.includes(tabId)) {
        btn.classList.add('active');
      }
    });
  }

  // Activate target content
  const targetEl = document.getElementById(tabId);
  if (targetEl) {
    targetEl.classList.add('active');
  }

  // Re-render and resize charts on tab switch to avoid 0px sizing issues
  setTimeout(() => {
    if (tabId === 'tab-baselines' && baselinesChart) baselinesChart.resize();
    if (tabId === 'tab-byzantine' && byzantineChart) byzantineChart.resize();
    if (tabId === 'tab-systems' && systemsChart) systemsChart.resize();
    if (tabId === 'tab-privacy' && privacyChart) privacyChart.resize();
    if (tabId === 'tab-pfl' && pflChart) pflChart.resize();
  }, 20);
}

// Chart.js dark theme defaults - Crisp, solid, zero glassmorphism
Chart.defaults.color = '#B0B0B0';
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.borderColor = '#262626';

Chart.defaults.plugins.legend.labels.color = '#E0E0E0';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.padding = 16;

Chart.defaults.plugins.tooltip.backgroundColor = '#1E1E1E';
Chart.defaults.plugins.tooltip.titleColor = '#E0E0E0';
Chart.defaults.plugins.tooltip.bodyColor = '#B0B0B0';
Chart.defaults.plugins.tooltip.borderColor = '#3D3D3D';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 4;
Chart.defaults.plugins.tooltip.displayColors = true;

// TAB 1: Consortium Baselines
function initBaselinesChart() {
  const ctx = document.getElementById('baselinesChart').getContext('2d');
  const hospitals = [
    'Cleveland (Site 0, N=153)',
    'Hungarian (Site 1, N=147)',
    'Zurich (Site 2, N=42)',
    'VA Long Beach (Site 3, N=100)',
    'Consortium Pooled'
  ];

  baselinesChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hospitals,
      datasets: [
        {
          label: 'Local-Only Model',
          data: [0.750, 0.752, 0.647, 0.700, 0.712],
          backgroundColor: '#606060',
          borderColor: '#757575',
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Federated FedAvg',
          data: [0.773, 0.778, 0.675, 0.591, 0.782],
          backgroundColor: '#3B86D1',
          borderColor: '#305CDE',
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Pooled Upper Bound',
          data: [0.785, 0.792, 0.710, 0.720, 0.796],
          backgroundColor: '#21BF06',
          borderColor: '#1CA305',
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#B0B0B0' }
        },
        y: {
          min: 0.5,
          max: 0.85,
          grid: { color: '#262626' },
          ticks: {
            color: '#B0B0B0',
            callback: v => (v * 100).toFixed(0) + '%'
          }
        }
      }
    }
  });
}

// TAB 2: Byzantine Defense Lab
function updateByzantineSim() {
  const attack = document.getElementById('attackSelect').value;
  const defense = document.getElementById('defenseSelect').value;
  const rounds = Array.from({ length: 50 }, (_, i) => i + 1);

  const cleanData = rounds.map(r => 0.55 + 0.23 * (1 - Math.exp(-r / 10)));
  let simData = [];
  let isUnderperforming = false;

  if (attack === 'none') {
    simData = cleanData;
  } else if (attack === 'garbage') {
    if (defense === 'fedavg') {
      simData = rounds.map(r => 0.54 + 0.05 * Math.sin(r / 3));
      isUnderperforming = true;
    } else {
      simData = rounds.map(r => 0.55 + 0.20 * (1 - Math.exp(-r / 12)) + (Math.random() * 0.008));
    }
  } else if (attack === 'labelflip') {
    if (defense === 'fedavg') {
      simData = rounds.map(r => 0.52 + 0.02 * (1 - Math.exp(-r / 15)));
      isUnderperforming = true;
    } else if (defense === 'median') {
      simData = rounds.map(r => 0.53 + 0.03 * (1 - Math.exp(-r / 15)));
      isUnderperforming = true;
    } else if (defense === 'krum' || defense === 'trimmed_mean') {
      simData = rounds.map(r => 0.55 + 0.16 * (1 - Math.exp(-r / 12)));
    }
  }

  byzantineChart.data.datasets[0].data = cleanData;
  byzantineChart.data.datasets[1].data = simData;
  byzantineChart.data.datasets[1].label = `Simulated: ${attack.toUpperCase()} + ${defense.toUpperCase()}`;

  // Solid color logic: Red if compromised, Blue if defended
  if (isUnderperforming) {
    byzantineChart.data.datasets[1].borderColor = '#E74C3C';
  } else {
    byzantineChart.data.datasets[1].borderColor = '#3B86D1';
  }

  byzantineChart.update();
}

function initByzantineChart() {
  const ctx = document.getElementById('byzantineChart').getContext('2d');
  const rounds = Array.from({ length: 50 }, (_, i) => i + 1);

  byzantineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rounds,
      datasets: [
        {
          label: 'Clean Benchmark (FedAvg)',
          data: [],
          borderColor: '#21BF06',
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0
        },
        {
          label: 'Simulated Scenario',
          data: [],
          borderColor: '#3B86D1',
          borderWidth: 2.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Communication Round', color: '#B0B0B0' },
          grid: { color: '#262626' },
          ticks: { color: '#B0B0B0' }
        },
        y: {
          min: 0.45,
          max: 0.85,
          title: { display: true, text: 'Consortium Accuracy', color: '#B0B0B0' },
          grid: { color: '#262626' },
          ticks: {
            color: '#B0B0B0',
            callback: v => (v * 100).toFixed(0) + '%'
          }
        }
      }
    }
  });

  updateByzantineSim();
}

// TAB 3: Stragglers & FedAsync
function updateSystemsSim() {
  const pDrop = parseFloat(document.getElementById('dropSelect').value);
  const deadline = document.getElementById('deadlineSelect').value;
  const rounds = Array.from({ length: 50 }, (_, i) => i + 1);

  // Synchronous FedAvg curve
  let syncData = [];
  if (deadline === '1.0') {
    syncData = rounds.map(() => 0.437 + Math.random() * 0.02); // Straggler cliff
  } else if (pDrop === 0.5) {
    syncData = rounds.map(r => 0.52 + 0.26 * (1 - Math.exp(-r / 16)));
  } else {
    syncData = rounds.map(r => 0.52 + 0.26 * (1 - Math.exp(-r / 10)));
  }

  // FedAsync curve with staleness damping
  const asyncData = rounds.map(r => 0.52 + 0.25 * (1 - Math.exp(-r / 12)));

  systemsChart.data.datasets[0].data = syncData;
  systemsChart.data.datasets[1].data = asyncData;
  systemsChart.update();
}

function initSystemsChart() {
  const ctx = document.getElementById('systemsChart').getContext('2d');
  const rounds = Array.from({ length: 50 }, (_, i) => i + 1);

  systemsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rounds,
      datasets: [
        {
          label: 'Synchronous FedAvg (Straggler Cliff)',
          data: [],
          borderColor: '#E74C3C',
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: 'FedAsync (Staleness Damped Engine)',
          data: [],
          borderColor: '#3B86D1',
          borderWidth: 2.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Training Elapsed / Rounds', color: '#B0B0B0' },
          grid: { color: '#262626' },
          ticks: { color: '#B0B0B0' }
        },
        y: {
          min: 0.4,
          max: 0.85,
          title: { display: true, text: 'Consortium Accuracy', color: '#B0B0B0' },
          grid: { color: '#262626' },
          ticks: {
            color: '#B0B0B0',
            callback: v => (v * 100).toFixed(0) + '%'
          }
        }
      }
    }
  });

  updateSystemsSim();
}

// TAB 4: Differential Privacy (RDP & Moments Accountant)
function updatePrivacySim() {
  const sigma = parseFloat(document.getElementById('noiseSlider').value);
  const rounds = parseInt(document.getElementById('roundsSlider').value);
  document.getElementById('noiseVal').innerText = sigma.toFixed(1);
  document.getElementById('roundsVal').innerText = rounds;

  const delta = 1e-5;
  const roundsArr = Array.from({ length: rounds }, (_, i) => i + 1);

  // Moments accountant with cross-silo subsampling q=0.5:
  const q = 0.5;
  const epsData = roundsArr.map(r => {
    return (q * r / (2 * sigma * sigma)) + (Math.sqrt(2 * q * r * Math.log(1 / delta)) / (sigma * 2.2));
  });

  const finalEps = epsData[epsData.length - 1];
  const badge = document.getElementById('complianceBadge');
  document.getElementById('kpi-privacy').innerHTML = `&epsilon; = ${finalEps.toFixed(2)}`;

  if (finalEps <= 2.0) {
    badge.className = 'compliance-badge strict';
    badge.innerText = `Strict HIPAA / GDPR Guarantee (ε = ${finalEps.toFixed(2)})`;
  } else if (finalEps <= 8.0) {
    badge.className = 'compliance-badge moderate';
    badge.innerText = `Standard Clinical Compliance (ε = ${finalEps.toFixed(2)})`;
  } else {
    badge.className = 'compliance-badge non-compliant';
    badge.innerText = `Relaxed / Non-Compliant (ε = ${finalEps.toFixed(2)})`;
  }

  privacyChart.data.labels = roundsArr;
  privacyChart.data.datasets[0].data = epsData;
  privacyChart.data.datasets[1].data = roundsArr.map(() => 2.0);
  privacyChart.data.datasets[2].data = roundsArr.map(() => 8.0);
  privacyChart.update();
}

function initPrivacyChart() {
  const ctx = document.getElementById('privacyChart').getContext('2d');
  privacyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Cumulative Privacy Budget (ε)',
          data: [],
          borderColor: '#3B86D1',
          borderWidth: 2.5,
          pointRadius: 0,
          fill: false
        },
        {
          label: 'Strict HIPAA Threshold (ε = 2.0)',
          data: [],
          borderColor: '#21BF06',
          borderWidth: 1.5,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
        },
        {
          label: 'Clinical Compliance Limit (ε = 8.0)',
          data: [],
          borderColor: '#E74C3C',
          borderWidth: 1.5,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Communication Rounds', color: '#B0B0B0' },
          grid: { color: '#262626' },
          ticks: { color: '#B0B0B0' }
        },
        y: {
          min: 0,
          title: { display: true, text: 'Cumulative Privacy Loss (ε)', color: '#B0B0B0' },
          grid: { color: '#262626' },
          ticks: { color: '#B0B0B0' }
        }
      }
    }
  });

  updatePrivacySim();
}

// TAB 5: Personalized FL (pFL)
function initPflChart() {
  const ctx = document.getElementById('pflChart').getContext('2d');
  pflChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [
        'Site 0 (Cleveland)',
        'Site 1 (Hungarian)',
        'Site 2 (Zurich)',
        'Site 3 (VA Long Beach)'
      ],
      datasets: [
        {
          label: 'Global FedAvg',
          data: [0.773, 0.778, 0.675, 0.591],
          backgroundColor: '#E74C3C',
          borderColor: '#C0392B',
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Ditto Personalization (λ=0.1)',
          data: [0.781, 0.782, 0.714, 0.710],
          backgroundColor: '#3B86D1',
          borderColor: '#305CDE',
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'FedPer (Private Classification Head)',
          data: [0.779, 0.785, 0.720, 0.700],
          backgroundColor: '#21BF06',
          borderColor: '#1CA305',
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#B0B0B0' }
        },
        y: {
          min: 0.5,
          max: 0.85,
          grid: { color: '#262626' },
          ticks: {
            color: '#B0B0B0',
            callback: v => (v * 100).toFixed(0) + '%'
          }
        }
      }
    }
  });
}

// Initialization on DOM Ready
window.addEventListener('DOMContentLoaded', () => {
  initBaselinesChart();
  initByzantineChart();
  initSystemsChart();
  initPrivacyChart();
  initPflChart();
});
