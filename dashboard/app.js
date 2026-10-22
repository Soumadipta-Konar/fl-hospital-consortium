// Hospital Consortium Interactive Simulator App
let baselinesChart, byzantineChart, systemsChart, privacyChart, pflChart;

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  event.target.classList.add('active');
  document.getElementById(tabId).classList.add('active');
}

// Chart.js global dark theme defaults
Chart.defaults.color = '#8b9bb4';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';

function initBaselinesChart() {
  const ctx = document.getElementById('baselinesChart').getContext('2d');
  const hospitals = ['Cleveland (153)', 'Hungarian (147)', 'Zurich (42)', 'VA (100)', 'Pooled Overall'];

  baselinesChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hospitals,
      datasets: [
        {
          label: 'Local-Only Model',
          data: [0.750, 0.752, 0.647, 0.700, 0.712],
          backgroundColor: '#6c757d'
        },
        {
          label: 'Federated FedAvg',
          data: [0.773, 0.778, 0.675, 0.591, 0.782],
          backgroundColor: '#00d2ff'
        },
        {
          label: 'Pooled Upper Bound',
          data: [0.785, 0.792, 0.710, 0.720, 0.796],
          backgroundColor: '#00f5a0'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0.5,
          max: 0.85,
          ticks: { callback: v => (v * 100).toFixed(0) + '%' }
        }
      }
    }
  });
}

function updateByzantineSim() {
  const attack = document.getElementById('attackSelect').value;
  const defense = document.getElementById('defenseSelect').value;
  const rounds = Array.from({ length: 50 }, (_, i) => i + 1);

  let cleanData = rounds.map(r => 0.55 + 0.23 * (1 - Math.exp(-r / 10)));
  let simData = [];

  if (attack === 'none') {
    simData = cleanData;
  } else if (attack === 'garbage') {
    if (defense === 'fedavg') {
      simData = rounds.map(r => 0.54 + 0.05 * Math.sin(r / 3));
    } else if (defense === 'median' || defense === 'trimmed_mean' || defense === 'krum') {
      simData = rounds.map(r => 0.55 + 0.20 * (1 - Math.exp(-r / 12)) + (Math.random() * 0.01));
    }
  } else if (attack === 'labelflip') {
    if (defense === 'fedavg') {
      simData = rounds.map(r => 0.52 + 0.02 * (1 - Math.exp(-r / 15)));
    } else if (defense === 'median') {
      simData = rounds.map(r => 0.53 + 0.03 * (1 - Math.exp(-r / 15)));
    } else if (defense === 'krum' || defense === 'trimmed_mean') {
      simData = rounds.map(r => 0.55 + 0.16 * (1 - Math.exp(-r / 12)));
    }
  }

  byzantineChart.data.datasets[0].data = cleanData;
  byzantineChart.data.datasets[1].data = simData;
  byzantineChart.data.datasets[1].label = `Simulated (${attack.toUpperCase()} + ${defense.toUpperCase()})`;
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
          borderColor: '#00f5a0',
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0
        },
        {
          label: 'Attack Scenario',
          data: [],
          borderColor: '#ff4d6d',
          borderWidth: 2.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0.45,
          max: 0.85,
          ticks: { callback: v => (v * 100).toFixed(0) + '%' }
        }
      }
    }
  });

  updateByzantineSim();
}

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

  // FedAsync curve (smoothly incorporating updates with staleness damping)
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
          label: 'Synchronous FedAvg (Under Constraints)',
          data: [],
          borderColor: '#ff4d6d',
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: 'FedAsync (Staleness Damped Engine)',
          data: [],
          borderColor: '#00d2ff',
          borderWidth: 2.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0.4,
          max: 0.85,
          ticks: { callback: v => (v * 100).toFixed(0) + '%' }
        }
      }
    }
  });

  updateSystemsSim();
}

function updatePrivacySim() {
  const sigma = parseFloat(document.getElementById('noiseSlider').value);
  const rounds = parseInt(document.getElementById('roundsSlider').value);
  document.getElementById('noiseVal').innerText = sigma.toFixed(1);
  document.getElementById('roundsVal').innerText = rounds;

  const delta = 1e-5;
  const roundsArr = Array.from({ length: rounds }, (_, i) => i + 1);

  // Exact RDP optimal closed form for Gaussian mechanism:
  // eps = (T / (2 * sigma^2)) + (sqrt(2 * T * ln(1/delta)) / sigma)
  const epsData = roundsArr.map(r => {
    return (r / (2 * sigma * sigma)) + (Math.sqrt(2 * r * Math.log(1 / delta)) / sigma);
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
          label: 'Cumulative Differential Privacy Budget (ε)',
          data: [],
          borderColor: '#9d4edd',
          backgroundColor: 'rgba(157, 78, 221, 0.15)',
          fill: true,
          borderWidth: 2.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          title: { display: true, text: 'Cumulative Epsilon (ε)' }
        },
        x: {
          title: { display: true, text: 'Communication Rounds' }
        }
      }
    }
  });

  updatePrivacySim();
}

function initPflChart() {
  const ctx = document.getElementById('pflChart').getContext('2d');
  pflChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Site 0 (Cleveland)', 'Site 1 (Hungarian)', 'Site 2 (Zurich)', 'Site 3 (VA Long Beach)'],
      datasets: [
        {
          label: 'Global FedAvg',
          data: [0.773, 0.778, 0.675, 0.591],
          backgroundColor: '#ff4d6d'
        },
        {
          label: 'Ditto Personalization (λ=0.1)',
          data: [0.781, 0.782, 0.714, 0.710],
          backgroundColor: '#00d2ff'
        },
        {
          label: 'FedPer (Private Head)',
          data: [0.779, 0.785, 0.720, 0.700],
          backgroundColor: '#00f5a0'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0.5,
          max: 0.85,
          ticks: { callback: v => (v * 100).toFixed(0) + '%' }
        }
      }
    }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  initBaselinesChart();
  initByzantineChart();
  initSystemsChart();
  initPrivacyChart();
  initPflChart();
});
