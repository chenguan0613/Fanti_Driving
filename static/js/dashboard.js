let lastLogKey = "";
const riskHistory = [];
const maxRiskPoints = 90;
const thresholdValue = 70;

function getLogClass(status) {
  if (status === "FATIGUE WARNING") return "log-warning";
  if (status === "Safe") return "log-safe";
  return "log-neutral";
}

function addLogEntry(status, reason, probability, perclos) {
  const logEl = document.getElementById("event-log");
  const entry = document.createElement("div");
  entry.className = `log-entry ${getLogClass(status)}`;
  entry.innerHTML = `<div class="log-topline"><span class="log-status">${status}</span><span class="log-time">${new Date().toLocaleTimeString()}</span></div><div class="log-reason">${reason}</div><div class="log-metrics">AI ${probability}% | PERCLOS ${perclos}</div>`;
  logEl.prepend(entry);
  while (logEl.children.length > 8) logEl.removeChild(logEl.lastChild);
}

function addRiskPoint(value) {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) return;
  riskHistory.push(Math.max(0, Math.min(100, parsed)));
  while (riskHistory.length > maxRiskPoints) riskHistory.shift();
  drawRiskChart();
}

function drawRiskChart() {
  const canvas = document.getElementById("risk-chart");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = rect.width;
  const height = rect.height;
  const padL = 42,
    padR = 14,
    padT = 16,
    padB = 28;
  const chartW = width - padL - padR;
  const chartH = height - padT - padB;
  const y = (value) => padT + chartH - (value / 100) * chartH;
  const x = (index) =>
    padL +
    (riskHistory.length <= 1
      ? chartW
      : (index / (riskHistory.length - 1)) * chartW);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#171729";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].forEach((value) => {
    ctx.beginPath();
    ctx.moveTo(padL, y(value));
    ctx.lineTo(width - padR, y(value));
    ctx.stroke();
    ctx.fillStyle = "#7f829f";
    ctx.font = "12px Courier New";
    ctx.fillText(`${value}`, 8, y(value) + 4);
  });

  ctx.setLineDash([7, 6]);
  ctx.strokeStyle = "#ff1744";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padL, y(thresholdValue));
  ctx.lineTo(width - padR, y(thresholdValue));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ff6b80";
  ctx.font = "12px Segoe UI";
  ctx.fillText(
    `${thresholdValue}% warning`,
    padL + 8,
    y(thresholdValue) - 8,
  );

  if (riskHistory.length > 0) {
    const gradient = ctx.createLinearGradient(
      0,
      padT,
      0,
      height - padB,
    );
    gradient.addColorStop(0, "rgba(0,229,255,0.28)");
    gradient.addColorStop(1, "rgba(0,229,255,0.02)");
    ctx.beginPath();
    ctx.moveTo(x(0), y(riskHistory[0]));
    riskHistory.forEach((value, index) =>
      ctx.lineTo(x(index), y(value)),
    );
    ctx.lineTo(x(riskHistory.length - 1), height - padB);
    ctx.lineTo(x(0), height - padB);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(x(0), y(riskHistory[0]));
    riskHistory.forEach((value, index) =>
      ctx.lineTo(x(index), y(value)),
    );
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 3;
    ctx.stroke();

    const latest = riskHistory[riskHistory.length - 1];
    ctx.beginPath();
    ctx.arc(x(riskHistory.length - 1), y(latest), 4, 0, Math.PI * 2);
    ctx.fillStyle = latest >= thresholdValue ? "#ff1744" : "#00e5ff";
    ctx.fill();
  }
}

async function fetchSystemStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    const reason = data.reason || "Waiting for monitoring data";
    const statusEl = document.getElementById("status-text");
    document.getElementById("prob-text").innerText =
      data.fatigue_prob + "%";
    document.getElementById("perclos-text").innerText = data.perclos;
    statusEl.innerText = data.status;
    statusEl.className = "stat-value";
    if (data.status === "FATIGUE WARNING")
      statusEl.classList.add("status-warning");
    else if (data.status === "Safe")
      statusEl.classList.add("status-safe");
    else statusEl.classList.add("status-neutral");
    addRiskPoint(data.fatigue_prob);
    const logKey = `${data.status}|${reason}`;
    if (logKey !== lastLogKey) {
      addLogEntry(data.status, reason, data.fatigue_prob, data.perclos);
      lastLogKey = logKey;
    }
  } catch (error) {
    console.error("Unable to connect to monitoring core:", error);
  }
}

function setLastWarningThreshold() {
  panel = document.getElementById("last-warning-threshold");
  legend = document.getElementById("threshold-legend");
  if (!panel) {
    return;
  }
  if (!legend) {
    return;
  }
  panel.innerText = `Last 90 seconds, warning threshold at ${thresholdValue}%`;
  legend.innerText = `${thresholdValue}% threshold`;
}

window.addEventListener("resize", drawRiskChart);
setInterval(fetchSystemStatus, 1000);
fetchSystemStatus();
setLastWarningThreshold();
