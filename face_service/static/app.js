"use strict";

const cards = new Map([...document.querySelectorAll(".camera-card")]
  .map(card => [card.dataset.source, card]));
const systemStatus = document.querySelector("#system-status");
const grid = document.querySelector("#camera-grid");
let mode = "dual";
let running = !document.hidden;
let statusTimer = null;

function isVisible(source) { return mode === "dual" || mode === source; }

function scheduleFrame(source, delay = 160) {
  if (!running || !isVisible(source)) return;
  const card = cards.get(source);
  const image = card.querySelector("img");
  window.setTimeout(() => {
    if (running && isVisible(source)) {
      image.src = `/debug/frame.jpg?source=${encodeURIComponent(source)}&t=${Date.now()}`;
    }
  }, delay);
}

for (const [source, card] of cards) {
  const image = card.querySelector("img");
  image.addEventListener("load", () => {
    card.classList.add("live");
    scheduleFrame(source);
  });
  image.addEventListener("error", () => {
    card.classList.remove("live");
    scheduleFrame(source, 800);
  });
}

function draw(card, sourceState) {
  const canvas = card.querySelector("canvas");
  const bounds = card.querySelector(".viewport").getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * scale);
  canvas.height = Math.round(bounds.height * scale);
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  context.lineWidth = 2;
  context.font = "600 13px ui-monospace, monospace";
  for (const track of sourceState.tracks || []) {
    const [x, y, width, height] = track.bbox;
    const px = x * bounds.width, py = y * bounds.height;
    const pw = width * bounds.width, ph = height * bounds.height;
    const color = track.state === "confirmed" ? "#71f6c1" : "#ffc76b";
    context.strokeStyle = color;
    context.fillStyle = "rgba(4,12,16,.82)";
    context.strokeRect(px, py, pw, ph);
    const label = `${track.track_id}  ${Math.round(track.score * 100)}%`;
    const labelWidth = context.measureText(label).width + 12;
    const labelY = Math.max(0, py - 25);
    context.fillRect(px, labelY, labelWidth, 23);
    context.fillStyle = color;
    context.fillText(label, px + 6, labelY + 16);
  }
  const badge = card.querySelector(".face-badge");
  badge.textContent = sourceState.ready ? `${sourceState.face_count} 张已确认` : "画面不可用";
  badge.classList.toggle("active", sourceState.face_count > 0);
  const metrics = card.querySelectorAll(".metrics span");
  metrics[0].textContent = `序列 ${sourceState.sequence || "—"}`;
  metrics[1].textContent = `延迟 ${sourceState.latency_ms == null ? "—" : sourceState.latency_ms + " ms"}`;
}

async function refreshStatus() {
  if (!running) return;
  try {
    const response = await fetch("/status", {cache: "no-store"});
    const state = await response.json();
    systemStatus.classList.toggle("ready", state.ready);
    systemStatus.lastChild.textContent = state.ready ? "后台感知运行中" : "感知服务异常";
    for (const sourceState of state.sources || []) {
      const card = cards.get(sourceState.source);
      if (card) draw(card, sourceState);
    }
  } catch (_error) {
    systemStatus.classList.remove("ready");
    systemStatus.lastChild.textContent = "连接中断";
  }
  statusTimer = window.setTimeout(refreshStatus, 300);
}

function applyMode(nextMode) {
  mode = nextMode;
  grid.classList.toggle("dual", mode === "dual");
  for (const [source, card] of cards) {
    const visible = isVisible(source);
    card.classList.toggle("hidden", !visible);
    if (visible && running) scheduleFrame(source, 0);
    if (!visible) card.querySelector("img").removeAttribute("src");
  }
}

for (const button of document.querySelectorAll("nav button")) {
  button.addEventListener("click", () => {
    document.querySelector("nav button.active").classList.remove("active");
    button.classList.add("active");
    applyMode(button.dataset.mode);
  });
}

document.addEventListener("visibilitychange", () => {
  running = !document.hidden;
  window.clearTimeout(statusTimer);
  if (running) {
    applyMode(mode);
    refreshStatus();
  } else {
    for (const card of cards.values()) card.querySelector("img").removeAttribute("src");
  }
});

applyMode("dual");
refreshStatus();
