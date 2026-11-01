/* Local presentation only. One existing state GET / 5s; no audio/media/model calls. */
(() => {
  const buddy = document.getElementById('home-buddy');
  const caption = document.getElementById('home-status');
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const postureState = document.getElementById('home-posture-state');
  const postureInfo = document.getElementById('home-posture-info');
  const postureLabels = {healthy: '坐姿在校准范围内', suspect: '正在确认坐姿变化',
    unhealthy: '检测到持续偏离', calibrating: '正在校准，请保持坐直',
    observing: '正在观察，尚需校准', unknown: '暂时看不清，不能判断坐姿',
    model_unavailable: '姿态模型暂未就绪', paused: '守护已暂停'};
  let postureController;
  const phases = {
    idle: ['curious', '嗯？你有个好主意？'],
    listening: ['listening', '我在听，慢慢说。'],
    thinking: ['listening', '让我想一想。'],
    speaking: ['curious', '我正在说话。'],
    attention: ['listening', '有一份提醒。'],
    acting: ['listening', '正在帮你处理。'],
    success: ['amused', '好啦！'],
    error: ['listening', '遇到一点问题，稍后再试。'],
  };
  let pollTimer, blinkTimer, blinkEnd, nudgeEnd, reactionTimer, controller, requestTimeout;
  let generation = 0, away = false, reacting = false, blinking = false, taps = 0;
  let phase = 'idle', baseText = phases.idle[1], connected = true;
  const active = () => !away && !document.hidden;
  async function refreshPosture(token) {
    postureController?.abort();
    const current = new AbortController(); postureController = current;
    const deadline = setTimeout(() => current.abort(), 2000);
    try {
      const response = await fetch('/api/posture/status', {cache: 'no-store', signal: current.signal});
      if (!response.ok) throw Error('posture unavailable');
      const value = await response.json();
      if (!active() || token !== generation) return;
      if (typeof value.enabled !== 'boolean') throw Error('invalid posture state');
      postureState.textContent = value.simulation ? '模拟测试中' : value.enabled ? '已开启' : '已暂停';
      const stale = !Number.isFinite(value.frame_age_ms) || value.frame_age_ms > 5000 ||
        !Number.isFinite(value.observation_age_ms) || value.observation_age_ms > 5000;
      postureInfo.textContent = !value.enabled ? '后台检测未开启；点击下方入口查看或开启。' :
        value.simulation ? '当前为合成演示，不代表真实姿势检测。' :
        (stale ? '已开启，但检测画面尚未更新。' : (postureLabels[value.state] || '状态未知，暂不能判断。')) +
        (value.calibrated ? value.low_head_calibrated===false?' 请补充低头校准。':' 已校准。' : ' 尚未校准。');
    } catch (_) {
      if (active() && token === generation) {
        postureState.textContent = '状态未知';
        postureInfo.textContent = '连接失败，无法确认后台是否开启；不会当作已关闭。';
      }
    } finally { clearTimeout(deadline); if (postureController === current) postureController = null; }
  }
  function render() {
    buddy.dataset.phase = phase;
    buddy.dataset.mood = reacting || blinking ? 'amused' : (phases[phase] || phases.idle)[0];
    buddy.dataset.reaction = String(reacting);
    // A touch never hides an active task cue or an honest connection warning.
    const text = reacting && phase === 'idle' && connected
      ? ['嘿，碰到我啦！', '有点痒，哈哈！', '我在呢。一起玩吧！'][(taps - 1) % 3]
      : baseText;
    if (caption.textContent !== text) caption.textContent = text;
  }
  function scheduleBlink() {
    clearTimeout(blinkTimer);
    if (!active() || motion.matches) return;
    blinkTimer = setTimeout(() => {
      if (!reacting && phase === 'idle' && connected) {
        buddy.dataset.nudge = 'true';
        nudgeEnd = setTimeout(() => { buddy.dataset.nudge = 'false'; }, 650);
        blinking = true; render();
        blinkEnd = setTimeout(() => { blinking = false; render(); }, 180);
      }
      scheduleBlink();
    }, 9000);
  }
  buddy.addEventListener('click', () => {
    if (!active()) return;
    clearTimeout(reactionTimer); clearTimeout(blinkEnd); clearTimeout(nudgeEnd);
    buddy.dataset.nudge = 'false';
    blinking = false; reacting = true; taps += 1; render();
    reactionTimer = setTimeout(() => { reacting = false; render(); }, 1800);
  });
  async function refresh() {
    clearTimeout(pollTimer);
    if (!active()) return;
    const token = ++generation;
    refreshPosture(token);
    controller = new AbortController();
    const current = controller;
    requestTimeout = setTimeout(() => current.abort(), 2000);
    try {
      const response = await fetch('/api/state', {cache: 'no-store', signal: current.signal});
      if (!response.ok) throw Error('state unavailable');
      const state = await response.json();
      if (!active() || token !== generation) return;
      connected = true;
      phase = Object.hasOwn(phases, state.phase) ? state.phase : 'idle';
      // Never put conversation/history/profile text on the shared home screen.
      const taskCue = state.producer === 'task' && ['attention', 'speaking', 'acting', 'success', 'error'].includes(phase);
      baseText = taskCue ? String(state.primary_text || '有一份提醒。').slice(0, 160) : phases[phase][1];
      render();
    } catch (_) {
      if (!active() || token !== generation) return;
      connected = false; phase = 'idle';
      baseText = '状态暂未连接；你仍可以选择模式。'; render();
    } finally {
      if (token === generation) {
        clearTimeout(requestTimeout); controller = null;
        if (active()) pollTimer = setTimeout(refresh, 5000);
      }
    }
  }
  function pause() {
    generation += 1;
    [pollTimer, blinkTimer, blinkEnd, nudgeEnd, reactionTimer, requestTimeout].forEach(clearTimeout);
    buddy.dataset.nudge = 'false';
    controller?.abort(); controller = null;
    postureController?.abort(); postureController = null;
    reacting = false; blinking = false; render();
    document.documentElement.dataset.homePaused = 'true';
  }
  function resume() {
    if (!active()) return;
    document.documentElement.dataset.homePaused = 'false';
    refresh(); scheduleBlink();
  }
  document.addEventListener('visibilitychange', () => { pause(); if (!document.hidden) resume(); });
  window.addEventListener('pagehide', () => { away = true; pause(); });
  window.addEventListener('pageshow', event => { if (event.persisted) { away = false; pause(); resume(); } });
  motion.addEventListener('change', () => {
    clearTimeout(blinkEnd); clearTimeout(nudgeEnd); buddy.dataset.nudge = 'false';
    blinking = false; render(); scheduleBlink();
  });
  resume();
})();
