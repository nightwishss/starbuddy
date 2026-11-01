/* Status GETs never acquire/renew a camera lease. Only an explicit click starts it. */
(() => {
  let panel = document.querySelector('#home-wave');
  if (!panel) {
    const anchor = document.querySelector('#buddy-hint, .companion-heading, .stage-top');
    if (!anchor) return;
    panel = document.createElement('section');
    panel.className = 'wave-panel'; panel.setAttribute('aria-label', '挥手互动');
    panel.innerHTML = '<strong id="wave-state" role="status">正在读取状态</strong>' +
      '<button type="button" id="wave-toggle" disabled aria-pressed="false">正在连接</button>' +
      '<p id="wave-info" role="status"></p><p id="wave-detail"></p>';
    anchor.insertAdjacentElement('afterend', panel);
  }
  const button = panel.querySelector('#wave-toggle'), info = panel.querySelector('#wave-info');
  const state = panel.querySelector('#wave-state'), detail = panel.querySelector('#wave-detail');
  const reasons = {
    off: '尚未开启', service_started: '服务启动后默认关闭，需手动开启',
    manual_stop: '你已手动关闭', page_hidden: '页面切到后台，已自动关闭',
    page_left: '页面已离开或刷新，已自动关闭', navigation: '切换页面，已自动关闭',
    user_interaction: '开始其他交互，已自动关闭', heartbeat_failed: '页面续期请求失败，已安全关闭',
    lease_expired: '页面心跳超过 8 秒未续期，已自动关闭',
    worker_exited: '检测进程已退出', worker_crashed: '检测进程异常退出',
    worker_timeout: '检测进程超过 15 秒未更新', worker_invalid_output: '检测进程返回异常数据',
    worker_start_failed: '检测进程启动失败', worker_io_error: '检测进程通信失败',
    worker_error: '检测进程发生错误', service_shutdown: '服务停止，挥手互动已关闭',
    fusion_or_output_unavailable: '仲裁或语音提交失败，已安全关闭',
    window_closed_or_worker_stopped: '旧版未区分页面失联与进程退出',
  };
  const decisions = {
    cooldown: '冷却中，间隔约 20 秒再挥手', playing: '语音正在播放',
    output_completed: '语音已完成', output_cancelled: '语音被取消',
    output_interrupted: '语音被打断', output_expired: '语音已过期',
    output_failed: '语音播放失败', output_cleanup_unknown: '输出清理尚未确认',
    user_turn_active: '用户交互优先', output_busy: '其他语音正在输出',
    ambient_reserved: '另一个主动行为已预约', quiet: '当前安静模式',
    greet_named: '挥手问候已通过仲裁', greet_generic: '挥手问候已通过仲裁',
    continue_engagement: '继续简短回应', admitted: '已通过仲裁',
    fusion_or_output_unavailable: '仲裁或输出暂不可用',
  };
  const handGuidance = {
    no_single_pose: '暂未看清单人上半身；请让头部和双肩进入画面。',
    upper_body_uncertain: '上半身还不够清楚；请让头部和双肩完整进入画面。',
    shoulder_scale: '请调整与摄像头的距离，让上半身清楚地出现在画面中。',
    no_hands: '还没看到手；请举起一只手，让手掌完整进入画面。',
    hand_lowered: '手还没有举高；请把手举到肩膀上方。',
    palm_not_open: '手掌还未清楚张开；请张开手指，掌心朝向镜头。',
    hand_scale: '手掌大小还不合适；请调整距离，让整只手掌清楚可见。',
    hand_outside: '手掌没有完整进入画面；请把手移回画面内。',
    hand_uncertain: '手掌还不够清楚；请让整只手掌朝向镜头。',
    no_single_open_palm: '尚未看清一只举起的张开手掌；请只举一只手，掌心朝向镜头。',
    hand_tracking: '已看到举起的张开手掌；请保持手掌可见，继续左右挥动。',
    motion_incomplete: '已看到举起的张开手掌；请左右多挥两下，让动作更明显。',
    cooldown: '刚刚已识别挥手；请稍等片刻再挥手。',
    wave_hand_rotation: '已识别挥手动作，正在等待处理结果。',
    wave_hand_translation: '已识别挥手动作，正在等待处理结果。',
    wave: '已识别挥手动作，正在等待处理结果。',
    frame_gap: '画面更新有间隔；请保持手掌可见，再左右挥两下。',
    body_jump: '请站稳，让上半身和手掌保持在画面内，再左右挥两下。',
    person_change: '请让一人站在画面中，再举起一只手挥动。',
  };
  function detectionInfo(value) {
    const detector = value.detector;
    if (!detector || typeof detector !== 'object') return '检测中：抬起手，左右挥两下。';
    // current_* describe this frame. The other reason keys and longest_run are
    // lifetime diagnostics, so their counts must never imply a current blocker.
    const hasCurrent = typeof detector.current_reason === 'string';
    let reason = hasCurrent ? detector.current_reason : detector.last_reason;
    if (hasCurrent && reason === 'no_single_open_palm' && Number.isSafeInteger(detector.eligible_hands) && detector.eligible_hands > 1) {
      return '看到了不止一只举起的张开手掌；请只举一只手挥动。';
    }
    if (hasCurrent && reason === 'no_single_open_palm' && detector.eligible_hands === 0) {
      const blockers = Array.isArray(detector.current_blockers) ? detector.current_blockers : [];
      reason = ['hand_outside', 'hand_scale', 'hand_lowered', 'palm_not_open', 'hand_uncertain']
        .find(blocker => blockers.includes(blocker)) || reason;
      if (reason === 'no_single_open_palm' && detector.hands_seen === 0) reason = 'no_hands';
    }
    const text = Object.hasOwn(handGuidance, reason) ? handGuidance[reason] : null;
    // Legacy workers have only last_reason; label its age rather than presenting
    // a historical observation as the live state. Never echo unknown metadata.
    return text ? (hasCurrent ? text : '最近检测：' + text) : '检测中：抬起手，左右挥两下。';
  }
  let owner, ownsLease = false, pending = false, timer, generation = 0, revision = 0, animation;
  let away = false, snapshot = null, refreshing = false;
  const visible = () => !away && !document.hidden;
  function newOwner() {
    return Array.from(crypto.getRandomValues(new Uint8Array(16)), v => v.toString(16).padStart(2, '0')).join('');
  }
  async function request(action, token = owner, keepalive = false, reason) {
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), 2500);
    const body = {action, owner: token};
    if (reason) body.reason = reason;
    try {
      const response = await fetch('/api/wave/control', {method: 'POST', cache: 'no-store', keepalive,
        signal: controller.signal, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      if (!response.ok) throw Error('wave unavailable');
      return await response.json();
    } finally { clearTimeout(deadline); }
  }
  async function status() {
    const controller = new AbortController(), deadline = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch('/api/wave/status', {cache: 'no-store', signal: controller.signal});
      if (!response.ok) throw Error('status unavailable');
      const value = await response.json();
      if (typeof value.enabled !== 'boolean' || typeof value.ready !== 'boolean') throw Error('invalid status');
      return value;
    } finally { clearTimeout(deadline); }
  }
  function draw(value = snapshot) {
    snapshot = value;
    const known = value && typeof value.enabled === 'boolean', enabled = known && value.enabled;
    button.disabled = pending || !known || (enabled && !ownsLease);
    button.textContent = pending ? '正在处理…' : !known ? '状态未连接' :
      enabled ? (ownsLease ? '关闭挥手互动' : '另一页面已开启') : '开启挥手互动';
    button.setAttribute('aria-pressed', String(!!enabled));
    panel.dataset.state = !known ? 'unknown' : enabled ? 'enabled' : 'off';
    if (!known) {
      state.textContent = '状态未知'; info.textContent = '状态连接失败，不能判断是否开启；不会自动启动摄像头。';
      detail.textContent = '正在尝试重新读取状态。'; return;
    }
    if (!enabled) {
      state.textContent = '已关闭';
      info.textContent = reasons[value.reason === 'off' ? value.last_stop_reason || 'off' : value.reason] || '已停止，原因暂未识别';
    } else if (!value.ready || !Number.isFinite(value.frame_age_ms) || value.frame_age_ms < 0 || value.frame_age_ms > 5000) {
      state.textContent = value.reason === 'loading' ? '准备中' : '检测未就绪';
      info.textContent = value.reason === 'loading'
        ? value.speech_preparation?.state === 'preparing'
          ? '正在提前准备本地问候语音，请稍候；此时不会播放声音。'
          : '正在准备本地挥手识别…'
        : '已开启，但摄像头或检测更新尚未就绪。';
    } else {
      state.textContent = '已开启';
      info.textContent = value.quiet ? '检测中；安静模式下不主动出声。' :
        value.user_busy || value.turn_active ? '检测中；当前用户交互优先，暂不问候。' :
        value.output_job && ['greet_named', 'greet_generic'].includes(value.last_decision)
          ? '挥手已识别，正在准备语音回应。' :
        value.output_job && value.last_decision === 'playing' ? '语音正在播放。' :
        value.output_job ? '语音处理尚未结束，请稍候。' : detectionInfo(value);
    }
    detail.textContent = (decisions[value.last_decision] ? '最近一次处理：' + decisions[value.last_decision] + '。' : '') +
      (enabled && ['failed', 'cleanup_unconfirmed', 'deferred'].includes(value.speech_preparation?.state)
        ? '语音提前准备尚未完成，首次回应可能较慢。' : '') +
      '离页即关闭 · 不保存视频 · 不开麦';
  }
  function react(text) {
    window.dispatchEvent(new CustomEvent('starbuddy-wave', {detail: {text}}));
    const figure = document.querySelector('.buddy-motion, #buddy, #mask');
    animation?.cancel();
    if (figure && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      animation = figure.animate([
        {transform: 'translateY(0) rotate(0)'}, {transform: 'translateY(-10px) rotate(-3deg)'},
        {transform: 'translateY(2px) rotate(3deg)'}, {transform: 'translateY(-4px) rotate(-1deg)'},
        {transform: 'translateY(0) rotate(0)'}
      ], {duration: 850, iterations: 1, easing: 'ease-in-out'});
    }
  }
  function schedule() {
    clearTimeout(timer);
    if (visible() && !pending) timer = setTimeout(refresh, ownsLease ? 1000 : 3000);
  }
  async function refresh() {
    clearTimeout(timer);
    if (!visible() || pending || refreshing) return;
    refreshing = true;
    const current = generation;
    try {
      const result = ownsLease ? await request('heartbeat') : await status();
      if (current !== generation || !visible()) return;
      if (!result.enabled) ownsLease = false;
      draw(result);
      if (ownsLease && result.greeting && result.greeting.revision > revision) {
        revision = result.greeting.revision; react(result.greeting.text);
      }
    } catch (_) {
      if (current !== generation || !visible()) return;
      if (ownsLease) { stop('heartbeat_failed'); return; }
      draw(null);
    } finally { refreshing = false; schedule(); }
  }
  async function stop(reason) {
    // Read-only page visits never send stop for another window's lease.
    if (!ownsLease && !pending) return;
    const oldOwner = owner, current = ++generation;
    ownsLease = false; pending = true; clearTimeout(timer); animation?.cancel(); draw();
    try {
      const result = await request('stop', oldOwner, true, reason);
      if (current === generation) snapshot = result;
    } catch (_) { if (current === generation) snapshot = null; }
    finally {
      if (current === generation) { pending = false; draw(); schedule(); }
    }
  }
  button.addEventListener('click', async () => {
    if (pending) return;
    if (ownsLease) { stop('manual_stop'); return; }
    if (snapshot?.enabled || !visible()) return;
    pending = true; owner = newOwner(); revision = 0; draw();
    const current = ++generation, token = owner;
    try {
      const result = await request('start', token);
      if (current !== generation || !visible()) {
        request('stop', token, true, 'page_left').catch(() => {}); return;
      }
      ownsLease = result.enabled === true; snapshot = result;
    } catch (_) {
      if (current === generation) {
        snapshot = null;
        // A lost start reply is not proof that the worker never started.
        request('stop', token, true, 'heartbeat_failed').catch(() => {});
      }
    }
    finally { if (current === generation) { pending = false; draw(); schedule(); } }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { clearTimeout(timer); stop('page_hidden'); }
    else refresh();
  });
  window.addEventListener('pagehide', () => { away = true; clearTimeout(timer); stop('page_left'); });
  window.addEventListener('pageshow', () => { away = false; refresh(); });
  document.addEventListener('click', event => {
    if (event.target.closest('a')) stop('navigation');
    else if (event.target.closest('#start, #voice, #voice-action, #voice-turn, #turn-stop, #silence')) stop('user_interaction');
  }, true);
  document.querySelector('#chat-form')?.addEventListener('submit', () => stop('user_interaction'), true);
  refresh();
})();
