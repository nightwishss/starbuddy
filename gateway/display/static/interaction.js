(() => {
  'use strict';
  const byId = (id) => document.getElementById(id);
  const ui = {
    status: byId('connection-status'), statusLabel: byId('connection-label'),
    buddy: byId('agent-buddy'), caption: byId('buddy-caption'), tasks: byId('task-list'),
    selected: byId('selected-task'), title: byId('selected-title'), prompt: byId('selected-prompt'),
    clear: byId('task-clear'), conversation: byId('conversation'), empty: byId('conversation-empty'),
    form: byId('message-form'), message: byId('message'), send: byId('send-button'),
    voice: byId('voice-button'), cancel: byId('cancel-button'), request: byId('request-status'),
    offers: byId('offers-toggle'), offer: byId('offer-panel'), offerText: byId('offer-text'),
    decline: byId('offer-decline'), intent: byId('decision-intent'), target: byId('decision-target'),
    reason: byId('decision-reason'), activity: byId('model-activity'),
    reset: byId('workspace-reset'), goalKind: byId('goal-kind'), goalTitle: byId('goal-title'),
    goalStep: byId('goal-step'), goalMissing: byId('goal-missing'), example: byId('example-reminder'),
    attention: byId('attention-label'), contextStatus: byId('context-status'), objects: byId('context-objects'),
    sources: byId('source-states'), actors: byId('actor-boundary'), actionPanel: byId('action-panel'),
    actionStatus: byId('action-status'), actionDraft: byId('action-draft'), actionSummary: byId('action-summary'),
    actionExpiry: byId('action-expiry'), actionConfirm: byId('action-confirm'), actionCancel: byId('action-cancel'),
    actionDetail: byId('action-detail'), actionReceipt: byId('action-receipt'),
    actionTaskState: byId('action-task-state'), actionTaskId: byId('action-task-id'),
    delivery: byId('delivery-panel'), deliveryState: byId('delivery-state'), deliveryDetail: byId('delivery-detail'),
    continuous: byId('continuous-voice-button'), continuousState: byId('continuous-voice-state'),
    continuousDetail: byId('continuous-voice-detail')
  };
  const busyPhases = new Set(['listening', 'thinking', 'speaking', 'acting', 'working']);
  const phases = {
    idle: ['等你开口', '告诉我你想做什么，我们一起找下一步。', 'curious'],
    attending: ['正在关注这次互动', '我在，告诉我你想一起做什么。', 'listening'],
    engaged: ['正在一起思考', '我们接着刚才的事继续。', 'amused'],
    listening: ['正在听这一句', '慢慢说，我在听。', 'listening'],
    thinking: ['正在想办法', '我在联系这次的线索，想一想怎么帮你。', 'listening'],
    speaking: ['正在回应', '这是我想到的，我们一起看看。', 'amused'],
    acting: ['正在处理', '正在完成这一步。', 'listening'],
    quiet: ['安静模式', '我会安静等待，不主动打扰。', 'curious'],
    error: ['暂时遇到问题', '这一步没有完成，可以稍后再试。', 'curious'],
    unavailable: ['暂时无法互动', '连接恢复后，我们再继续。', 'curious']
  };
  const intentLabels = { task_help: '请求当前题目的帮助', help_task: '请求当前题目的帮助',
    task_answer: '回答当前题目', answer_task: '回答当前题目', task_selection: '选择小挑战',
    decline_help: '暂时不需要帮助', clarify: '先确认你的意思', inspect_scene: '了解当前看见的事物',
    chat: '聊一聊', cancel: '停止这次互动', social: '打个招呼', unknown: '还需要更多线索' };
  const reasonLabels = { explicit_request: '依据你本次明确提出的请求。', task_selected: '依据屏幕上明确选中的题目。',
    answer_checked: '已针对当前选中的题目核对答案。', uncertain: '现有线索不够明确，需要先确认。',
    missing_task: '还没有确定你指的是哪一道题。', offer_eligible: '你已允许建议，且当前任务满足提示条件。',
    user_declined: '你已经表示暂时不需要帮助。', cancelled: '你已停止这次互动。', task_complete: '当前题目的完成条件已满足。',
    no_action: '当前没有需要执行的动作。', model_decision: '结合本轮输入和可用情境，形成这次互动判断。' };
  const actionLabels = {
    idle: ['尚无已提交的动作', '涉及预约等实际动作，会先展示提议；点确认后才提交。'],
    draft: ['只是一个提议 · 尚未提交', '请先核对内容；没有点确认，不会创建预约。'],
    proposed: ['等你确认 · 尚未预约', '确认后才提交调度器；你也可以放弃这次提议。'],
    awaiting_confirmation: ['等你确认 · 尚未预约', '确认后才提交调度器；你也可以放弃这次提议。'],
    submitting: ['正在提交 · 预约尚未确认', '正在等待调度回执，请不要重复确认。'],
    scheduled: ['已预约 · 尚未执行', '调度器已接受任务；这不代表任务已经执行。'],
    running: ['任务正在执行', '执行中，还不能视为完成或已经听到声音。'],
    completed: ['调度器报告任务完成', '这是任务回执；不等于已核实现场听到声音。'],
    cancelled: ['任务已取消', '以调度器当前回执为准，不再把原提议当作待执行任务。'],
    cancelling: ['正在请求取消 · 结果待确认', '取消请求还没有最终回执，暂不把原任务当作已取消。'],
    expired: ['提议已过期 · 尚未预约', '这次未确认的提议已经失效，请重新说明需求。'],
    failed: ['动作执行失败', '没有完成这次动作；请查看回执或调整后重新发起。'],
    unknown: ['结果尚未确认', '当前没有可靠的成功回执，请勿据此认为已经预约或执行。']
  };
  const taskStateLabels = { queued: '任务已入队，等待执行', scheduled: '任务已预约',
    pending: '等待执行', running: '正在执行', completed: '调度器报告完成',
    succeeded: '调度器报告成功', cancelled: '已取消', failed: '执行失败', expired: '已过期', unknown: '结果未确认' };
  const sourceLabels = { available: '本轮可用', stale: '线索已过期', unavailable: '暂不可用',
    disabled: '未开启', unknown: '尚未确认', error: '读取失败', idle: '当前无可用线索' };
  const missingLabels = { time: '时间', duration: '多久以后', when: '提醒时间', content: '要做的事',
    message: '提醒内容', target: '具体对象', object: '具体对象', object_reference: '具体对象',
    minutes: '几分钟后', topic: '想聊的内容', identity: '主体是谁' };
  let state = null;
  let connected = false;
  let pending = false;
  let cancelling = false;
  let cancelUnconfirmed = false;
  let cancelAcknowledged = false;
  let requestGeneration = 0;
  let controller = null;
  let pollTimer = null;
  let loadingState = false;
  let stateRequest = 0;
  let taskSignature = '';
  let objectSignature = '';
  let leftPage = false;
  let voiceSession = null;
  const activeTurn = () => Boolean(state && (busyPhases.has(state.phase) || state.input_active === true ||
    state.output_pending === true || state.turn_pending === true || state.scheduler && state.scheduler.busy === true));
  const text = (value, limit = 2000) => typeof value === 'string' ? value.slice(0, limit) : '';
  const setText = (node, value) => { if (node.textContent !== value) node.textContent = value; };
  const field = (value) => {
    if (typeof value === 'string') return text(value, 240);
    if (value && typeof value === 'object') return text(value.title || value.label || value.name || value.id, 240);
    return '';
  };
  const workspaceValue = () => state && state.workspace && typeof state.workspace === 'object' ? state.workspace : null;
  const actionValue = () => state && state.action && typeof state.action === 'object' ? state.action : {};
  const confirmableDraft = () => {
    const action = actionValue();
    const draft = action.draft;
    return Boolean(['draft', 'proposed', 'awaiting_confirmation'].includes(action.status) && draft && typeof draft.token === 'string' && draft.token &&
      typeof draft.expires_in_s === 'number' && Number.isFinite(draft.expires_in_s) && draft.expires_in_s > 0 &&
      ['pending', 'draft', 'proposed', 'awaiting_confirmation'].includes(draft.state));
  };
  function goalStageText(goal) {
    if (!goal) return '可以聊眼前的事，也可以一起计划下一步。';
    const step = typeof goal.step === 'string' ? goal.step : '';
    if (step === 'target_unavailable') return '原来关注的对象已不可用，请重新选择线索或说明。';
    if (step === 'clarify' || goal.state === 'needs_clarification') return '还有一些信息不明确，需要先和你确认。';
    // A conversation state is not a task-completion or scheduling receipt.
    if (goal.kind === 'conversation') return goal.state === 'cancelled'
      ? '这次交流已停止；需要时可以再聊一个新话题。'
      : '围绕你说的事继续交流，需要时再一起确认下一步。';
    if (goal.kind === 'inspect_object') return goal.state === 'cancelled'
      ? '已停止了解这个对象，可以重新选择线索。'
      : '先确认你关注的对象，再结合已有线索继续了解。';
    const stateStages = {
      awaiting_confirmation: '先核对提议；点确认后才会提交。',
      submitting: '已经请求提交，正在等待真实任务回执。',
      scheduled: '调度器已接受预约，尚未执行。',
      unknown: '任务结果还未确认；先核对回执，不重复提交。',
      cancelled: '已收到任务取消回执，不再继续原预约。',
      completed: '调度器报告任务完成；不代表已核实现场听到。',
      failed: '任务执行失败，可以先查看回执再决定下一步。'
    };
    if (stateStages[goal.state]) return stateStages[goal.state];
    const stepStages = {
      proposed: '这是当前提议，还需要和你确认下一步。',
      confirmation: '先核对提议，等待你明确确认。',
      task_receipt: '以实际任务回执为准，暂不推断已经完成。'
    };
    return stepStages[step] || '我们先确认当前目标，再一起决定下一步。';
  }

  function renderWorkspace() {
    const workspace = workspaceValue();
    const goal = workspace && workspace.goal && typeof workspace.goal === 'object' ? workspace.goal : null;
    const goalKinds = { reminder: '一个提醒计划', schedule_reminder: '一个提醒计划', task: '当前任务',
      learning: '一起学一点', explain: '一起理解', plan: '一起计划', generic: '这次的小目标',
      inspect_object: '了解眼前的对象', conversation: '这次聊的事' };
    ui.goalKind.textContent = goal ? goalKinds[goal.kind] || '这次的小目标' : '一个小目标';
    ui.goalTitle.textContent = goal ? text(goal.title, 180) || '目标还需要确认' : '先告诉我，你想做什么？';
    ui.goalStep.textContent = goalStageText(goal);
    const missing = goal && Array.isArray(goal.missing) ? [...new Set(goal.missing.slice(0, 5).map((value) => typeof value === 'string' ? missingLabels[value] || '具体信息' : '').filter(Boolean))] : [];
    ui.goalMissing.hidden = !missing.length;
    ui.goalMissing.textContent = missing.length ? `还需要确认：${missing.join('、')}` : '';
    ui.example.hidden = Boolean(goal);
    const attention = workspace && workspace.attention;
    ui.attention.textContent = attention && typeof attention === 'object' ? text(attention.label, 100) || '已选中一条线索' : '还没有选定对象';
    const sources = state && state.source_states && typeof state.source_states === 'object' ? state.source_states : {};
    const sourceValues = Object.values(sources);
    ui.contextStatus.textContent = !connected ? '情境连接暂时中断' : !workspace ? '情境尚未接通' : sourceValues.includes('available') ? '可用线索快照 · 非实时' : sourceValues.includes('stale') ? '已有线索可能过期' : '尚无已确认的现场线索';
    ui.sources.replaceChildren();
    for (const [key, label] of [['people', '人物'], ['objects', '物体'], ['posture', '坐姿'], ['affect', '表情']]) {
      const source = document.createElement('span');
      const sourceState = connected && typeof sources[key] === 'string' ? sources[key] : 'unknown';
      source.className = 'source-chip';
      source.dataset.state = ['available', 'stale', 'unavailable', 'disabled', 'unknown', 'error'].includes(sourceState) ? sourceState : 'unknown';
      source.textContent = `${label}：${sourceLabels[sourceState] || '尚未确认'}`;
      ui.sources.append(source);
    }
    const actors = workspace && workspace.actors;
    const peopleCount = actors && Array.isArray(actors.visible_people) ? Math.min(99, actors.visible_people.length) : null;
    const peopleText = connected && sources.people === 'available' && peopleCount !== null ? `本轮有 ${peopleCount} 条匿名人物线索。` : '人物线索尚未确认。';
    ui.actors.textContent = peopleText + '声音与触摸身份未证实，不自动当作镜头中的同一个人，也不据此读取个人档案。';
    const objects = workspace && Array.isArray(workspace.objects) ? workspace.objects.slice(0, 8).filter((item) => item && typeof item.ref === 'string' && typeof item.label === 'string') : [];
    const signature = JSON.stringify(objects.map((item) => [item.ref, item.label, item.count]));
    if (signature !== objectSignature || !ui.objects.children.length) {
      objectSignature = signature;
      ui.objects.replaceChildren();
      for (const item of objects) {
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'context-object'; button.dataset.ref = item.ref;
        const count = Number.isSafeInteger(item.count) && item.count > 0 ? Math.min(item.count, 999) : null;
        button.textContent = text(item.label, 60) + (count !== null ? ` · ${count}` : '');
        button.addEventListener('click', () => {
          const current = workspaceValue();
          if (current && Number.isSafeInteger(current.revision)) mutate('/api/agent/attention', { ref: item.ref, revision: current.revision }, '正在确认这次关注的对象…');
        });
        ui.objects.append(button);
      }
      if (!objects.length) {
        const note = document.createElement('p'); note.className = 'context-empty';
        note.textContent = '暂时没有可用的物体线索，不会自动开启摄像头。'; ui.objects.append(note);
      }
    }
    ui.objects.querySelectorAll('button').forEach((button) => button.setAttribute('aria-pressed', String(Boolean(attention && button.dataset.ref === attention.ref))));
    const scheduler = state && state.scheduler;
    const calls = scheduler && Number.isSafeInteger(scheduler.model_calls) ? scheduler.model_calls : null;
    const inspections = scheduler && Number.isSafeInteger(scheduler.inspection_calls) ? scheduler.inspection_calls : null;
    ui.activity.textContent = calls !== null || inspections !== null ? `理解 ${calls === null ? '未确认' : calls} 次 · 视觉精查 ${inspections === null ? '未确认' : inspections} 次（服务端记录）` : '尚无本轮调用记录';
  }

  function renderAction() {
    const action = actionValue();
    const status = typeof action.status === 'string' ? action.status : 'unknown';
    const labels = actionLabels[status] || actionLabels.unknown;
    if (action.reason !== 'cancel_requires_explicit_retry') setText(ui.actionStatus, (connected ? '' : '上次回执：') + labels[0]);
    ui.actionDetail.textContent = labels[1] + (connected ? '' : '连接中断，当前结果需恢复后核对。');
    ui.actionPanel.dataset.state = Object.prototype.hasOwnProperty.call(actionLabels, status) ? status : 'unknown';
    const draft = action.draft && typeof action.draft === 'object' ? action.draft : null;
    ui.actionDraft.hidden = !draft;
    ui.actionSummary.textContent = draft ? text(draft.summary, 500) || '提议内容尚未完整，请先补充。' : '';
    const remaining = draft && typeof draft.expires_in_s === 'number' && Number.isFinite(draft.expires_in_s) ? Math.max(0, Math.ceil(draft.expires_in_s)) : null;
    ui.actionExpiry.textContent = draft ? remaining === 0 ? '提议已过期，请重新说明需求。' : remaining !== null ? `约 ${remaining} 秒后过期，以服务端校验为准。` : '有效期尚未确认，暂不能提交。' : '';
    const task = action.task && typeof action.task === 'object' ? action.task : null;
    const taskId = task && text(task.task_id, 128);
    ui.actionReceipt.hidden = !taskId;
    ui.actionTaskId.textContent = taskId || '';
    ui.actionTaskState.textContent = ['unknown', 'cancelling'].includes(status) ? '当前结果未确认，不视为成功' : task ? taskStateLabels[task.state] || '服务端状态尚未确认，不视为成功' : '未确认';
    ui.actionCancel.hidden = !draft && !taskId && action.reason !== 'cancel_requires_explicit_retry';
    ui.actionCancel.textContent = taskId ? '取消关联任务' : '放弃这次提议';
    if (taskId && !['cancelled', 'completed', 'failed'].includes(status)) ui.actionDetail.textContent += '要结束本次情境，请先取消关联任务。';
    if (action.reason === 'cancel_requires_explicit_retry') {
      setText(ui.actionStatus, (connected ? '' : '上次回执：') + '已查到预约 · 取消尚未提交');
      ui.actionDetail.textContent = '已查到预约，但取消尚未提交；请再次点击取消，核对真实回执。';
      ui.actionCancel.textContent = '再次取消关联任务';
      ui.actionPanel.dataset.state = 'unknown';
    }
    const delivery = state && state.delivery && typeof state.delivery === 'object' ? state.delivery : null;
    const deliveryLabels = {
      queued: ['语音回应已入队', '等待播放服务处理，还不代表已经出声。'],
      playing: ['播放服务报告播放中', '这里没有进行现场听音确认。'],
      completed: ['播放服务报告完成', '这是服务回执，不等于已确认现场听到。'],
      failed: ['语音播放失败', '未完成播放，请以服务端回执为准。'],
      cancelled: ['语音播放已取消', '本轮不再继续原来的语音回应。'],
      interrupted: ['语音播放已中断', '不把中断的回应视为完整播放。'],
      expired: ['语音回应已过期', '原来的回应不再继续播放。'],
      unknown: ['语音播放结果未确认', '没有可靠完成回执，不能视为已播放。']
    };
    ui.delivery.hidden = !delivery || delivery.state === 'idle';
    if (delivery && delivery.state !== 'idle') {
      const terminal = ['completed', 'failed', 'cancelled', 'interrupted', 'expired'].includes(delivery.state);
      const deliveryUI = terminal && delivery.confirmed_terminal !== true ? deliveryLabels.unknown : deliveryLabels[delivery.state] || deliveryLabels.unknown;
      setText(ui.deliveryState, (connected ? '' : '上次回执：') + deliveryUI[0]); ui.deliveryDetail.textContent = deliveryUI[1];
      ui.delivery.dataset.state = delivery.state;
    }
  }

  function setRequest(message, kind = 'info') {
    ui.request.textContent = message;
    ui.request.dataset.kind = kind;
    ui.request.hidden = !message;
  }

  function displayError(error) {
    const reason = error && error.message ? error.message : '请求没有完成，请稍后重试。';
    setRequest(reason, 'error');
  }

  async function fetchJSON(path, options = {}) {
    const response = await fetch(path, {
      credentials: 'same-origin', cache: 'no-store',
      ...options,
      headers: { 'Accept': 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers }
    });
    let data;
    try { data = await response.json(); } catch (_) { throw new Error('连接没有返回有效结果，请稍后重试。'); }
    if (!response.ok || !data || data.ok === false) {
      const reason = text(data && (data.message || data.error || data.reason), 240);
      throw new Error(reason || '这一步暂时没有完成，请稍后重试。');
    }
    return data;
  }

  function renderControls() {
    const serverBusy = activeTurn();
    const sessionActive = Boolean(voiceSession && voiceSession.engaged);
    const locked = !connected || pending || cancelling || serverBusy || cancelUnconfirmed || sessionActive;
    ui.message.disabled = locked;
    ui.send.disabled = locked || !ui.message.value.trim();
    ui.voice.disabled = locked || !window.StarBuddyContinuousVoice ||
      !window.StarBuddyContinuousVoice.controlFor(state) || state.voice_ready !== true;
    ui.clear.disabled = locked;
    ui.offers.disabled = locked;
    ui.decline.disabled = locked;
    const workspace = workspaceValue();
    const action = actionValue();
    const linkedTask = action.reason === 'cancel_requires_explicit_retry' || action.task && action.task.task_id && !['cancelled', 'completed', 'failed'].includes(action.status);
    ui.reset.disabled = locked || !workspace || Boolean(linkedTask);
    ui.reset.title = linkedTask ? '请先取消关联任务，再结束本次情境。' : '';
    ui.example.disabled = locked;
    ui.actionConfirm.disabled = locked || !confirmableDraft();
    ui.actionCancel.disabled = !connected || pending || cancelling || cancelUnconfirmed || sessionActive;
    ui.objects.querySelectorAll('button').forEach((button) => { button.disabled = locked || !workspace || !Number.isSafeInteger(workspace.revision); });
    ui.tasks.querySelectorAll('button').forEach((button) => { button.disabled = locked; });
    ui.cancel.disabled = cancelling || (!pending && !serverBusy && !cancelUnconfirmed && !sessionActive);
    ui.cancel.textContent = cancelling ? '正在停止…' : '停止';
    if (ui.continuous) {
      ui.continuous.disabled = !sessionActive && (!voiceSession || pending || cancelling || cancelUnconfirmed || !voiceSession.canStart());
      ui.continuous.setAttribute('aria-pressed', String(Boolean(voiceSession && voiceSession.enabled)));
    }
  }

  function renderState() {
    const phase = state && typeof state.phase === 'string' ? state.phase : '';
    let phaseUI = phases[phase];
    if (!connected) phaseUI = ['连接暂时中断', '目前无法确认互动状态，恢复连接后再继续。', 'curious'];
    else if (!phaseUI) phaseUI = ['已连接 · 状态更新中', '告诉我眼前的事，或者说一个小目标。', 'curious'];
    else if (phase === 'idle' && state.state === 'completed') phaseUI = ['小挑战已完成', '这一题完成了！想休息一下，还是再选一个小挑战？', 'amused'];
    if ((pending || activeTurn()) && connected && !busyPhases.has(phase)) phaseUI = ['正在处理这次请求', '正在处理这次互动，请稍等一下。', 'listening'];
    if (cancelUnconfirmed && connected) phaseUI = ['停止尚未确认', '还没能确认停止，请再点一次“停止”。', 'curious'];
    setText(ui.statusLabel, phaseUI[0]);
    setText(ui.caption, phaseUI[1]);
    ui.status.dataset.state = !connected || cancelUnconfirmed || phase === 'error' || phase === 'unavailable' ? 'error' : pending || activeTurn() ? 'working' : 'ready';
    ui.buddy.dataset.phase = !connected ? 'error' : (pending || activeTurn()) && !busyPhases.has(phase) ? 'working' : phase || 'idle';
    ui.buddy.dataset.mood = phaseUI[2];
    if (!state) { renderControls(); return; }
    renderWorkspace(); renderAction();
    const tasks = Array.isArray(state.tasks) ? state.tasks.slice(0, 3).filter((task) => task && typeof task.id === 'string') : [];
    const signature = JSON.stringify(tasks.map((task) => [task.id, task.title]));
    if (signature !== taskSignature) {
      taskSignature = signature;
      ui.tasks.replaceChildren();
      tasks.forEach((task, index) => {
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'task-card'; button.dataset.taskId = task.id;
        const number = document.createElement('span'); number.className = 'task-number'; number.textContent = `小挑战 0${index + 1}`;
        const name = document.createElement('span'); name.className = 'task-name'; name.textContent = text(task.title, 80) || '一起想一想';
        button.append(number, name);
        button.addEventListener('click', () => mutate('/api/agent/task', { task_id: task.id }, '正在切换这次的小挑战…'));
        ui.tasks.append(button);
      });
      if (!tasks.length) {
        const note = document.createElement('p'); note.className = 'loading-note'; note.textContent = '暂时没有可选的小挑战，也可以直接和芽芽说。'; ui.tasks.append(note);
      }
    }
    const task = state.task && typeof state.task === 'object' ? state.task : null;
    ui.selected.hidden = !task;
    ui.clear.hidden = !task;
    ui.title.textContent = task ? text(task.title, 120) : '';
    ui.prompt.textContent = task ? text(task.prompt, 1000) : '';
    ui.tasks.querySelectorAll('button').forEach((button) => button.setAttribute('aria-pressed', String(Boolean(task && task.id === button.dataset.taskId))));
    if (!pending) ui.offers.checked = state.offers_enabled === true || Boolean(state.offers && state.offers.enabled === true);
    const offer = state.offer;
    const offerText = typeof offer === 'string' ? text(offer, 500) : offer && typeof offer === 'object' ? text(offer.text || offer.message || offer.title, 500) : '';
    ui.offerText.textContent = offerText;
    ui.offer.hidden = !offerText;
    const decision = state.decision && typeof state.decision === 'object' ? state.decision : null;
    const intent = decision && field(decision.intent || decision.intention || decision.goal);
    const reason = decision && field(decision.reason || decision.basis);
    ui.intent.textContent = decision ? intentLabels[intent] || intent || '尚未确定' : '还没有本次理解结果';
    const target = decision && field(decision.target || decision.object || decision.task);
    ui.target.textContent = decision ? task && target === task.id ? text(task.title, 120) : target === 'current_turn' ? '本轮互动' : target || '尚未确定' : '尚未确定';
    ui.reason.textContent = decision ? reasonLabels[reason] || reason || '本次未提供可展示的判断依据' : '等待你发起互动';
    renderControls();
  }

  function applyState(data) {
    const incoming = data && data.state;
    if (incoming && typeof incoming === 'object') {
      const oldRevision = state && Number.isFinite(state.revision) ? state.revision : -1;
      const nextRevision = Number.isFinite(incoming.revision) ? incoming.revision : oldRevision;
      const epochChanged = state && typeof incoming.control_epoch === 'string' && incoming.control_epoch !== state.control_epoch;
      if (epochChanged) state = incoming;
      else if (nextRevision >= oldRevision) state = { ...state, ...incoming };
    }
    connected = true;
    if (cancelUnconfirmed && cancelAcknowledged && incoming && typeof incoming === 'object' &&
        window.StarBuddyContinuousVoice && window.StarBuddyContinuousVoice.quiescent(state)) cancelUnconfirmed = false;
    renderState();
    if (voiceSession) voiceSession.observe(state, true);
  }

  async function refreshState() {
    if (loadingState || document.hidden || leftPage) return;
    loadingState = true;
    const serial = ++stateRequest;
    const stateController = new AbortController();
    const timeout = setTimeout(() => stateController.abort(), 8000);
    try {
      const data = await fetchJSON('/api/agent/state', { signal: stateController.signal });
      if (serial === stateRequest && !leftPage) applyState(data);
    } catch (_) {
      if (serial === stateRequest && !leftPage) {
        connected = false;
        if (voiceSession) voiceSession.observe(state, false);
        renderState();
      }
    } finally { clearTimeout(timeout); loadingState = false; }
  }

  function addMessage(role, message) {
    const content = text(message, 6000);
    if (!content) return;
    ui.empty.hidden = true;
    const item = document.createElement('div'); item.className = `message ${role}`;
    const label = document.createElement('span'); label.className = 'message-label'; label.textContent = role === 'user' ? '你' : '芽芽';
    const paragraph = document.createElement('p'); paragraph.className = 'message-text'; paragraph.textContent = content;
    item.append(label, paragraph); ui.conversation.append(item);
    const messages = ui.conversation.querySelectorAll('.message');
    Array.from(messages).slice(0, Math.max(0, messages.length - 8)).forEach((messageNode) => messageNode.remove());
    ui.conversation.scrollTop = ui.conversation.scrollHeight;
  }

  async function mutate(path, body, status, onResult, { allowWhileBusy = false, sessionTurn = false, propagateErrors = false } = {}) {
    if (pending || cancelling || cancelUnconfirmed || !connected || leftPage || activeTurn() && !allowWhileBusy ||
        voiceSession && voiceSession.engaged && !sessionTurn) return;
    pending = true;
    const generation = ++requestGeneration;
    controller = new AbortController();
    const timeout = setTimeout(() => { if (controller && generation === requestGeneration) controller.abort(); }, 120000);
    setRequest(status); renderState();
    try {
      const data = await fetchJSON(path, { method: 'POST', body: JSON.stringify(body), signal: controller.signal });
      if (generation !== requestGeneration || leftPage) return;
      applyState(data);
      setRequest('');
      if (onResult) onResult(data);
      return data;
    } catch (error) {
      if (generation === requestGeneration && !leftPage && error.name !== 'AbortError') displayError(error);
      if (propagateErrors) throw error;
    } finally {
      clearTimeout(timeout);
      if (generation === requestGeneration) { pending = false; controller = null; renderState(); refreshState(); }
    }
  }

  async function cancel() {
    if (voiceSession && voiceSession.engaged) {
      const stopped = voiceSession.stop('user');
      setRequest(stopped ? '持续对话已结束。' : '正在结束当前一句；收音和处理退出尚未确认。');
      return stopped;
    }
    if (cancelling) return false;
    if (ui.cancel.disabled) return true;
    cancelling = true;
    ++requestGeneration;
    if (controller) controller.abort();
    controller = null;
    cancelAcknowledged = false;
    const cancelController = new AbortController();
    const cancelTimeout = setTimeout(() => cancelController.abort(), 8000);
    renderControls(); setRequest('正在请求停止这次互动…');
    try {
      const data = await fetchJSON('/api/agent/cancel', { method: 'POST', body: '{}', signal: cancelController.signal });
      if (leftPage) return;
      cancelUnconfirmed = true;
      cancelAcknowledged = true;
      applyState(data); setRequest('已发送停止请求；以芽芽当前状态为准。');
      return !cancelUnconfirmed;
    } catch (error) { cancelUnconfirmed = true; if (!leftPage) displayError(error); return false; }
    finally { clearTimeout(cancelTimeout); pending = false; cancelling = false; if (!leftPage) { renderState(); refreshState(); } }
  }

  ui.form.addEventListener('submit', (event) => {
    event.preventDefault();
    const message = ui.message.value.trim();
    if (!message || ui.send.disabled) return;
    addMessage('user', message);
    ui.message.value = ''; ui.message.style.height = '';
    mutate('/api/agent/chat', { message }, '正在理解这次的问题…', (data) => {
      const answer = text(data.answer || data.result && data.result.answer, 6000);
      if (answer) addMessage('assistant', answer);
      else setRequest('本次没有返回可展示的文字答复。');
    });
  });
  ui.message.addEventListener('input', () => {
    ui.message.style.height = 'auto'; ui.message.style.height = `${Math.min(ui.message.scrollHeight, 112)}px`; renderControls();
  });
  ui.message.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); ui.form.requestSubmit(); }
  });
  ui.voice.addEventListener('click', () => {
    const control = window.StarBuddyContinuousVoice && window.StarBuddyContinuousVoice.controlFor(state);
    if (!control || ui.voice.disabled) return;
    mutate('/api/agent/voice', control, '正在处理语音请求；你可以随时点“停止”。', (data) => {
    const result = data.result && typeof data.result === 'object' ? data.result : data;
    if (result.accepted !== true) {
      setRequest(result.reason === 'no_speech' ? '刚才没有听到清晰语音；准备好后再点“说一句”。'
        : '这次语音没有被接受或已停止；请核对当前状态后再试。', 'error');
      return;
    }
    const answer = text(data.answer || result.answer, 6000);
    if (answer) addMessage('assistant', answer);
    });
  });
  if (ui.continuous && window.StarBuddyContinuousVoice) {
    const sessionLabels = {
      off: '持续对话未开启', waiting: '等你接着说', starting: '正在启动下一句收音',
      listening: '正在听你说', thinking: '正在理解', speaking: '芽芽正在回应 · 暂不收音',
      connecting: '连接状态待确认', stopping: '正在结束当前一句 · 收音退出待确认',
      draining: '收音已退出 · 正在等待处理和播放结束'
    };
    const reasonLabels = {
      off: '点一次开启，芽芽说完后会继续听；不需要每句话再点。',
      user: '已结束持续对话；再次交流时需要你主动开启。',
      silence: '这一轮没有听到说话，已结束续听；想聊时再开启。',
      limit: '已达到本次时长或轮数上限，续听已结束；需要时可以重新开启。',
      hidden: '页面离开前台，已请求结束；回到页面也不会自动续听。',
      navigation: '正在离开本页，已请求结束持续对话。',
      disconnected: '连接中断，已停止发起新收音；当前一句退出需恢复连接后确认。',
      confirmation: '有一个提议需要你在屏幕核对；不会自动确认或创建预约。',
      delivery_unknown: '语音播放回执未确认，已停止续听，避免把芽芽的声音当作你的话。',
      delivery_failed: '语音回应未完整播放，已停止续听。',
      delivery_timeout: '等待语音回应超时，已停止续听并请求结束。',
      request_failed: '本轮收音或理解没有完成，已停止续听；不会自动重试。',
      input_cleanup_unknown: '本次临时声音清理尚未确认，已停止续听。'
    };
    voiceSession = new window.StarBuddyContinuousVoice.Session({
      requestVoice: (control) => mutate('/api/agent/voice', control, '持续对话中；芽芽说话时不会发起下一次收音。', null,
        { sessionTurn: true, propagateErrors: true }),
      cancelTurn: async () => {
        const cancelController = new AbortController();
        const timeout = setTimeout(() => cancelController.abort(), 8000);
        try {
          const data = await fetchJSON('/api/agent/cancel', { method: 'POST', body: '{}', signal: cancelController.signal });
          if (!leftPage) { applyState(data); refreshState(); }
          return data;
        } finally { clearTimeout(timeout); }
      },
      onAnswer: (data) => {
        const result = data.result && typeof data.result === 'object' ? data.result : data;
        // The server deliberately omits private, single-use transcription.
        const answer = text(data.answer || result.answer, 6000);
        if (answer) addMessage('assistant', answer);
      },
      onUpdate: (view) => {
        ui.continuous.textContent = view.engaged ? view.enabled ? '结束持续对话' : '再次请求停止' : '开启持续对话';
        if (ui.continuousState) setText(ui.continuousState, sessionLabels[view.stage] || sessionLabels.off);
        if (ui.continuousDetail) setText(ui.continuousDetail, view.enabled
          ? '最长 3 分钟 / 20 轮；约 8 秒无语音后结束。仅本次处理，不保存录音。'
          : (view.reason === 'input_cleanup_unknown' ? reasonLabels.input_cleanup_unknown + (view.engaged ? '正在等待当前处理结束确认。' : '')
            : view.engaged ? '停止已请求；不能把取消回执当作麦克风已关闭。请等真实状态确认。' : reasonLabels[view.reason] || reasonLabels.off));
        renderControls();
      }
    });
    ui.continuous.addEventListener('click', async () => {
      if (voiceSession.engaged) cancel();
      else {
        const prepared = await mutate('/api/agent/prepare-voice', {}, '正在准备中英文声音，尚未开启麦克风；首次约需几秒…');
        if (!prepared || prepared.prepared !== true || leftPage || document.hidden) return;
        await refreshState();
        if (voiceSession.start()) { setRequest('持续对话已开启；请等“正在听你说”再开口。'); schedulePoll(); }
        else setRequest('声音已就绪；当前互动结束后可开启持续对话。');
      }
    });
  }
  ui.clear.addEventListener('click', () => mutate('/api/agent/task', { task_id: null }, '正在结束这道题…'));
  ui.offers.addEventListener('change', () => {
    const enabled = ui.offers.checked;
    mutate('/api/agent/offers', { enabled }, '正在更新主动建议设置…');
    renderState();
  });
  ui.decline.addEventListener('click', () => mutate('/api/agent/decline', {}, '正在告诉芽芽暂时不用…'));
  ui.cancel.addEventListener('click', cancel);
  ui.example.addEventListener('click', () => {
    if (ui.example.disabled) return;
    ui.message.value = '十分钟后提醒我喝水';
    ui.message.dispatchEvent(new Event('input', { bubbles: true }));
    ui.message.focus();
  });
  ui.actionConfirm.addEventListener('click', () => {
    if (ui.actionConfirm.disabled || !confirmableDraft()) return;
    const token = actionValue().draft.token;
    mutate('/api/agent/confirm', { token }, '正在提交这次确认，等待真实调度回执…');
  });
  ui.actionCancel.addEventListener('click', () => {
    if (ui.actionCancel.disabled) return;
    mutate('/api/agent/action-cancel', {}, '正在取消关联动作，以服务端回执为准…', null, { allowWhileBusy: true });
  });
  ui.reset.addEventListener('click', () => {
    if (ui.reset.disabled) return;
    mutate('/api/agent/reset', {}, '正在结束本次临时情境…', () => {
      ui.conversation.querySelectorAll('.message').forEach((message) => message.remove());
      ui.empty.hidden = false;
    });
  });
  document.querySelectorAll('a[href]').forEach((link) => link.addEventListener('click', async (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || (!pending && !activeTurn() && !cancelUnconfirmed && !(voiceSession && voiceSession.engaged))) return;
    event.preventDefault();
    if (await cancel()) window.location.assign(link.href);
    else setRequest('暂时没能确认停止，尚未离开本页。请再点一次“停止”。', 'error');
  }));

  function schedulePoll() {
    clearTimeout(pollTimer); pollTimer = null;
    if (document.hidden || leftPage) return;
    const poll = async () => {
      await refreshState();
      if (!document.hidden && !leftPage) pollTimer = setTimeout(poll, voiceSession && voiceSession.engaged ? 1000 : 3000);
    };
    poll();
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && voiceSession && voiceSession.engaged) voiceSession.stop('hidden');
    schedulePoll();
  });
  window.addEventListener('pagehide', () => {
    leftPage = true; clearTimeout(pollTimer); ++requestGeneration;
    if (pending || activeTurn() || cancelUnconfirmed || voiceSession && voiceSession.engaged) {
      let queued = false;
      try { queued = navigator.sendBeacon('/api/agent/cancel', new Blob(['{}'], { type: 'application/json' })); } catch (_) { /* Fall back for older kiosk engines. */ }
      if (!queued) fetch('/api/agent/cancel', { method: 'POST', credentials: 'same-origin', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
    }
    if (voiceSession) voiceSession.dispose();
    if (controller) controller.abort();
  });
  window.addEventListener('pageshow', (event) => { if (event.persisted) { leftPage = false; pending = false; cancelling = false; connected = false; renderState(); schedulePoll(); } });
  renderControls(); schedulePoll();
})();
