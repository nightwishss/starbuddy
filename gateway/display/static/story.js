'use strict';
const $ = id => document.getElementById(id);
let owner, durable = true, current = null, busy = false, generation = 0, composing = false;
let audioGeneration = 0, audioTimer = null, audioPending = false;
let storyTextSource = null;
let composerOpen = true, cleanupPending = false, captureGeneration = 0;
let sampleOpen = false, sampleGeneration = 0, sampleRequested = false;
const sampleVideoPath = '/story-media/story-garden-demo-v1.webm';
const sampleVideoFallbackPath = '/story-media/story-garden-demo-v1.mp4';
const artLabels = {flower:'花盆帽', cloud:'云朵帽', butterfly:'小蝴蝶', snail:'小蜗牛', garden:'小花园', picnic:'野餐', star:'小星星', moon:'月亮', robot:'机器人', boat:'小船', book:'故事书', cat:'小猫', dog:'小狗', rabbit:'小兔', bear:'小熊', child:'小朋友', dinosaur:'小恐龙', train:'小火车', ball:'皮球', paint:'画笔', blocks:'积木', seed:'种子'};
const sceneLabels = {shop:'帽子店', meadow:'故事小天地', forest:'森林', garden:'小花园', space:'星空', seaside:'海边', picnic:'野餐', home:'家里', school:'学校', park:'公园'};
const storySubjects = new Set(['butterfly','snail','robot','cat','dog','rabbit','bear','child','dinosaur','train','star','moon']);
let choiceButtons = [];
try {
  owner = localStorage.getItem('vq.story.browser');
  if (!/^[a-f0-9]{64}$/.test(owner || '')) {
    owner = Array.from(crypto.getRandomValues(new Uint8Array(32)), b => b.toString(16).padStart(2, '0')).join('');
    localStorage.setItem('vq.story.browser', owner);
  }
} catch (_) {
  durable = false;
  owner = Array.from(crypto.getRandomValues(new Uint8Array(32)), b => b.toString(16).padStart(2, '0')).join('');
}
function status(text) { $('status').textContent = text; }
function audioStatus(text, state = 'idle') {
  $('audio-status').textContent = text; $('audio-status').dataset.state = state;
  // This is a playback indicator, not a timer or a claim that synthesis played.
  const stage = $('story-stage');
  if (stage) stage.dataset.speaking = String(state === 'playing' && Boolean(current) &&
    stage.dataset.storyId === current.id && stage.dataset.revision === String(current.revision));
}
function clearAudioWatch() {
  ++audioGeneration; clearTimeout(audioTimer); audioTimer = null;
  const stage = $('story-stage'); if (stage) stage.dataset.speaking = 'false';
}
function readyHint() {
  if (cleanupPending) return '本次收音清理尚未确认，暂不开始下一轮；不会自动录音。';
  return current && current.phase === 'finished' ? '故事完成啦，可以再听一遍或重玩。' :
    current && current.can_continue === false ? '故事已经很丰富了，点结束故事一起收尾吧。' :
    current && current.adventure ? '轮到你来选，没有标准答案。' : '轮到你啦，点“说一句”接个新点子。';
}
function watchAudio(result, turnGeneration) {
  clearAudioWatch();
  if (result.audio_error) { audioPending = false; audioStatus(result.audio_error + '，可点“再听一遍”。', 'failed'); controls(); return; }
  if (!result.playback_id || !result.story) { audioPending = false; audioStatus('未收到朗读任务，请点“再听一遍”重试。', 'unknown'); controls(); return; }
  const watch = audioGeneration, storyId = result.story.id, revision = result.story.revision;
  const playbackId = result.playback_id, deadline = Date.now() + 120000;
  let failures = 0;
  audioPending = true; audioStatus('正在准备故事语音…', 'queued'); controls();
  const labels = {queued:'语音正在排队…', synthesizing:'正在准备故事语音…', playing:'芽芽正在讲这一页…',
    completed:'这一页已播放完。', cancelled:'朗读已停止。', stopped:'朗读已停止。', interrupted:'朗读已中断。',
    expired:'朗读等待过久，已过期。可点“再听一遍”。', failed:'故事语音失败了，可点“再听一遍”。'};
  async function poll() {
    if (watch !== audioGeneration || turnGeneration !== generation) return;
    try {
      const receipt = await api('audio-status', {story_id:storyId, revision, playback_id:playbackId});
      if (watch !== audioGeneration || turnGeneration !== generation || !current || current.id !== storyId || current.revision !== revision) return;
      if (receipt.playback_id !== playbackId || receipt.story_id !== storyId || receipt.revision !== revision) throw Error('朗读回执不匹配');
      const terminal = ['completed','cancelled','stopped','interrupted','expired','failed'].includes(receipt.state);
      if (receipt.ended === true && !terminal) throw Error('朗读终态未确认');
      failures = 0;
      if (terminal && receipt.ended !== true) audioStatus('正在等待播放结束确认，请稍等…', 'draining');
      else audioStatus(labels[receipt.state] || '朗读状态待确认…', receipt.state || 'unknown');
      if (receipt.ended === true) {
        audioPending = false; controls();
        status(['failed','expired'].includes(receipt.state) ? '文字已经保留，声音没有完成；可重新朗读或继续。' : readyHint());
        return;
      }
    } catch (_) {
      if (watch !== audioGeneration || turnGeneration !== generation) return;
      ++failures;
    }
    if (Date.now() >= deadline || failures >= 3) {
      // No automatic retry/capture or false completion after a lost receipt.
      audioStatus('朗读结束状态未确认，请点“停止”后再继续。', 'unknown');
      status('声音状态没有确认，不会自动重复播放或开启麦克风。');
      return;
    }
    audioTimer = setTimeout(poll, failures ? 1500 : 600);
  }
  audioTimer = setTimeout(poll, 0);
}
function api(action, data = {}) {return window.StarBuddyMode ? window.StarBuddyMode.track(()=>request(action,data)) : request(action,data);}
async function request(action, data = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ['start', 'continue', 'choose', 'finish', 'voice', 'voice-start'].includes(action) ? 120000 : 8000);
  try {
    const response = await fetch('/api/story/' + action, {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({owner, ...data}), signal: controller.signal});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || '故事工作台暂时不可用');
    return body;
  } finally { clearTimeout(timer); }
}
function controls() {
  const active = current && current.phase === 'active';
  const blocked = busy || audioPending || cleanupPending;
  const canContinue = active && !composerOpen && current.can_continue !== false && current.rounds < current.max_rounds && !blocked && !current.busy;
  for (const id of ['send', 'voice', 'contribution', 'write', 'free-send', 'free-contribution']) $(id).disabled = !canContinue;
  for (const button of choiceButtons) button.disabled = !canContinue;
  for (const id of ['start', 'custom-start', 'theme-voice', 'theme', 'new-story', 'return-story', 'restart']) $(id).disabled = blocked;
  $('replay').disabled = blocked || !current || current.busy || !current.messages.some(m => m.role === 'assistant');
  $('finish').disabled = blocked || !active || current.busy;
  $('finish').hidden = $('write').hidden = !current || composerOpen || current.phase === 'finished';
  $('write').hidden = $('write').hidden || !current?.adventure;
  $('save').hidden = !current || composerOpen || current.phase !== 'finished' || current.saved || current.private;
  $('save').disabled = blocked || !durable;
  $('save-option').disabled = busy || !durable || Boolean(current && current.private);
  if (!durable || (current && current.private)) $('save-option').checked = false;
}
function setArt(element, key) {
  // The server can only select a local drawing, never provide HTML or an image URL.
  const href = Object.hasOwn(artLabels, key) ? '#art-' + key : '';
  element.setAttribute('href', href);
  element.setAttributeNS('http://www.w3.org/1999/xlink', 'href', href);
}
function renderIllustration(story) {
  const stage = $('story-stage'), visual = story && story.illustration;
  const page = story ? story.messages.filter(message => message.role === 'assistant').length : 0;
  const valid = Boolean(story && visual && visual.source === 'local-vector-palette' &&
    visual.story_id === story.id && visual.revision === story.revision && visual.page === page &&
    Object.hasOwn(sceneLabels, visual.scene) && Array.isArray(visual.props) && visual.props.length <= 3 &&
    visual.props.every(key => Object.hasOwn(artLabels, key)) &&
    (visual.subject == null || (storySubjects.has(visual.subject) && visual.props.includes(visual.subject))));
  const scene = valid ? visual.scene : 'meadow';
  const props = valid ? visual.props : [];
  const subject = valid && !story.adventure ? (visual.subject === undefined ? props.find(key => storySubjects.has(key)) : visual.subject) : null;
  stage.hidden = !story;
  $('story-buddy').hidden = Boolean(story);
  stage.dataset.storyId = story ? story.id : '';
  stage.dataset.revision = story ? String(story.revision) : '';
  stage.dataset.page = String(page);
  stage.dataset.scene = scene;
  stage.dataset.speaking = 'false';
  stage.dataset.illustrationValid = String(valid);
  stage.dataset.subject = subject || '';
  $('scene-backdrop').setAttribute('href', '#scene-' + scene);
  const label = [sceneLabels[scene], ...props.filter(key => key !== scene).map(key => !story?.adventure && key === 'flower' ? '花朵' : artLabels[key])].join(' · ');
  $('chosen-path').textContent = '本地故事场景 · ' + label;
  $('scene-title').textContent = `这一页的故事插画：${label}`;
  setArt($('story-subject'), subject);
  $('story-subject').setAttribute('visibility', subject ? 'visible' : 'hidden');
  const guide = $('story-guide');
  for (const [key, value] of Object.entries(subject ? {x:12,y:12,width:90,height:90} : {x:146,y:45,width:208,height:192})) guide.setAttribute(key, String(value));
  for (const [index, id] of ['worn-hat','story-friend','story-activity'].entries()) {
    setArt($(id), props[index]);
    $(id).setAttribute('visibility', props[index] === subject ? 'hidden' : 'visible');
  }
}
function renderTextSource(story, result = null) {
  if (!story || storyTextSource?.id !== story.id || storyTextSource?.revision !== story.revision) storyTextSource = null;
  if (story && result && result.story?.id === story.id && result.story?.revision === story.revision && result.accepted !== false) {
    const source = result.backend === 'cloud-text-api' && result.local_only === false ? 'api' : result.local_only === true ? 'local' : null;
    if (source) storyTextSource = {id:story.id, revision:story.revision, source};
  }
  $('story-backend').textContent = storyTextSource?.source === 'api' ? '联网文字' : storyTextSource?.source === 'local' ? '本地文字' : '本地插画';
  $('story-backend').title = storyTextSource?.source === 'api' ? '这一页文字使用已授权的联网续写；插画来自本机素材。' :
    storyTextSource?.source === 'local' ? '这一页文字由本地生成；插画来自本机素材。' : '插画来自本机素材；此页文字来源没有在当前页面确认。';
}
function renderChoices(story) {
  const adventure = story && story.adventure;
  $('choice-panel').hidden = !adventure || story.phase === 'finished';
  $('story-book').classList.toggle('hat-story', Boolean(adventure));
  $('story-workspace').classList.toggle('hat-playing', Boolean(adventure));
  $('choices').replaceChildren(); choiceButtons = [];
  if (!adventure) return;
  $('choice-question').textContent = story.phase === 'finished' ? '这一页先轻轻合上' : story.question;
  $('choice-hint').textContent = '没有正确答案，选你喜欢的就好。也可以点“说一句”。';
  for (const choice of story.choices || []) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'choice-card';
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 100 100'); svg.setAttribute('aria-hidden', 'true');
    const drawing = document.createElementNS('http://www.w3.org/2000/svg', 'use'); setArt(drawing, choice.art); svg.append(drawing);
    const label = document.createElement('strong'); label.textContent = choice.label;
    const caption = document.createElement('small'); caption.textContent = choice.caption;
    button.append(svg, label, caption);
    const revision = story.revision, storyId = story.id;
    button.addEventListener('click', () => {
      if (busy || button.disabled || !current || current.id !== storyId || current.revision !== revision) return;
      run('choose', {...ids(), choice_id:choice.id, save:$('save-option').checked});
    });
    choiceButtons.push(button); $('choices').append(button);
  }
}
function sampleHint(text) { $('sample-status').textContent = text; }
function sampleSource(video) {
  if (video.canPlayType('video/webm; codecs="vp8"')) return sampleVideoPath;
  if (video.canPlayType('video/mp4; codecs="avc1.42E01E"')) return sampleVideoFallbackPath;
  return null;
}
function sampleButton(playing) {
  $('sample-play').textContent = playing ? 'Ⅱ 暂停示例' : '▶ 播放示例短片';
  $('sample-play').setAttribute('aria-pressed', String(playing));
}
function stopSample(reset = false) {
  ++sampleGeneration; sampleRequested = false;
  const video = $('story-sample-video'); video.pause();
  if (reset) { try { video.currentTime = 0; } catch (_) {} }
  sampleButton(false);
  sampleHint('静音示例，不是当前故事实时生成的视频。');
}
function renderSample() {
  const launch = !current || composerOpen, visible = launch || sampleOpen;
  $('sample-panel').hidden = !visible;
  $('story-hero').hidden = visible;
  // A deliberately opened sample gets a real viewing area, not the thin space
  // left over beside current-page prose. The story DOM/state remain intact.
  $('pages').hidden = visible;
  $('sample-close').hidden = launch;
  $('show-sample').hidden = launch || sampleOpen;
  $('story-workspace').classList.toggle('compose-mode', launch);
  $('story-workspace').classList.toggle('sample-mode', visible);
  if (!visible) stopSample();
}
async function playSample() {
  const video = $('story-sample-video');
  if ($('sample-panel').hidden || document.hidden) return;
  if (sampleRequested) { stopSample(); return; }
  const source = sampleSource(video);
  if (!source) {
    video.hidden = true; $('sample-panel').dataset.failed = 'true'; sampleButton(false);
    sampleHint('当前浏览器不支持示例短片格式；故事插画仍可正常查看。');
    return;
  }
  const mine = ++sampleGeneration;
  sampleRequested = true;
  video.muted = true; video.defaultMuted = true; video.volume = 0; video.playsInline = true;
  video.hidden = false; $('sample-panel').dataset.failed = 'false';
  if (video.getAttribute('src') !== source) { video.setAttribute('src', source); video.load(); }
  else if (video.error) video.load();
  sampleButton(true); sampleHint('正在加载静音示例…');
  try {
    await video.play();
    // A delayed old play() must not pause a newer explicit play request.
    if (mine !== sampleGeneration) { if (!sampleRequested) video.pause(); return; }
    if (!sampleRequested || document.hidden || $('sample-panel').hidden) { video.pause(); return; }
    if (!video.videoWidth || video.readyState < 2) throw Error('示例视频尚未解码');
    sampleHint('正在播放预先制作的示例 · 无声音，不是当前故事实时生成。');
  } catch (_) {
    if (mine !== sampleGeneration) return;
    sampleRequested = false; video.pause(); video.hidden = true;
    $('sample-panel').dataset.failed = 'true'; sampleButton(false);
    sampleHint('示例短片未能播放，可重试；不影响编故事。');
  }
}
function openComposer() {
  if (busy || audioPending || cleanupPending) return;
  stopSample(true); sampleOpen = false; composerOpen = true; render(current);
  status('输入主题，或点“说出主题”后确认文字；不会自动录音。');
  $('theme').focus();
}
function paragraphs(target, messages) {
  target.replaceChildren();
  for (const message of messages) {
    const block = document.createElement('div');
    block.className = 'paragraph ' + (message.role === 'user' ? 'user' : 'buddy-text');
    const author = document.createElement('small');
    author.textContent = message.role === 'user' ? '你 / 故事指令' : '星伴';
    const text = document.createElement('p');
    text.textContent = message.content;
    block.append(author, text); target.append(block);
  }
  target.scrollTop = target.scrollHeight;
}
function render(story) {
  if (current?.id !== story?.id || current?.revision !== story?.revision) { stopSample(true); sampleOpen = false; }
  current = story;
  const launch = !story || composerOpen;
  $('launch-panel').hidden = !launch;
  $('return-story').hidden = !story;
  $('free-panel').hidden = launch || Boolean(story.adventure) || story.phase === 'finished';
  $('finished-panel').hidden = launch || story.phase !== 'finished';
  $('restart').hidden = launch || story.phase !== 'finished';
  $('story-actions').hidden = $('story-mic-note').hidden = launch;
  $('book-title').textContent = story ? story.theme : '一个点子，就是故事的开始';
  $('round').textContent = story ? `${story.phase === 'finished' ? '已完成' : '接龙中'} · ${story.rounds} / ${story.max_rounds} 轮${story.private ? ' · 不保存' : story.saved ? ' · 已存本机' : ''}` : '准备好了';
  if (story && story.adventure) {
    const step = {hat:'1 / 3 · 选帽子', friend:'2 / 3 · 选伙伴', activity:'3 / 3 · 选活动', done:'已完成'}[story.adventure.step];
    $('round').textContent = `${story.phase === 'finished' ? '已完成 · 帽子店小冒险' : step}${story.private ? ' · 不保存' : story.saved ? ' · 已存本机' : ''}`;
  }
  renderChoices(story);
  if (launch) $('choice-panel').hidden = true;
  renderIllustration(story);
  renderTextSource(story);
  const latest = story && [...story.messages].reverse().find(message => message.role === 'assistant');
  if (latest) { paragraphs($('pages'), [latest]); $('pages').scrollTop = 0; }
  else { $('pages').replaceChildren(); const empty = document.createElement('p'); empty.className = 'empty'; empty.textContent = '把你的一个点子，变成我们一起讲的故事。'; $('pages').append(empty); }
  $('history').disabled = !story || !story.messages.length;
  $('history').hidden = launch;
  renderSample();
  controls();
}
async function voiceStartControl() {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch('/api/interaction/state', {cache:'no-store', signal:controller.signal});
    if (!response.ok) throw Error('本次收音尚未就绪，请稍后再试。');
    const state = await response.json();
    if (!state || typeof state.control_epoch !== 'string' || !/^[a-f0-9]{32}$/.test(state.control_epoch) ||
        !Number.isSafeInteger(state.control_revision) || state.control_revision < 0) throw Error('未收到有效的收音状态，麦克风没有开启。');
    return {control_epoch:state.control_epoch, control_revision:state.control_revision};
  } finally { clearTimeout(timer); }
}
async function run(action, data) {
  if (busy || cleanupPending || (action !== 'save' && audioPending)) return;
  const mine = ++generation;
  clearAudioWatch(); stopSample();
  let submitted = false, capture = 0;
  const voice = action === 'voice' || action === 'voice-start';
  busy = true; controls();
  status(voice ? '正在准备本次收音，说完后自然停顿一下…' : action === 'replay' ? '正在准备朗读这一页…' : action === 'choose' ? '故事跟着你的选择往前走…' : '芽芽正在续写，可以随时停止…');
  if (action !== 'save') audioStatus(voice ? '只进行这一次收音，不自动续听。' : '正在准备这一页…', 'preparing');
  try {
    if (action === 'voice-start') {
      const control = await voiceStartControl();
      if (mine !== generation) return;
      data = {...data, ...control};
    }
    if (voice) { capture = ++captureGeneration; cleanupPending = true; }
    submitted = true;
    const result = await api(action, data);
    if (voice && capture === captureGeneration && result.input_cleanup_confirmed === true) {
      cleanupPending = false; controls();
      if (mine !== generation && !busy) status('本次收音已清理；不会自动开启麦克风。');
    }
    if (mine !== generation) return;
    if (action === 'voice-start' && result.accepted !== false) {
      // ASR is a draft, never permission to generate or replace the current story.
      composerOpen = true; audioPending = false; render(current);
      const validDraft = typeof result.draft_theme === 'string' && result.draft_theme.trim().length > 0;
      if (validDraft) $('theme').value = result.draft_theme;
      const hint = cleanupPending ? '本次收音清理尚未确认，暂不开始下一轮。' : validDraft ?
        '请确认或修改主题，再点“开始我的故事”。' : '未收到主题草稿，没有开始编故事；请重新说出或输入主题。';
      audioStatus(hint, cleanupPending || !validDraft ? 'unknown' : 'idle'); status(hint);
      return;
    }
    if (action === 'start' && result.accepted !== false && result.story) composerOpen = false;
    if (result.story) render(result.story);
    // Replay is local audio reuse, not a new provenance claim for the text.
    if (['start','continue','choose','finish','voice'].includes(action)) renderTextSource(current, result);
    if (action === 'continue' || action === 'voice') { $('contribution').value = ''; $('free-contribution').value = ''; }
    if (result.accepted === false) {
      audioPending = false;
      audioStatus(cleanupPending ? '本次收音清理尚未确认，暂不开始下一轮。' : result.reason === 'stale_control' ? '本次收音已取消，请重新点击。' : '没有听清，本次收音已结束；可再试一次。', cleanupPending ? 'unknown' : 'idle');
    }
    else if (action !== 'save') watchAudio(result, mine);
    status(cleanupPending ? '本次收音清理尚未确认，暂不开始下一轮。' : result.save_error || (result.accepted === false ? result.reason === 'stale_control' ? '收音状态已经变化，请重新点击。' : '没有听清，请再试一次。' : current && current.saved ? '已保存在本机故事册。' : readyHint()));
  } catch (error) {
    if (mine !== generation) return;
    // Cancel on uncertain timeout; never auto-retry or double-generate a turn.
    if (submitted) { try { const result = await api('stop'); if (mine === generation) render(result.story); } catch (_) {} }
    if (mine === generation) { audioPending = submitted; audioStatus(cleanupPending ? '本次收音清理尚未确认，暂不开始下一轮。' : submitted ? '本轮状态未确认，请点“停止”后继续。' : '麦克风没有开启，可稍后重试。', 'unknown'); status(error.name === 'AbortError' ? '等待超时，未自动重试。' : error.message); }
  } finally { if (mine === generation) { busy = false; controls(); } }
}
async function stop() {
  const mine = ++generation;
  clearAudioWatch(); stopSample();
  busy = true; controls();
  try {
    const result = await api('stop');
    if (mine === generation) { audioPending = false; render(result.story); audioStatus(cleanupPending ? '已请求停止；本次收音清理尚未确认。' : '已发出停止请求，不会自动开启麦克风。', cleanupPending ? 'unknown' : 'cancelled'); status(cleanupPending ? '本次收音清理尚未确认，暂不开始下一轮。' : '已停止。已写好的内容还在，可以继续。'); }
  } catch (_) { if (mine === generation) status('停止请求未确认，请检查本地连接。'); }
  finally { if (mine === generation) { busy = false; controls(); } }
}
function ids() { return {story_id: current.id, revision: current.revision}; }
document.querySelectorAll('[data-theme]').forEach(button => button.addEventListener('click', () => {
  $('theme').value = button.dataset.theme;
  closeDialog('settings-dialog'); openComposer();
}));
function openDialog(id) { const dialog = $(id); if (!dialog.open) dialog.showModal(); }
function closeDialog(id) { if ($(id).open) $(id).close(); }
document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => closeDialog(button.dataset.close)));
$('options').addEventListener('click', () => openDialog('settings-dialog'));
$('pick-theme').addEventListener('click', () => openDialog('settings-dialog'));
$('write').addEventListener('click', () => { if (!$('write').disabled) openDialog('write-dialog'); });
$('history').addEventListener('click', () => {
  if (!current) return;
  paragraphs($('history-pages'), current.messages);
  openDialog('history-dialog');
});
function startStory(theme) {
  if (composing || !theme.trim() || busy || audioPending || cleanupPending) return;
  if (theme.trim().length > 100) { status('主题较长，请修改到 100 字以内，再开始故事。'); return; }
  if (current && !current.saved && current.messages.length && !confirm('开始新故事会清除当前未保存的故事，继续吗？')) return;
  closeDialog('settings-dialog');
  run('start', {text:theme.trim()});
}
for (const id of ['theme', 'contribution', 'free-contribution']) {
  $(id).addEventListener('compositionstart', () => { composing = true; });
  $(id).addEventListener('compositionend', () => { composing = false; });
  $(id).addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.isComposing || composing || event.keyCode === 229)) event.stopPropagation();
  });
}
$('start-form').addEventListener('submit', event => {
  event.preventDefault();
  startStory($('theme').value);
});
$('start').addEventListener('click', () => startStory('小怪兽的奇怪帽子店'));
$('new-story').addEventListener('click', openComposer);
$('return-story').addEventListener('click', () => { if (busy || audioPending || cleanupPending) return; composerOpen = false; stopSample(true); sampleOpen = false; render(current); status(readyHint()); });
$('theme-voice').addEventListener('click', () => {
  if ($('theme-voice').disabled) return;
  run('voice-start', {});
});
$('restart').addEventListener('click', () => startStory(current ? current.theme : '小怪兽的奇怪帽子店'));
$('continue-form').addEventListener('submit', event => {
  event.preventDefault();
  if (composing || !current || !$('contribution').value.trim() || $('send').disabled) return;
  closeDialog('write-dialog');
  run('continue', {...ids(), text: $('contribution').value.trim()});
});
$('free-continue-form').addEventListener('submit', event => {
  event.preventDefault();
  if (composing || !current || !$('free-contribution').value.trim() || $('free-send').disabled) return;
  run('continue', {...ids(), text:$('free-contribution').value.trim()});
});
$('sample-play').addEventListener('click', playSample);
$('show-sample').addEventListener('click', () => { sampleOpen = true; renderSample(); playSample(); });
$('sample-close').addEventListener('click', () => { stopSample(true); sampleOpen = false; renderSample(); });
const sampleVideo = $('story-sample-video');
sampleVideo.addEventListener('ended', () => { sampleRequested = false; sampleButton(false); sampleHint('示例已结束；当前故事仍使用本地插画。'); });
sampleVideo.addEventListener('playing', () => { if (!sampleRequested || document.hidden || $('sample-panel').hidden) sampleVideo.pause(); });
sampleVideo.addEventListener('error', () => {
  if ($('sample-panel').hidden) return;
  ++sampleGeneration; sampleRequested = false; sampleVideo.pause(); sampleVideo.hidden = true;
  $('sample-panel').dataset.failed = 'true'; sampleButton(false); sampleHint('示例短片暂时不可用；你仍然可以正常编故事。');
});
document.addEventListener('visibilitychange', () => { if (document.hidden) stopSample(); });
$('voice').addEventListener('click', () => { if (current && !$('voice').disabled) run('voice', ids()); });
$('replay').addEventListener('click', () => { if (current && !$('replay').disabled) run('replay', ids()); });
$('finish').addEventListener('click', () => { if (current && !$('finish').disabled) run('finish', {...ids(), save: $('save-option').checked}); });
$('save').addEventListener('click', () => { if (current && !$('save').disabled) run('save', {story_id: current.id}); });
$('stop').addEventListener('click', () => window.StarBuddyMode ? window.StarBuddyMode.stop() : stop());
$('back').addEventListener('click', async event => { event.preventDefault(); await stop(); location.assign('/'); });
window.addEventListener('pagehide', () => {
  ++generation; clearAudioWatch(); stopSample();
  // Best effort for browser back/close; normal Back button awaits acknowledgement.
  navigator.sendBeacon('/api/story/stop', new Blob([JSON.stringify({owner})], {type: 'application/json'}));
});
async function library() {
  openDialog('shelf');
  try {
    const result = await api('list');
    $('books').replaceChildren();
    if (!result.stories.length) $('books').textContent = '还没有保存的故事。';
    for (const book of result.stories) {
      const row = document.createElement('div'); row.className = 'book-row';
      const title = document.createElement('span'); title.textContent = book.theme;
      const open = document.createElement('button'); open.textContent = '打开'; open.type = 'button';
      open.addEventListener('click', async () => {
        try { const saved = (await api('read', {story_id: book.id})).story;
          $('saved-book').hidden = false; $('saved-title').textContent = saved.theme;
          paragraphs($('saved-pages'), saved.messages);
        } catch (error) { status(error.message); }
      });
      const remove = document.createElement('button'); remove.textContent = '删除'; remove.type = 'button'; remove.className = 'quiet';
      remove.addEventListener('click', async () => {
        if (!confirm('删除这篇本机故事？删除后不能恢复。')) return;
        try { await api('delete', {story_id: book.id}); $('saved-book').hidden = true;
          if (current && current.id === book.id) { current.saved = false; controls(); }
          await library(); status('已删除这篇本机故事，无法恢复。');
        } catch (error) { status(error.message); }
      });
      row.append(title, open, remove); $('books').append(row);
    }
  } catch (error) { status(error.message); }
}
$('library').addEventListener('click', library);
$('close-library').addEventListener('click', () => { closeDialog('shelf'); $('saved-book').hidden = true; });
busy = true; controls();
const initialGeneration=generation;
api('status').then(result => { if(initialGeneration!==generation)return;render(result.story); status(durable ? '输入主题，或点“说出主题”后确认文字；不会自动录音。' : '浏览器储存不可用，本次只能临时创作。'); })
  .catch(() => status('故事工作台未连接，请稍后刷新。'))
  .finally(() => { if (initialGeneration === generation) { busy = false; controls(); } });
window.StarBuddyMode?.configure({name:'共创故事',
  freeze(){++generation;clearAudioWatch();stopSample();busy=true;controls();},
  async stop(){const result=await api('stop');render(result.story);},
  ready:window.StarBuddyMode.ready,
  resume(){busy=false;audioPending=false;controls();audioStatus(cleanupPending ? '已请求停止；本次收音清理尚未确认。' : '续写和声音已停止，不会自动开启麦克风。', cleanupPending ? 'unknown' : 'cancelled');status(cleanupPending ? '本次收音清理尚未确认，暂不开始下一轮。' : '已有故事还在，可以继续。');},
  stopped:'已停止续写和声音；未自动保存故事。'});
