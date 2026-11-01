'use strict';
const $ = id => document.getElementById(id);
let owner, programmes = [], current = null, selected = 'moon-pocket', mode = 'bedtime';
let busy = false, timer = null, errors = 0, available = false, occupied = false, leaving = false;
let appointment = null, scheduleAvailable = false, scheduleOccupied = false, retrySchedule = null;
try { owner = localStorage.getItem('vq.quiet.browser'); } catch (_) {}
if (!/^[a-f0-9]{64}$/.test(owner || '')) {
  owner = Array.from(crypto.getRandomValues(new Uint8Array(32)), b => b.toString(16).padStart(2, '0')).join('');
  try { localStorage.setItem('vq.quiet.browser', owner); } catch (_) {}
}
const reasons = {conversation:'聊天优先，节目已暂停；不会自行抢话。', output_busy:'扬声器正忙，请稍后点继续。', your_turn:'可以慢慢想。想听答案时，再点继续。', page_missing:'页面离开后已暂停，请手动继续。', user_pause:'已暂停；继续时会从本段开头播放。', user_stop:'已停止。', session_limit:'本次已达到 30 分钟上限，已停止。', submission_unknown:'播放提交结果不明，已停止自动重试。请检查 Output 服务。', output_unavailable:'播放服务暂不可用，未自动重播。', cleanup_unconfirmed:'尚未确认声音已停，不会开始新节目。'};
function message(text) { $('status').textContent = text; }
function api(action,data={}){return window.StarBuddyMode ? window.StarBuddyMode.track(()=>request(action,data)) : request(action,data);}
async function request(action, data = {}) {
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 4000);
  try {
    const response = await fetch('/api/quiet/' + action, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({owner,...data}), signal:controller.signal});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || '本地播放服务暂不可用');
    return result;
  } finally { clearTimeout(timeout); }
}
function renderCards() {
  $('catalogue').replaceChildren();
  for (const programme of programmes.filter(p => p.mode === mode)) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'card' + (selected === programme.id ? ' selected' : '');
    button.setAttribute('aria-pressed', String(selected === programme.id));
    const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 100 100'); svg.setAttribute('aria-hidden','true');
    const use = document.createElementNS('http://www.w3.org/2000/svg','use');
    const href = ['moon','leaf','star','shell'].includes(programme.symbol) ? '#' + programme.symbol : '';
    use.setAttribute('href',href); use.setAttributeNS('http://www.w3.org/1999/xlink','href',href); svg.append(use);
    const words = document.createElement('span'), title = document.createElement('strong'), detail = document.createElement('small');
    title.textContent = programme.title; detail.textContent = programme.description; words.append(title,detail); button.append(svg,words);
    button.addEventListener('click',() => { selected = programme.id; renderCards(); render(); });
    $('catalogue').append(button);
  }
}
function render() {
  const active = current && ['playing','paused'].includes(current.state);
  const pending = Boolean(current && current.pending_output);
  const locked = Boolean(current && current.quarantined);
  const showCurrent = current && (active || pending || current.program_id === selected);
  const programme = programmes.find(p => p.id === (showCurrent ? current.program_id : selected));
  $('title').textContent = programme ? programme.title : '选择一个节目';
  $('progress').textContent = showCurrent ? `第 ${current.paragraph} / ${current.paragraphs} 段 · ${({'playing':'播放中','paused':'暂停','completed':'已播完','stopped':'已停止','error':'需检查'})[current.state] || current.state}` : '选一个节目开始';
  $('transcript').textContent = showCurrent ? current.text : '这里会显示正在朗读的小段文字。';
  $('play').disabled = busy || !available || occupied || locked;
  $('volume').disabled = busy || occupied;
  $('speed').disabled = busy || occupied;
  if (current && (active || pending) && [0.6,0.8,1].includes(current.speed)) $('speed').value=String(current.speed);
  $('schedule-create').disabled = busy || !scheduleAvailable || scheduleOccupied;
  $('schedule-cancel').disabled = busy || !appointment || !['submitting','scheduled'].includes(appointment.state);
  const labels = {submitting:'提交未确认，可取消后重试',scheduled:'已预约',cancelled:'已取消',missed:'已过期或未送达，不补播',claimed:'已处理，不会重播',started:'已启动（播放进度见上方）',skipped_busy:'当时正在互动或有节目，已跳过'};
  $('schedule-status').textContent = appointment ? `${labels[appointment.state] || appointment.state} · ${new Date(appointment.run_at*1000).toLocaleString()} · ${appointment.speed}×` : scheduleOccupied ? '另一页面已有预约。' : scheduleAvailable ? '没有待播预约。' : '预约服务未就绪。';
  $('pause').disabled = busy || !current || current.state !== 'playing';
  $('resume').disabled = busy || !current || current.state !== 'paused' || pending || locked;
  $('resume').textContent = current && current.reason === 'your_turn' ? '听听答案' : '从本段继续';
  $('next').disabled = busy || !active || pending && current.state !== 'playing' || locked;
  $('stop').disabled = busy || !current || (!active && !pending);
  if (current) message(pending && current.state !== 'playing' ? '正在停止声音，请稍等…' : reasons[current.reason] || (current.state === 'completed' ? '听完啦。可以换一个，也可以安静一会儿。' : current.state === 'playing' ? '轻轻听。想停的时候，点暂停或停止。' : '节目已暂停，可以稍后继续。'));
  else message(!available ? 'Output 未就绪；不会直接绕路播放。' : occupied ? '另一页面正在听节目。' : '已就绪。选择节目后，点开始听。');
}
function apply(result) { current = result.player; available = result.available; occupied = result.occupied;
  appointment = result.schedule || null; scheduleAvailable = Boolean(result.schedule_available); scheduleOccupied = Boolean(result.schedule_occupied); render(); }
function schedule() {
  clearTimeout(timer);
  if (!leaving && !document.hidden && errors < 3) {
    if (current && !current.quarantined && (current.state === 'playing' || current.pending_output)) timer = setTimeout(refresh,2000);
    else if (scheduleOccupied) timer = setTimeout(refresh,10000);
  }
}
async function refresh() {
  if (busy || leaving) return;
  busy = true;
  try { const result = await api('status'); busy = false; apply(result); errors = 0; }
  catch (_) { errors++; message('连接中断，未重复发起播放；页面失联后节目会暂停。可点刷新状态。'); }
  finally { busy = false; schedule(); }
}
async function run(action) {
  if (busy || leaving) return;
  busy = true; clearTimeout(timer); render();
  try {
    const result = action === 'start' ? await api('start',{program_id:selected,volume_percent:Number($('volume').value),speed:Number($('speed').value)}) : await api('control',{session_id:current.id,revision:current.revision,action});
    busy = false; errors = 0; apply(result);
  } catch (error) { busy = false; render(); message(error.message || '操作未确认，请刷新状态。'); }
  finally { schedule(); }
}
for (const value of ['bedtime','radio']) $(value).addEventListener('click',() => {
  mode = value; $('bedtime').setAttribute('aria-pressed',String(mode === 'bedtime')); $('radio').setAttribute('aria-pressed',String(mode === 'radio'));
  document.body.classList.toggle('night',mode === 'bedtime');
  selected = (programmes.find(p => p.mode === mode) || {}).id; renderCards(); render();
});
$('play').addEventListener('click',() => run('start'));
for (const action of ['pause','resume','next','stop']) $(action).addEventListener('click',() => { if (!$(action).disabled && current) {if(action==='stop'&&window.StarBuddyMode)window.StarBuddyMode.stop();else run(action);} });
$('refresh').addEventListener('click',refresh);
$('delay').addEventListener('change',() => { $('schedule-time').hidden = $('delay').value !== 'at'; });
async function book(cancel=false) {
  if (busy || leaving) return;
  busy=true; clearTimeout(timer); render();
  try {
    let result;
    if (cancel) result = await api('cancel-schedule',{task_id:appointment.task_id});
    else {
      if (!retrySchedule) {
        const runAt = $('delay').value === 'at' ? new Date($('schedule-time').value).getTime()/1000 : Date.now()/1000+Number($('delay').value);
        if (!Number.isFinite(runAt) || runAt-Date.now()/1000 < 30 || runAt-Date.now()/1000 > 86400) throw Error('请选择未来30秒至24小时内的时间。');
        retrySchedule={program_id:selected,volume_percent:Number($('volume').value),speed:Number($('speed').value),run_at:runAt,
          request_id:Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('')};
      }
      result=await api('schedule',retrySchedule);
    }
    retrySchedule=null; busy=false; errors=0; apply(result);
  } catch(error) { busy=false; render(); message(error.message || '预约未确认，请刷新状态。'); }
  finally { schedule(); }
}
$('schedule-create').addEventListener('click',() => book());
$('schedule-cancel').addEventListener('click',() => { if (appointment) book(true); });
function leave() {
  leaving = true; clearTimeout(timer);
  if (current) navigator.sendBeacon('/api/quiet/control',new Blob([JSON.stringify({owner,session_id:current.id,revision:current.revision,action:'stop'})],{type:'application/json'}));
}
$('back').addEventListener('click',event => { event.preventDefault(); leave(); location.assign('/'); });
window.addEventListener('pagehide',leave);
window.addEventListener('pageshow',() => { leaving = false; refresh(); });
document.addEventListener('visibilitychange',() => { if (document.hidden) { clearTimeout(timer); if (current && current.state === 'playing') run('pause'); } else refresh(); });
async function init() {
  document.body.classList.toggle('night',true);
  try { const response = await fetch('/api/quiet/catalogue'); if (!response.ok) throw Error(); programmes = (await response.json()).programmes; renderCards(); await refresh(); }
  catch (_) { message('节目目录未加载，请刷新页面重试。'); }
}
init();
window.StarBuddyMode?.configure({name:'晚安 / 电台',
  freeze(){leaving=true;clearTimeout(timer);},
  async stop(){const result=await api('status');current=result.player;
    if(current)apply(await api('control',{session_id:current.id,revision:current.revision,action:'stop'}));},
  async ready(){const result=await api('status');apply(result);return !current || (!current.pending_output&&!current.quarantined&&current.state!=='playing');},
  resume(){leaving=false;busy=false;render();schedule();},
  stopped:'当前节目已停止。未来预约仍保留，可在下方单独取消。'});
