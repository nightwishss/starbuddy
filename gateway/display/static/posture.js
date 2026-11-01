/* This page owns a preview only. The service owns the persistent guardian. */
const $=id=>document.getElementById(id);
const app=$('app'),gauge=$('gauge'),progress=$('progress'),gaugeLabel=$('gauge-label');
const title=$('state-title'),message=$('state-message'),source=$('source'),success=$('success'),latency=$('latency');
const reason=$('reason'),cameraPill=$('camera-pill'),calibrate=$('calibrate'),simulate=$('simulate'),reset=$('reset');
const postureBack=$('posture-back'),monitorToggle=$('monitor-toggle'),monitorState=$('monitor-state');
const reminderStatus=$('reminder-status'),requestStatus=$('request-status'),boundary=$('posture-boundary');
const reasonNames={torso_lean:'上半身前倾',head_forward:'头部前伸',shoulder_asymmetry:'肩膀不对称',too_close:'距离过近'};
const deferredReasons={conversation_busy:'正在对话，空闲后再提醒',robot_speaking:'其他语音正在播放，稍后再提醒',
  user_snoozed:'已按你的要求暂缓提醒',resource_busy:'音频资源正在使用，稍后再提醒',compute_busy:'计算资源正在使用，稍后再提醒',
  resources_unknown:'资源状态暂未确认',output_unknown:'语音通道状态暂未确认',interaction_unknown:'对话状态暂未确认',
  intent_unknown:'互动状态暂未确认',ambiguous_subject:'画面中有多人，暂不作个人提醒',
  fusion_unavailable:'互动服务暂未连接',posture_stale:'姿态画面已过期，等待新观测',
  reservation_busy:'另一个互动正在处理',quiet_mode:'当前为安静模式'};
let framePending=false,frameWatchdog=null,frameTimer=null,frameGeneration=0;
let leaving=false,monitorEnabled=null,actionPending=false,lastStatus=null;
let refreshTimer=null,refreshPending=false,statusController=null;

function post(path){return window.StarBuddyMode ? window.StarBuddyMode.track(()=>requestPost(path)) : requestPost(path);}
async function fetchJson(path,options={}){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),5000);
  try{
    const r=await fetch(path,{cache:'no-store',...options,signal:controller.signal});
    if(!r.ok)throw Error('服务连接未确认');
    const data=await r.json();if(!data||typeof data!=='object')throw Error('状态未确认');
    return data;
  }finally{clearTimeout(timer);}
}
function requestPost(path){return fetchJson(path,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}
function setReason(icon,label,detail){
  reason.querySelector('i').textContent=icon;reason.querySelector('b').textContent=label;
  reason.querySelector('small').textContent=detail;
}
function renderReminder(data){
  const state=data.reminder_delivery||'idle',pending=data.reminder_receipt_pending===true;
  reminderStatus.dataset.state=state;
  if(['completed','cancelled','interrupted','expired','failed'].includes(state)&&pending){
    reminderStatus.textContent='提醒正在结束，等待播放结束确认。';return;
  }
  const states={queued:'提醒已排队，等待语音播放。',preparing:'正在准备语音提醒。',synthesizing:'正在准备语音提醒。',
    playing:'正在播放坐姿提醒。',completed:'提醒播放已完成。',failed:'提醒播放失败；姿态仍需提醒时会稍后重试。',
    cancelled:'本次提醒已取消。',interrupted:'本次提醒已中断。',expired:'本次提醒已过期。',
    submission_unknown:'提醒提交尚未确认，不会重复提交。',unknown:'提醒播放状态尚未确认，不会重复提交。'};
  if(state==='deferred'){
    reminderStatus.textContent='语音暂缓：'+(deferredReasons[data.reminder_deferred_reason]||'正在等待当前互动或资源空闲。');
  }else if(states[state])reminderStatus.textContent=states[state];
  else if(data.reminder_inflight===true)reminderStatus.textContent='正在准备语音提醒。';
  else if(data.reminder_pending===true)reminderStatus.textContent='姿态已持续偏离，等待语音提醒。';
  else reminderStatus.textContent=monitorEnabled===false?'守护已暂停，不会新发起坐姿提醒。':'需要调整坐姿时，会轻声提醒；对话时不打断。';
}
function render(data){
  lastStatus=data;
  if(typeof data.enabled==='boolean')monitorEnabled=data.enabled;
  const startFailed=Boolean(data.monitor_start_error);
  const observedState=(['healthy','suspect','unhealthy'].includes(data.state)&&
    typeof data.source_state==='string'&&data.source_state!=='fresh')?'unknown':data.state;
  const state=monitorEnabled===false?(startFailed?'model_unavailable':'paused'):observedState||'model_unavailable';
  const p=Math.max(0,Math.min(1,Number(data.progress)||0));
  const confirmation=Number.isFinite(data.confirmation_seconds)?data.confirmation_seconds:20;
  app.dataset.state=state;gauge.style.setProperty('--p',`${p*360}deg`);
  monitorToggle.disabled=actionPending||monitorEnabled===null;
  monitorToggle.textContent=monitorEnabled===true?'暂停守护':'开启坐姿守护';
  monitorToggle.setAttribute('aria-pressed',String(monitorEnabled===true));
  monitorState.textContent=monitorEnabled===true?'后台守护已开启 · 切换应用也会继续':startFailed?'后台守护尚未启动 · 可点开启重试':monitorEnabled===false?'守护已暂停 · 校准仍保留':'正在确认守护状态';
  const cameraReady=Boolean(data.camera&&data.camera.ready);
  cameraPill.querySelector('b').textContent=monitorEnabled===false?'预览已暂停':cameraReady?'视频流正常':'等待视频流';
  cameraPill.classList.toggle('offline',!cameraReady||monitorEnabled!==true);
  const pose=data.pose||{},timing=pose.latency_ms||{};
  source.textContent=data.simulation?'模拟信号':pose.ready?'板端姿态模型':'等待模型';
  success.textContent=pose.window_frames?`${Math.round((pose.upper_body_success_rate||0)*100)}%`:'--';
  latency.textContent=timing.median==null?'--':`${Math.round(timing.median)} ms`;
  boundary.textContent=`先校准；持续偏离 ${confirmation} 秒才提醒，看不清时不作判断。`;
  if(data.calibrated===true&&data.low_head_calibrated===false){
    boundary.textContent=`原基线仍在使用，头肩前倾等提醒不受影响；仅新增低头线索尚未校准。可选重新校准会更新完整基线。可靠偏离累计 ${confirmation} 秒才提醒。`;
  }
  if(state==='paused'){
    progress.textContent='暂停';gaugeLabel.textContent='校准已保留';title.textContent='坐姿守护已暂停';
    message.textContent='点击“开启坐姿守护”后恢复；返回主页不会擅自改变你的设置。';
    setReason('—','当前不作姿势判断','切换应用不会自动开启或关闭守护');
  }else if(state==='calibrating'){
    progress.textContent=`${Math.round(p*100)}%`;gaugeLabel.textContent='建立个人基线';
    title.textContent='保持现在的健康坐姿';message.textContent=data.message||'请自然坐直，不需要刻意僵硬。';
    setReason('•',`已采集 ${data.calibration_samples||0} 帧`,'校准期间请尽量保持稳定');
  }else if(state==='suspect'){
    progress.textContent=`${Math.round(p*100)}%`;gaugeLabel.textContent='持续确认中';
    title.textContent='姿势与基线有些不同';message.textContent=data.message||'保持观察，暂时不会提醒。';
    setReason('!',(data.reasons||[]).map(x=>reasonNames[x]||x).join('、')||'轻微偏离',`可靠偏离累计达到 ${confirmation} 秒才提醒`);
  }else if(state==='unhealthy'){
    progress.textContent='调整';gaugeLabel.textContent='该调整坐姿啦';
    title.textContent='肩膀放松，坐直一点';message.textContent=data.message||'姿势已持续偏离个人基线，可以稍微调整一下。';
    setReason('!',(data.reasons||[]).map(x=>reasonNames[x]||x).join('、')||'姿态持续偏离','提醒会遵守语音状态和冷却时间');
  }else if(state==='healthy'){
    progress.textContent='正常';gaugeLabel.textContent='接近个人基线';title.textContent='现在坐得很好';
    message.textContent=data.message||'继续保持舒服、自然的坐姿。';
    setReason('✓','没有持续偏离','判断依据是你的个人校准基线');
  }else if(state==='observing'){
    const rate=Math.round((pose.upper_body_success_rate||0)*100);
    progress.textContent=`${rate}%`;gaugeLabel.textContent='上半身跟踪';title.textContent='姿态关键点已识别';
    message.textContent=data.message||'模型正在板端持续观察。';
    setReason('✓','头部和双肩清晰',data.calibrated?'正在使用已保存的个人基线':'请先校准个人健康坐姿');
  }else{
    progress.textContent='--';gaugeLabel.textContent='等待画面';title.textContent='请进入取景区域';
    message.textContent=data.message||'正在等待姿态模型和摄像头画面。';
    if(startFailed){title.textContent='守护暂未启动';message.textContent='相机或姿态服务可能还在准备；已保存的基线不变，可以点“开启坐姿守护”重试。';}
    setReason('—','当前无姿势结论','看不清或画面过期时不作判断');
  }
  renderReminder(data);
  calibrate.disabled=actionPending||monitorEnabled!==true||state==='calibrating'||Boolean(data.simulation);
  calibrate.querySelector('span').textContent=state==='calibrating'?'校准中…':data.calibrated?'重新校准（可选）':'校准健康坐姿';
  simulate.disabled=actionPending||monitorEnabled!==true||Boolean(data.simulation)||state==='calibrating'||state==='unhealthy';
  reset.disabled=actionPending;
  if(monitorEnabled===true&&!leaving)ensureCamera();else stopPreview();
}
function stopPreview(){
  ++frameGeneration;clearTimeout(frameWatchdog);clearTimeout(frameTimer);framePending=false;
  $('camera').removeAttribute('src');
}
function ensureCamera(){
  // Preview retries never re-enable the persistent guardian or steal a skill lease.
  if(!leaving&&monitorEnabled===true&&!document.hidden)requestFrame();
}
function requestFrame(){
  if(leaving||monitorEnabled!==true||document.hidden||framePending)return;
  clearTimeout(frameTimer);framePending=true;
  const camera=$('camera'),generation=++frameGeneration;
  const complete=()=>{
    if(!framePending||generation!==frameGeneration)return;
    framePending=false;clearTimeout(frameWatchdog);frameTimer=setTimeout(requestFrame,120);
  };
  camera.onload=complete;camera.onerror=complete;
  frameWatchdog=setTimeout(()=>{if(generation!==frameGeneration)return;framePending=false;frameTimer=setTimeout(requestFrame,500);},2000);
  camera.src=`/api/pose/frame.jpg?t=${Date.now()}`;
}
async function refresh(){
  clearTimeout(refreshTimer);if(leaving||refreshPending)return;
  refreshPending=true;
  const controller=new AbortController();statusController=controller;
  const timer=setTimeout(()=>controller.abort(),5000);
  try{
    const r=await fetch('/api/posture/status',{cache:'no-store',signal:controller.signal});
    if(!r.ok)throw Error('status unavailable');
    const data=await r.json();if(typeof data.enabled!=='boolean')throw Error('invalid status');
    if(!leaving){render(data);requestStatus.textContent='';}
  }catch{
    if(!leaving){
      // Do not leave a green "healthy" conclusion on screen after its
      // producer is unreachable. Preserve only the last manual enable intent.
      app.dataset.state='unknown';gauge.style.setProperty('--p','0deg');
      progress.textContent='--';gaugeLabel.textContent='状态未确认';
      title.textContent='暂时无法确认姿态';message.textContent='连接恢复后会继续显示新观测，现在不沿用刚才的姿态结论。';
      setReason('—','当前无姿势结论','连接中断不等于坐姿异常，也不等于守护已关闭');
      source.textContent='状态未确认';success.textContent='--';latency.textContent='--';
      cameraPill.querySelector('b').textContent='状态未确认';cameraPill.classList.toggle('offline',true);
      monitorState.textContent='连接中断，守护状态尚未确认';
      reminderStatus.textContent='无法确认提醒播放状态；不会重复提交提醒。';
      requestStatus.textContent='无法确认后台状态；不会当作已关闭，也不会自动重启。';
    }
  }finally{
    clearTimeout(timer);if(statusController===controller)statusController=null;refreshPending=false;
    if(!leaving)refreshTimer=setTimeout(refresh,2000);
  }
}
async function action(path){
  if(actionPending||leaving)return;actionPending=true;
  requestStatus.textContent='正在处理…';if(lastStatus)render(lastStatus);
  try{
    const result=await post(path);if(!leaving){render(result);requestStatus.textContent=result.accepted===false?'未能开始，请等模型就绪后再试。':'';}
  }catch{
    if(!leaving)requestStatus.textContent='操作结果尚未确认，请稍后查看状态；不会重复提交。';
  }finally{actionPending=false;if(!leaving){if(lastStatus)render(lastStatus);refresh();}}
}
monitorToggle.addEventListener('click',()=>action(monitorEnabled===true?'/api/posture/pause':'/api/posture/start'));
simulate.addEventListener('click',()=>action('/api/posture/simulate'));
calibrate.addEventListener('click',()=>action('/api/posture/calibrate'));
reset.addEventListener('click',()=>action('/api/posture/reset'));
postureBack.addEventListener('click',()=>{if(window.StarBuddyMode)window.StarBuddyMode.exit();else{postureBack.disabled=true;location.replace('/');}});
function updateClock(){$('clock').textContent=new Intl.DateTimeFormat('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date());}
function freezePreview(){leaving=true;clearTimeout(refreshTimer);statusController?.abort();stopPreview();}
window.StarBuddyMode?.configure({name:'坐姿守护',stopLabel:'暂停坐姿守护',
  freeze(){freezePreview();},
  async stop(exit){if(!exit)render(await post('/api/posture/pause'));},
  async ready(){return true;},resume(){leaving=false;refresh();},
  stopped:'坐姿守护已暂停，校准保留。退出页面只关闭预览，不暂停已开启的后台守护。'});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPreview();else ensureCamera();});
window.addEventListener('pagehide',freezePreview);
updateClock();setInterval(updateClock,30000);refresh();
