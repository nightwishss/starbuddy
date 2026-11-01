const $=id=>document.getElementById(id);
const stage=$('stage'),primary=$('primary'),secondary=$('secondary'),phase=$('phase');
const buddy=$('buddy'),choices=$('choices'),visionCard=$('vision-card');
const cameraState=$('camera-state'),cameraFeeds=$('camera-feeds'),cameraTabs=$('camera-tabs');
const visionToggle=$('vision-toggle'),observeAction=$('observe-action');
const voiceAction=$('voice-action');
const chatForm=$('chat-form'),chatText=$('chat-text'),chatSend=$('chat-send'),chatReset=$('chat-reset');
const chatCloud=$('chat-cloud'),chatCloudLabel=$('chat-cloud-label');
const turnDetails=$('turn-details'),turnEvidence=$('turn-evidence'),turnTiming=$('turn-timing');
const privacyMode=$('privacy-mode');
const turnStop=$('turn-stop');
const taskConfirm=$('task-confirm'),taskSummary=$('task-summary');
const taskAccept=$('task-accept'),taskCancel=$('task-cancel');
const postureStatus=$('posture-status');
const identityEntry=$('identity-entry');
const objectChips=$('object-chips'),visionDescription=$('vision-description'),visionTime=$('vision-time');
let sequence=-1,dialogueBusy=false,visionEnabled=false,previewStarted=false,cameraMode='dual',activeDraftId=null;
let cloudHistory=[];
let offlineOnly=false;
let currentDiagnosticTurn=null;
function renderCloudMode(){
  if(offlineOnly){
    chatCloud.disabled=true;chatCloud.checked=false;cloudHistory=[];
    chatCloudLabel.textContent='比赛离线模式 · 云端已禁用';
    privacyMode.innerHTML='<span class="trust-dot"></span>比赛离线模式 · 对话使用板端模型';
    return;
  }
  chatCloudLabel.textContent=chatCloud.disabled?'云端未配置，默认板端 Qwen':'允许云端对话/任务理解：当前文字、选定聊天历史或公开能力说明发往 CSIC；不发送感知/画像';
  privacyMode.innerHTML=chatCloud.checked?'<span class="trust-dot"></span>云端会话已开启 · 不发送原始音视频或用户资料':'<span class="trust-dot"></span>默认板端处理 · 云端对话可选';
}
function rememberCloudTurn(data,text){
  // A local fallback can use private perception/profile context. Never send
  // that derived answer back to the cloud as history on the next turn.
  if(!chatCloud.checked||data.intent!=='chat'||data.backend!=='cloud-deepseek')return;
  cloudHistory.push({role:'user',content:text.slice(0,500)},{role:'assistant',content:String(data.answer||'').slice(0,240)});
  cloudHistory=cloudHistory.slice(-6);
}
function renderTurnDiagnostics(data){
  if(!data||!data.evidence)return;
  if(currentDiagnosticTurn&&data.turn_id!==currentDiagnosticTurn)return;
  currentDiagnosticTurn=data.turn_id;
  const labels={conversation:'对话',identity:'身份',people:'人物',objects:'物体',posture:'姿态',affect:'表情线索',tasks:'任务',perception:'视觉'};
  const states={available:'有效',present:'有效观测',verified:'已确认',unconfirmed:'未确认',missing:'缺失/过期',unknown:'未知',stale:'已过期',observed_absent:'明确未观测到',disabled:'已关闭',uncertain:'置信度不足',unavailable:'暂不可用',not_sent_to_cloud:'未发送云端'};
  const evidence=Object.entries(data.evidence).map(([key,item])=>{
    const age=item.observed_at!=null?` ${Math.max(0,Math.round(Date.now()/1000-item.observed_at))}秒前`:'';
    const source=item.provider||item.source;
    const confidence=item.confidence!=null?` ${(item.confidence*100).toFixed(0)}%`:'';
    const count=key==='conversation'&&item.turns!=null?` ${item.turns}轮`:key==='tasks'&&item.count!=null?` ${item.count}项`:'';
    return `${labels[key]||key}：${states[item.state]||item.state}${count}${source?` · ${source}`:''}${age}${confidence}${item.version?` · ${item.version.slice(0,8)} / ${item.mode}`:''}`;
  }).join(' | ');
  const timing=data.timing_ms||{};
  const names={capture:'录音',stt:'转写',context_acquire:'读取状态',route:'路由',context:'组装上下文',response:'回答准备',output_enqueue:'提交语音',output_queue:'等待输出',output_synthesis:'语音合成',output_playback_submit:'提交播放',output_playback_lead_in:'设备前导缓冲',first_audio:'进入播放状态（软件）',total_to_output_queue:'至排队总计',total_from_listen_to_output_queue:'语音至排队总计'};
  turnDetails.querySelector('summary').textContent=`本轮依据与耗时 · ${data.mode==='cloud-text-only'?'云端文字':data.mode==='cloud-proposal'?'云端提案 / 板端验证':'板端处理'} · ${Math.round(timing.total_to_output_queue||0)}ms`;
  const domains={presence:'人物可见性',identity:'身份',location:'位置边界',scene:'当前画面',posture:'坐姿',affect:'表情线索',chat:'普通聊天'};
  const route=data.context_route;
  const routeText=route?`本轮路径：${domains[route.domain]||route.domain}${route.local_context_required?' · 使用本地感知':' · 云端仅文字时不使用感知'} | `:'';
  const decision=data.decision;
  const routes={rule:'规则',local:'本地 Qwen',cloud:'授权云端'};
  const kinds={reply:'对话',clarify:'需澄清',wait:'等待',task_draft:'任务草案',control:'明确控制'};
  const decisionText=decision&&decision.kind?`编排：${routes[decision.route]||decision.route} → ${kinds[decision.kind]||decision.kind} · 模型 ${decision.model_calls||0} 次 · ${decision.reason} | `:'';
  turnEvidence.textContent=decisionText+routeText+evidence;
  turnTiming.textContent=Object.entries(names).filter(([key])=>timing[key]!=null).map(([key,label])=>`${label} ${Math.round(timing[key])}ms`).join(' · ');
}
async function refreshLastTurnDiagnostics(){
  if(dialogueBusy)return;
  try{const response=await fetch('/api/companion/turn/last',{cache:'no-store'});if(response.ok)renderTurnDiagnostics(await response.json());}catch{}
}
const PREVIEW_INTERVAL_MS={single:300,dual:500};
const PREVIEW_RETRY_MS=1000,PREVIEW_REQUEST_TIMEOUT_MS=3000;
const previews=new Map([...document.querySelectorAll('.camera-feed')].map(feed=>[
  feed.dataset.source,{feed,image:feed.querySelector('img'),running:false,generation:0,nextTimer:null,requestTimer:null}
]));
const phaseLabels={idle:'准备好了',listening:'正在听你说',thinking:'正在认真想',speaking:'正在回答',error:'我们再试一次',observing:'正在观察周围',task_confirm:'等待你确认',task_scheduled:'提醒已安排',acting:'正在执行任务',attention:'请注意提醒',success:'任务完成'};
const labelNames={person:'人',cup:'杯子',bottle:'瓶子','cell phone':'手机',laptop:'电脑',keyboard:'键盘',mouse:'鼠标',book:'书',chair:'椅子',tv:'屏幕',dog:'小狗',cat:'小猫',backpack:'背包',handbag:'手提包',clock:'时钟'};

function post(path,body={}){return window.StarBuddyMode ? window.StarBuddyMode.track(()=>requestPost(path,body)) : requestPost(path,body);}
async function requestPost(path,body={}){
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'互动暂时不可用');
  return data;
}

function cameraVisible(source){return cameraMode==='dual'||cameraMode===source;}

function startSourcePreview(source){
  const preview=previews.get(source);
  if(!preview||preview.running||!previewStarted||document.hidden||!cameraVisible(source))return;
  preview.running=true;const generation=++preview.generation;
  const clearTimers=()=>{
    if(preview.nextTimer!==null){clearTimeout(preview.nextTimer);preview.nextTimer=null;}
    if(preview.requestTimer!==null){clearTimeout(preview.requestTimer);preview.requestTimer=null;}
  };
  const schedule=(delay)=>{
    clearTimers();
    if(!previewStarted||!preview.running||document.hidden||generation!==preview.generation)return;
    preview.nextTimer=setTimeout(pump,delay);
  };
  const pump=()=>{
    if(!previewStarted||!preview.running||document.hidden||generation!==preview.generation)return;
    clearTimers();
    preview.image.onload=()=>schedule(PREVIEW_INTERVAL_MS[cameraMode]);
    preview.image.onerror=()=>schedule(PREVIEW_RETRY_MS);
    preview.image.src=`/api/camera/source-frame.jpg?source=${encodeURIComponent(source)}&t=${Date.now()}`;
    preview.requestTimer=setTimeout(()=>{
      preview.image.onload=null;preview.image.onerror=null;preview.image.removeAttribute('src');
      schedule(PREVIEW_RETRY_MS);
    },PREVIEW_REQUEST_TIMEOUT_MS);
  };pump();
}

function stopSourcePreview(source){
  const preview=previews.get(source);if(!preview)return;
  preview.running=false;preview.generation++;preview.image.onload=null;preview.image.onerror=null;
  if(preview.nextTimer!==null){clearTimeout(preview.nextTimer);preview.nextTimer=null;}
  if(preview.requestTimer!==null){clearTimeout(preview.requestTimer);preview.requestTimer=null;}
  preview.image.removeAttribute('src');
}

function applyCameraMode(mode){
  cameraMode=mode;cameraFeeds.className=`camera-feeds ${mode==='dual'?'dual':'single'}`;
  for(const [source,preview] of previews){
    const visible=cameraVisible(source);preview.feed.hidden=!visible;
    if(previewStarted&&visible)startSourcePreview(source);else if(!visible)stopSourcePreview(source);
  }
  cameraTabs.querySelectorAll('button').forEach(button=>button.classList.toggle('active',button.dataset.cameraMode===mode));
}

function startPreview(){
  if(window.StarBuddyMode?.locked)return;
  if(previewStarted)return;previewStarted=true;
  for(const source of previews.keys())if(cameraVisible(source))startSourcePreview(source);
}

function stopPreview(){
  previewStarted=false;for(const source of previews.keys())stopSourcePreview(source);
}

async function startVision(){
  if(visionEnabled)return observeNow();
  visionEnabled=true;
  renderVisionMode();
  startPreview();
  try{await post('/api/companion/vision/start');}
  catch{visionEnabled=false;stopPreview();renderVisionMode();phase.textContent='视觉暂时没有准备好';}
}

async function stopVision(){
  try{await post('/api/companion/vision/stop');}catch{}
  visionEnabled=false;stopPreview();renderVisionMode();
}

async function toggleVision(){return visionEnabled?stopVision():startVision();}

async function observeNow(){
  if(!visionEnabled){await startVision();return;}
  observeAction.classList.add('working');
  try{await post('/api/companion/observe');}
  catch{phase.textContent='我刚刚没有看清';}
  finally{setTimeout(()=>observeAction.classList.remove('working'),900);}
}

async function touchBuddy(){
  if(dialogueBusy)return;
  dialogueBusy=true;choices.hidden=false;
  primary.textContent='怎么啦？';secondary.textContent='选一句想告诉我的话';phase.textContent='碰到我啦';
  try{await post('/api/companion/touch');}catch{phase.textContent='互动暂时不可用';}
  finally{dialogueBusy=false;}
}

async function voiceTurn(){
  if(dialogueBusy)return;
  currentDiagnosticTurn=null;
  dialogueBusy=true;choices.hidden=true;voiceAction.disabled=true;
  primary.textContent='我在听';secondary.textContent='请直接说话，说完后自然停顿一下';phase.textContent='正在听你说';
  try{
    const useCloud=chatCloud.checked;
    const data=await post('/api/companion/voice-turn',{cloud:useCloud,...(useCloud?{cloud_history:cloudHistory}:{})});
    if(!data.accepted)phase.textContent='没有听清，请再试一次';
    else{
      handleConversationResponse(data);rememberCloudTurn(data,data.transcript||'');
      renderTurnDiagnostics(data.diagnostics);setTimeout(refreshLastTurnDiagnostics,1200);setTimeout(refreshLastTurnDiagnostics,3500);setTimeout(refreshLastTurnDiagnostics,8000);
      if(data.intent!=='task')secondary.textContent=`识别为：${data.transcript||'（空）'} · ${data.backend==='cloud-deepseek'?'云端回答':'板端回答'}`;
      phase.textContent=data.intent==='task'?'请确认提醒':`听懂并开始回答 · ${Math.round((data.timing?.total_ms||0)/100)/10} 秒`;
    }
  }catch(error){phase.textContent=error.message||'语音互动暂时不可用';}
  finally{dialogueBusy=false;voiceAction.disabled=false;}
}

function handleConversationResponse(data){
  if(data.intent==='task'&&data.task_draft){
    activeDraftId=data.task_draft.draft_id;taskSummary.textContent=data.task_draft.execution_preview||data.task_draft.summary;
    taskConfirm.hidden=false;primary.textContent='请确认这个任务';secondary.textContent=data.task_draft.summary;
  }else if(data.intent!=='task'){
    taskConfirm.hidden=true;activeDraftId=null;
  }
}

async function sendTextConversation(event){
  event.preventDefault();
  const text=chatText.value.trim();
  if(!text||dialogueBusy)return;
  currentDiagnosticTurn=null;
  const useCloud=chatCloud.checked;
  dialogueBusy=true;chatSend.disabled=true;chatReset.disabled=true;chatText.disabled=true;
  primary.textContent='我在认真想';secondary.textContent=text;phase.textContent=useCloud?'云端模型正在回答':'本地 Qwen 正在回答';
  try{
    const data=await post('/api/companion/chat',{text,cloud:useCloud,...(useCloud?{cloud_history:cloudHistory}:{})});
    handleConversationResponse(data);
    if(data.intent==='task'){primary.textContent='请确认这个任务';secondary.textContent=data.task_draft.summary;phase.textContent='等待你确认';}
    else{primary.textContent=data.answer;secondary.textContent=data.backend==='cloud-deepseek'?'云端会话 · 最近三轮文字上下文':'你可以继续输入，星伴记得刚才几轮对话';phase.textContent=data.intent==='ambiguous'?'请补充任务信息':data.backend==='local-qwen-fallback'?'云端不可用，已改用板端回答':'正在从显示器扬声器回答';}
    rememberCloudTurn(data,text);renderTurnDiagnostics(data.diagnostics);
    setTimeout(refreshLastTurnDiagnostics,1200);setTimeout(refreshLastTurnDiagnostics,3500);setTimeout(refreshLastTurnDiagnostics,8000);
    chatText.value='';
  }catch(error){phase.textContent=error.message||'文字对话暂时不可用';}
  finally{dialogueBusy=false;chatSend.disabled=false;chatReset.disabled=false;chatText.disabled=false;chatText.focus();}
}

async function refreshCloudStatus(){
  try{
    const response=await fetch('/api/companion/cloud-status');
    const status=await response.json();
    offlineOnly=status.offline_only===true;
    chatCloud.disabled=!response.ok||!status.configured;
    if(chatCloud.disabled){chatCloud.checked=false;cloudHistory=[];}
    renderCloudMode();
  }catch{chatCloud.disabled=true;chatCloud.checked=false;cloudHistory=[];renderCloudMode();}
}

async function resolveTaskDraft(confirm){
  if(!activeDraftId||dialogueBusy)return;
  dialogueBusy=true;taskAccept.disabled=true;taskCancel.disabled=true;
  try{
    const data=await post(confirm?'/api/companion/task/confirm':'/api/companion/task/cancel',{draft_id:activeDraftId});
    activeDraftId=null;taskConfirm.hidden=true;
    if(confirm){primary.textContent=data.reconciled?'已核对任务实际状态':'任务已经安排好';secondary.textContent=data.task.summary;phase.textContent=data.reconciled?data.task.state:'星伴会按这个安排执行';}
    else if(data.task_created){primary.textContent=data.answer;secondary.textContent=data.task.state;phase.textContent=data.cancelled?'已取消任务':'请核对任务状态';}
    else{primary.textContent='好的，不创建这个任务';secondary.textContent='你可以换一种说法再告诉我';phase.textContent='已取消';}
  }catch(error){phase.textContent=error.message||'暂时无法处理这个提醒';}
  finally{dialogueBusy=false;taskAccept.disabled=false;taskCancel.disabled=false;}
}

async function resetTextConversation(){
  if(dialogueBusy)return;
  chatReset.disabled=true;
  try{
    await post('/api/companion/conversation/reset');
    cloudHistory=[];renderCloudMode();
    currentDiagnosticTurn=null;
    turnDetails.querySelector('summary').textContent='本轮依据与耗时：等待第一次对话';turnEvidence.textContent='尚无本轮状态';turnTiming.textContent='';
    taskConfirm.hidden=true;activeDraftId=null;
    primary.textContent='新的对话开始啦';secondary.textContent='刚才的临时对话已经清空';phase.textContent='短期上下文已清空';chatText.value='';
  }catch(error){phase.textContent=error.message||'暂时无法清空对话';}
  finally{chatReset.disabled=false;chatText.focus();}
}

async function stopCurrentTurn(){
  try{
    const result=await post('/api/companion/turn/cancel');
    if(result.cancelled){primary.textContent='好的，先停一下';secondary.textContent='想继续时再告诉我';phase.textContent='已停止';}
  }catch(error){phase.textContent=error.message||'暂时无法停止';}
}

async function choose(event){
  const choice=event.currentTarget.dataset.choice;
  if(dialogueBusy||!choice)return;
  dialogueBusy=true;
  choices.querySelectorAll('button').forEach(button=>button.disabled=true);
  try{await post('/api/companion/reply',{choice});choices.hidden=true;}
  catch{phase.textContent='本地模型暂时没有回答';}
  finally{dialogueBusy=false;choices.querySelectorAll('button').forEach(button=>button.disabled=false);}
}

function renderVisionMode(){
  visionCard.classList.toggle('offline',!visionEnabled);
  visionToggle.classList.toggle('active',visionEnabled);
  visionToggle.querySelector('b').textContent=visionEnabled?'暂停视觉':'开启视觉';
  visionToggle.querySelector('small').textContent=visionEnabled?'静默更新上下文':'静默感知';
  cameraState.innerHTML=visionEnabled?'<i></i>观察中':'<i></i>未开启';
  if(!visionEnabled){objectChips.innerHTML='<span class="empty-chip">尚无识别结果</span>';visionTime.textContent='等待观察';}
}

function renderVisionStatus(data){
  visionEnabled=Boolean(data.enabled);
  if(visionEnabled&&!previewStarted)startPreview();
  if(!visionEnabled&&previewStarted)stopPreview();
  renderVisionMode();
  visionCard.classList.toggle('analyzing',Boolean(data.busy));
  if(data.description)visionDescription.textContent=data.description;
  const counts=data.counts||{};
  const entries=Object.entries(counts);
  if(entries.length){
    const chips=entries.slice(0,8).map(([label,count])=>{
      const chip=document.createElement('span'),badge=document.createElement('b');
      chip.append(document.createTextNode(labelNames[label]||label));
      badge.textContent=String(count);chip.append(badge);return chip;
    });
    objectChips.replaceChildren(...chips);
  }
  if(data.observed_at){
    const age=Math.max(0,Math.round(Date.now()/1000-data.observed_at));
    visionTime.textContent=age<3?'刚刚':`${age} 秒前`;
  }
  if(data.error)visionTime.textContent='识别暂不可用';
}

async function refreshState(){
  if(window.StarBuddyMode?.locked){setTimeout(refreshState,1000);return;}
  try{
    const response=await fetch('/api/state',{cache:'no-store'}),data=await response.json();
    if(response.ok&&data.sequence!==sequence){
      sequence=data.sequence;stage.className=`stage ${data.phase} ${data.visual_style}`;
      primary.textContent=data.primary_text;secondary.textContent=data.secondary_text;
      phase.textContent=phaseLabels[data.phase]||data.phase;
    }
  }catch{phase.textContent='正在连接星伴';}
  setTimeout(refreshState,500);
}

async function refreshVision(){
  if(window.StarBuddyMode?.locked){setTimeout(refreshVision,1000);return;}
  try{const response=await fetch('/api/companion/vision-status',{cache:'no-store'});if(response.ok)renderVisionStatus(await response.json());}catch{}
  setTimeout(refreshVision,1000);
}

function renderPosture(data){
  const state=data.state||'unknown';
  const labels={healthy:'坐姿状态良好',suspect:'正在确认坐姿变化',unhealthy:'请稍微调整坐姿',calibrating:'正在校准坐姿',observing:'坐姿守护待校准',paused:'坐姿守护已暂停',unknown:'暂时没有看清',model_loading:'姿态模型连接中',model_unavailable:'坐姿守护未就绪'};
  postureStatus.className=`posture-status ${state}`;
  postureStatus.querySelector('span').textContent=labels[state]||'坐姿守护运行中';
  postureStatus.querySelector('small').textContent=data.calibrated?'后台运行 · 点此查看':'点此完成一次校准';
}

async function refreshPosture(){
  try{const response=await fetch('/api/posture/status',{cache:'no-store'});if(response.ok)renderPosture(await response.json());}
  catch{postureStatus.querySelector('span').textContent='坐姿守护暂不可用';}
  setTimeout(refreshPosture,2000);
}

async function syncIdentity(){
  try{
    const response=await fetch('/api/companion/identity',{cache:'no-store'}),data=await response.json();
    if(response.ok&&data.status==='known'){
      identityEntry.querySelector('span').textContent=data.label;
      identityEntry.classList.add('known');
    }
  }catch{}
  refreshIdentityState();
}
async function refreshIdentityState(){
  try{
    const response=await fetch('/api/companion/identity-current',{cache:'no-store'}),data=await response.json();
    if(response.ok&&data.status==='known'){
      identityEntry.querySelector('span').textContent=data.label;identityEntry.classList.add('known');
    }else{
      identityEntry.querySelector('span').textContent='身份未确认';identityEntry.classList.remove('known');
    }
  }catch{identityEntry.querySelector('span').textContent='身份暂不可用';identityEntry.classList.remove('known');}
  setTimeout(refreshIdentityState,3000);
}

function updateClock(){
  $('clock').textContent=new Intl.DateTimeFormat('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date());
}

buddy.addEventListener('click',touchBuddy);
$('voice-action').addEventListener('click',voiceTurn);
$('touch-action').addEventListener('click',touchBuddy);
$('camera-surface').addEventListener('click',()=>visionEnabled?observeNow():startVision());
observeAction.addEventListener('click',observeNow);
visionToggle.addEventListener('click',toggleVision);
cameraTabs.querySelectorAll('button').forEach(button=>button.addEventListener('click',event=>{
  event.stopPropagation();applyCameraMode(button.dataset.cameraMode);
}));
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){for(const source of previews.keys())stopSourcePreview(source);}
  else if(previewStarted){for(const source of previews.keys())if(cameraVisible(source))startSourcePreview(source);}
});
choices.querySelectorAll('button').forEach(button=>button.addEventListener('click',choose));
chatForm.addEventListener('submit',sendTextConversation);
chatCloud.addEventListener('change',()=>{cloudHistory=[];renderCloudMode();});
chatReset.addEventListener('click',resetTextConversation);
turnStop.addEventListener('click',()=>window.StarBuddyMode ? window.StarBuddyMode.stop() : stopCurrentTurn());
taskAccept.addEventListener('click',()=>resolveTaskDraft(true));
taskCancel.addEventListener('click',()=>resolveTaskDraft(false));
updateClock();setInterval(updateClock,30000);applyCameraMode('dual');renderVisionMode();syncIdentity();refreshState();refreshVision();refreshPosture();refreshCloudStatus();
window.StarBuddyMode?.configure({name:'聊天陪伴',
  freeze(){dialogueBusy=true;stopPreview();},
  async stop(){await post('/api/companion/turn/cancel');await post('/api/companion/vision/stop');visionEnabled=false;stopPreview();renderVisionMode();},
  ready:window.StarBuddyMode.ready,
  resume(){dialogueBusy=false;chatSend.disabled=chatReset.disabled=chatText.disabled=false;},
  stopped:'当前对话、收音和主动视觉已停止；后台坐姿守护及已确认的任务保留。'});
