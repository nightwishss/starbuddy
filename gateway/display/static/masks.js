'use strict';
const $ = id => document.getElementById(id);
// Per-page bearer, not a profile or a persistent identifier.
const owner = Array.from(crypto.getRandomValues(new Uint8Array(32)), b => b.toString(16).padStart(2, '0')).join('');
const catalogue = [
  ['happy','开心'],['calm','平静'],['surprised','惊讶'],
  ['sad','难过'],['angry','生气'],['worried','担心']
];
let current = null, busy = false, generation = 0, intensity = 1, composing = false, left = false;
let cameraEnabled=false,cameraBusy=false,cameraGeneration=0,cameraTimer=null,frameTimer=null;
let cameraController=null,frameController=null,frameUrl=null,pendingFrameUrl=null,frameSequence=0;
let pendingFrameImage=null,frameFreshTimer=null,lastFrameAt=0,latestCamera=null;
const choiceButtons = [], levelButtons = [...document.querySelectorAll('[data-level]')];
function shape(svg, name, attrs) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  svg.append(node); return node;
}
const emotionHints = {
  happy:'眉毛舒展，嘴角上扬；明显时，眼睛也笑弯了。',
  calm:'眉毛放松，嘴巴轻轻合着；眼睛慢慢放松下来。',
  surprised:'眼睛睁开，嘴巴变成小小的圆圈。',
  sad:'眉毛内侧抬起，嘴角向下，像有一点失落。',
  angry:'眉毛往中间压低，嘴巴抿紧，像有一点不高兴。',
  worried:'眉毛不太对称，嘴巴弯弯曲曲，像有一点拿不准。'
};
// Teaching cues describe the authored buddy, not measured facial landmarks.
const faceCues={
  happy:['舒展开','弯弯的','向上扬'],calm:['放松','自然放松','轻轻合上'],
  surprised:['往上抬','睁得圆圆','张成小圆圈'],sad:['内侧抬起','轻轻垂下','往下弯'],
  angry:['往中间靠','微微眯起','抿得紧紧'],worried:['轻轻皱起','有点睁大','有一点紧']
};
// Same public FER categories as the backend. This is an appearance estimate,
// never a diagnosis or a claim about what the child actually feels.
const expressionNames={neutral:['平静 / 中性','calm'],happy_like:['笑脸 / 开心样','happy'],
  surprised_like:['惊讶样','surprised'],sad_like:['难过样','sad'],
  angry_like:['皱眉 / 生气样','angry'],fear_like:['担心样','worried'],
  disgust_like:['厌恶样',null],contempt_like:['轻蔑样',null]};
let previewKind = 'happy';
const previewButtons = [];
function clearMotion() {
  $('mask-touch').classList.remove('buddy-touch','buddy-nod','buddy-celebrate');
}
function react(kind='touch') {
  if (left || document.hidden) return;
  clearMotion();
  const matched=kind==='matched';
  $('buddy-reaction').textContent=matched?'我们想到一起啦！':kind==='shared'?'谢谢你告诉我，我们一起看看。':kind==='change'?'换个表情，慢慢看。':'我在呢，轻轻碰一下就会回应。';
  $('buddy-reaction').dataset.reaction=kind;
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  // A finite gesture only. No idle loop, audio synthesis or network request.
  void $('mask').getBoundingClientRect();
  $('mask-touch').classList.add(matched?'buddy-celebrate':kind==='touch'?'buddy-touch':'buddy-nod');
}
function drawMask(svg, kind, strength) {
  kind = catalogue.some(c => c[0] === kind) ? kind : 'happy';
  const n = [1,2,3].includes(strength) ? strength : 1;
  // One approved 3D-rendered character, 18 bounded image views.
  // Same image atlas is reused; no canvas, model or camera loop.
  svg.replaceChildren();
  svg.setAttribute('data-emotion', kind);
  svg.setAttribute('data-intensity', String(n));
  shape(svg, 'image', {
    href:'/home-masks.svg#'+kind+'-'+n,
    x:20, y:0, width:280, height:280,
    preserveAspectRatio:'xMidYMid meet'
  });
}
function expressionCaption(kind) {
  $('expression-detail').textContent = emotionHints[kind] || emotionHints.happy;
  const cues=faceCues[kind]||faceCues.happy;
  ['brows','eyes','mouth'].forEach((part,index)=>$('cue-'+part).textContent=cues[index]);
}
function preview(kind) {
  if (busy || current && ['inviting','feedback'].includes(current.phase)) return;
  previewKind = kind;
  drawMask($('mask'), kind, intensity);
  $('mask').setAttribute('aria-label', '芽芽伙伴：'+emotionHints[kind]);
  $('mask-caption').textContent = catalogue.find(item=>item[0]===kind)[1]+' · 看看眉毛、眼睛和嘴角';
  expressionCaption(kind);
  for (const button of previewButtons) button.setAttribute('aria-pressed',String(button.dataset.emotion===kind));
  react('change');
}
function status(text) { $('status').textContent = text; }
function ids() { return {session_id:current ? current.id : null, revision:current ? current.revision : 0, speak:$('speak').checked}; }
function controls() {
  const active = current && ['inviting','feedback'].includes(current.phase), ready = active && current.phase==='inviting' && !busy;
  for (const button of choiceButtons) button.disabled = !ready;
  for (const id of ['imitated','unsure','send','voice','description']) $(id).disabled = !ready;
  for (const id of ['next','repeat','finish']) $(id).disabled = busy || !active;
  $('start').disabled = busy;
  $('mask-touch').disabled = busy;
  for (const button of previewButtons) button.disabled = busy;
  for (const button of levelButtons) { button.disabled = busy; button.setAttribute('aria-pressed',String(Number(button.dataset.level)===intensity)); }
  $('imitation-toggle').disabled=cameraBusy;
  $('imitation-toggle').textContent=cameraEnabled?'关闭我的画面':'开启我的画面';
  $('imitation-toggle').setAttribute('aria-pressed',String(cameraEnabled));
  $('compare').disabled=busy||!cameraEnabled||!active;
}
function clearFrame() {
  clearTimeout(frameFreshTimer);frameFreshTimer=null;lastFrameAt=0;
  $('imitation-frame').hidden=true;$('imitation-frame').removeAttribute('src');
  if(frameUrl)URL.revokeObjectURL(frameUrl);frameUrl=null;
  if(pendingFrameUrl)URL.revokeObjectURL(pendingFrameUrl);pendingFrameUrl=null;
  pendingFrameImage?.removeAttribute('src');pendingFrameImage=null;
}
function loadFrame(image,url,signal) {
  return new Promise((resolve,reject)=>{
    let settled=false;
    const finish=error=>{
      if(settled)return;settled=true;
      image.removeEventListener('load',loaded);image.removeEventListener('error',failed);
      signal.removeEventListener('abort',aborted);
      error?reject(error):resolve();
    };
    const decoded=()=>finish(image.complete&&image.naturalWidth>0?null:Error('frame decode failed'));
    const loaded=()=>{if(typeof image.decode!=='function')decoded();};
    const failed=()=>finish(Error('frame decode failed'));
    const aborted=()=>finish(Error('frame load aborted'));
    image.addEventListener('load',loaded);image.addEventListener('error',failed);
    signal.addEventListener('abort',aborted,{once:true});
    if(signal.aborted){aborted();return;}
    image.src=url;
    // The real image must load/decode before a frame is labelled live. This
    // catches CSP, invalid JPEGs and decode errors that HTTP 200 cannot detect.
    if(typeof image.decode==='function')image.decode().then(decoded,failed);
  });
}
function renderCamera(value) {
  latestCamera=value;
  const estimate=value?.estimate;
  const category=expressionNames[estimate?.label],confidence=estimate?.confidence;
  const valid=estimate?.state==='estimated'&&category&&Number.isFinite(confidence)&&confidence>=.6&&confidence<=1;
  $('imitation-estimate').closest('.recognition').dataset.state=valid?'estimated':cameraEnabled?'unknown':'off';
  $('imitation-confidence').textContent=valid?'模型置信度 '+Math.round(confidence*100)+'% · 不是心情分数':'';
  if(valid){
    $('imitation-estimate').textContent=category[0];
    const target=current&&['inviting','feedback'].includes(current.phase)?current.mask?.id:null;
    $('imitation-match').textContent=target&&category[1]?(target===category[1]?'和芽芽有相似线索':'和芽芽不同')+' · 不代表真实心情':'面部外观估计 · 不代表真实心情';
  }else{
    $('imitation-estimate').textContent=cameraEnabled?'暂时没看清':'等待开启画面';
    $('imitation-match').textContent=!cameraEnabled?'画面默认关闭，想试时再开启。':estimate?.reason==='no_single_face'?'请让一张脸进入画面，暂时不作判断。':estimate?.reason==='low_confidence'?'这次不太确定，暂时不作判断。':'等待新鲜、稳定的表情线索，不猜心情。';
  }
}
async function refreshCamera(token) {
  if(!cameraEnabled||left||document.hidden||token!==cameraGeneration)return;
  cameraController=new AbortController();const activeController=cameraController;
  const deadline=setTimeout(()=>activeController.abort(),4000);
  try {
    const response=await fetch('/api/masks/camera-status',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({owner}),signal:activeController.signal});
    if(!response.ok)throw Error();const value=await response.json();
    if(token!==cameraGeneration||!cameraEnabled)return;
    if(value.camera?.enabled!==true){stopCameraLocal();$('imitation-state').textContent='画面已结束，需要时可重新开启。';return;}
    renderCamera(value.camera);
  } catch(_) {if(token===cameraGeneration&&cameraEnabled){renderCamera({estimate:{state:'unknown'}});$('imitation-match').textContent='识别暂未连接，暂时不作判断。';}}
  finally {
    clearTimeout(deadline);if(cameraController===activeController)cameraController=null;
    if(token===cameraGeneration&&cameraEnabled&&!left&&!document.hidden)cameraTimer=setTimeout(()=>refreshCamera(token),1000);
  }
}
async function refreshFrame(token) {
  if(!cameraEnabled||left||document.hidden||token!==cameraGeneration)return;
  frameController=new AbortController();const activeController=frameController;
  const deadline=setTimeout(()=>activeController.abort(),5000);
  try {
    const response=await fetch('/api/masks/camera-frame.jpg?owner='+encodeURIComponent(owner)+'&after='+frameSequence,{cache:'no-store',signal:activeController.signal});
    if(!response.ok)throw Error();const blob=await response.blob();
    if(token!==cameraGeneration||!cameraEnabled)return;
    const next=URL.createObjectURL(blob);pendingFrameUrl=next;
    // Decode a detached candidate. The displayed image is never pointed at an
    // undecoded URL; a good previous frame stays painted until atomic commit.
    const image=new Image();pendingFrameImage=image;
    image.alt='我的本地实时画面';image.decoding='async';
    await loadFrame(image,next,activeController.signal);
    if(token!==cameraGeneration||!cameraEnabled||left||document.hidden)return;
    frameSequence=Number(response.headers.get('X-VQ-Sequence'))||frameSequence;
    image.id='imitation-frame';image.hidden=false;
    $('imitation-frame').replaceWith(image);
    const previous=frameUrl;frameUrl=next;pendingFrameUrl=null;
    pendingFrameImage=null;lastFrameAt=Date.now();
    if(previous)URL.revokeObjectURL(previous);
    $('imitation-placeholder').hidden=true;
    $('imitation-state').textContent='我的实时画面 · 前置镜头';
    clearTimeout(frameFreshTimer);
    frameFreshTimer=setTimeout(()=>{
      if(token===cameraGeneration&&cameraEnabled&&!left&&frameUrl&&Date.now()-lastFrameAt>=3000)
        $('imitation-state').textContent='画面暂未更新 · 显示上一帧';
    },3100);
  }catch(_){if(token===cameraGeneration&&cameraEnabled){
    if(frameUrl)$('imitation-state').textContent='连接波动 · 显示上一帧，等待更新';
    else{$('imitation-placeholder').hidden=false;$('imitation-placeholder').textContent='画面暂未显示';$('imitation-state').textContent='画面未能显示，正在等待下一帧。';}
  }}
  finally {
    if(pendingFrameUrl&&frameController===activeController){URL.revokeObjectURL(pendingFrameUrl);pendingFrameUrl=null;pendingFrameImage?.removeAttribute('src');pendingFrameImage=null;}
    clearTimeout(deadline);if(frameController===activeController)frameController=null;
    if(token===cameraGeneration&&cameraEnabled&&!left&&!document.hidden)frameTimer=setTimeout(()=>refreshFrame(token),120);
  }
}
function stopCameraLocal() {
  cameraEnabled=false;++cameraGeneration;clearTimeout(cameraTimer);clearTimeout(frameTimer);
  cameraController?.abort();frameController?.abort();cameraController=frameController=null;clearFrame();
  $('imitation-placeholder').hidden=false;$('imitation-placeholder').textContent='点下方开启自己的画面';
  $('imitation-state').textContent='本页画面未开启';renderCamera(null);controls();
}
async function toggleCamera() {
  if(cameraBusy)return;
  if(cameraEnabled){stopCameraLocal();try{await api('camera-stop');}catch(_){$('imitation-state').textContent='已停止本页画面；释放回执未确认。';}return;}
  cameraBusy=true;controls();$('imitation-state').textContent='正在开启本地画面…';
  const token=++cameraGeneration;
  try {
    const result=await api('camera-start');
    if(token!==cameraGeneration||left||document.hidden)return;
    if(result.camera?.enabled!==true)throw Error();
    cameraEnabled=true;frameSequence=0;renderCamera(result.camera);refreshCamera(token);refreshFrame(token);
  }catch(_){if(token===cameraGeneration)$('imitation-state').textContent='画面未能开启，可以再试一次。';}
  finally{cameraBusy=false;controls();}
}
function render(game) {
  clearMotion();
  $('buddy-reaction').textContent='轻碰芽芽，我会动一动。';
  delete $('buddy-reaction').dataset.reaction;
  current = game;
  if (game) intensity = game.intensity;
  const active = game && ['inviting','feedback'].includes(game.phase);
  document.querySelector('.play').dataset.phase=active?game.phase:'preview';
  $('intro').hidden = Boolean(active); $('game').hidden = !active;
  $('choices').hidden=game?.phase==='feedback';
  $('question').textContent=game?.phase==='feedback'?'这张叫“'+game.mask.label+'”，听听芽芽怎么说':'给芽芽的表情选个名字';
  $('start').textContent = game ? '再玩一次' : '开始认表情';
  $('round').textContent = game ? (active ? `${game.round} / ${game.total_rounds}` : game.phase==='finished' ? '这一轮完成啦' : '先休息一下') : '六张小面具';
  const mask = game && game.mask;
  const kind = active && mask ? mask.id : previewKind;
  drawMask($('mask'), kind, intensity);
  $('mask').setAttribute('aria-label','芽芽伙伴：'+(emotionHints[kind] || emotionHints.happy));
  expressionCaption(kind);
  $('mask-caption').textContent = active ? (game.phase==='feedback' ? `${mask.label} · 你也可以有不同理解` : '看看眉毛、眼睛和嘴角。') : catalogue.find(item=>item[0]===kind)[1]+' · 先看一看';
  $('answer').textContent = game ? game.answer : '';
  $('next').textContent = game && game.round===6 ? '完成这一轮 ✓' : game && game.phase==='feedback' ? '下一张 →' : '跳过 / 下一张 →';
  renderCamera(latestCamera);
  controls();
  // Keep the model's feedback readable even on the short landscape screen.
  // Scroll only this pane; the camera, compare and Stop controls stay fixed.
  const pane=document.querySelector('.game-scroll');
  if(pane)pane.scrollTop=game?.phase==='feedback'?pane.scrollHeight:0;
}
function api(action,data={}){return window.StarBuddyMode ? window.StarBuddyMode.track(()=>request(action,data)) : request(action,data);}
async function request(action, data={}) {
  const controller = new AbortController(), timeout = setTimeout(()=>controller.abort(),action==='voice'?60000:15000);
  try {
    const response = await fetch('/api/masks/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({owner,...data}),signal:controller.signal});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || '这次没有连接上');
    return result;
  } finally {clearTimeout(timeout);}
}
async function run(action, data={}) {
  if (busy || left) return;
  const mine = ++generation; busy=true; controls();
  status(action==='voice'?'我在听，说完后自然停顿。可随时停止收音。':action==='compare'?'芽芽正在看看这次的模仿，准备回应…':'芽芽正在想怎么回应…');
  try {
    const result = await api(action,{...ids(),...data});
    if (mine!==generation || left) return;
    if (result.game) render(result.game);
    if (['respond','describe','voice','compare'].includes(action) && result.game) {
      // Acknowledge participation equally: matching a facial prototype is not
      // a better/normal expression, and an uncertain sensor is not a mistake.
      react('shared');
    } else if (result.game?.phase==='inviting' && ['start','next','intensity'].includes(action)) {
      react('change');
    }
    if (action==='describe'||action==='voice') $('description').value='';
    status(result.audio_error || (result.accepted===false ? '没有听清，可以点选或再试一次。' : result.playback_id ? '已交给声音服务；可随时停止。下一步由你决定。' : '不着急。可以跳过，也可以结束。'));
  } catch(error) {
    if(mine!==generation || left)return;
    // No automatic replay on uncertain delivery.
    try {const result=await api('silence');if(mine===generation)render(result.game);}catch(_){}
    if(mine===generation)status(error.name==='AbortError'?'等待超时，已尝试停止声音；请检查连接后再试。':error.message);
  } finally {if(mine===generation){busy=false;controls();}}
}
async function stop(silence=false) {
  clearMotion();
  if(!silence)stopCameraLocal();
  const mine=++generation;busy=true;controls();
  try {const result=await api(silence?'silence':'stop');if(mine===generation){render(result.game);status(silence?'已请求停止声音 / 收音。可以继续点选。':'游戏已结束，没有保存回答。');}}
  catch(_){if(mine===generation)status('停止请求未确认，请检查连接。');}
  finally{if(mine===generation){busy=false;controls();}}
}
async function observeOnly() {
  if(busy||left)return;
  const mine=++generation;busy=true;stopCameraLocal();controls();
  let released=true;
  try{await api('camera-stop');}catch(_){released=false;}
  if(mine!==generation||left)return;
  busy=false;controls();
  await run('respond',{choice:'unsure'});
  if(!released&&!left)status('本页画面已停止，释放回执未确认；可以只看卡片，不用模仿。');
}
function leave() {
  clearMotion();
  stopCameraLocal();
  left=true;++generation;busy=false;controls();
  navigator.sendBeacon('/api/masks/camera-stop',new Blob([JSON.stringify({owner})],{type:'application/json'}));
  navigator.sendBeacon('/api/masks/stop',new Blob([JSON.stringify({owner})],{type:'application/json'}));
}
for (const [id,label] of catalogue) {
  const button=document.createElement('button');button.type='button';button.className='choice';
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 320 280');svg.setAttribute('aria-hidden','true');drawMask(svg,id,3);
  const name=document.createElement('span');name.textContent=label;button.append(svg,name);
  button.addEventListener('click',()=>run('respond',{choice:id}));choiceButtons.push(button);$('choices').append(button);
  const previewButton=document.createElement('button');previewButton.type='button';previewButton.className='choice';
  previewButton.dataset.emotion=id;previewButton.setAttribute('aria-pressed',String(id===previewKind));
  previewButton.setAttribute('aria-label','看看芽芽的'+label+'表情');
  previewButton.append(svg.cloneNode(true),name.cloneNode(true));
  previewButton.addEventListener('click',()=>preview(id));previewButtons.push(previewButton);$('emotion-preview').append(previewButton);
}
for(const button of levelButtons)button.addEventListener('click',()=>{
  if(busy)return;
  const level=Number(button.dataset.level);
  if(current&&['inviting','feedback'].includes(current.phase))run('intensity',{intensity:level});
  else{intensity=level;preview(previewKind);controls();}
});
$('mask-touch').addEventListener('click',()=>{if(!busy)react('touch');});
$('mask').addEventListener('animationend',clearMotion);
$('start').addEventListener('click',()=>run('start',{intensity}));
$('imitation-toggle').addEventListener('click',toggleCamera);
$('compare').addEventListener('click',()=>{if(!$('compare').disabled)run('compare');});
for(const action of ['next','repeat'])$(action).addEventListener('click',()=>run(action));
$('imitated').addEventListener('click',()=>run('respond',{choice:'imitated'}));
$('unsure').addEventListener('click',observeOnly);
$('voice').addEventListener('click',()=>run('voice'));
$('description').addEventListener('compositionstart',()=>{composing=true;});
$('description').addEventListener('compositionend',()=>{composing=false;});
$('describe-form').addEventListener('submit',event=>{event.preventDefault();if(!composing&&current&&current.phase==='inviting'&&$('description').value.trim())run('describe',{text:$('description').value.trim()});});
$('silence').addEventListener('click',()=>window.StarBuddyMode ? window.StarBuddyMode.stop() : stop(true));$('finish').addEventListener('click',()=>stop());
$('speak').addEventListener('change',()=>{if(!$('speak').checked)stop(true);});
$('back').addEventListener('click',async event=>{event.preventDefault();leave();location.assign('/');});
window.addEventListener('pagehide',leave);
document.addEventListener('visibilitychange',()=>{if(document.hidden)leave();});
window.addEventListener('pageshow',()=>{if(left){left=false;api('status').then(result=>render(result.game)).catch(()=>status('连接未确认，请刷新。'));}});
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&left){left=false;api('status').then(result=>render(result.game)).catch(()=>status('连接未确认，请刷新。'));}});
render(null); // No timers, polling, model, microphone or playback on page load.
window.StarBuddyMode?.configure({name:'情绪面具',
  freeze(){clearMotion();stopCameraLocal();left=true;++generation;busy=true;controls();},
  async stop(exit){await api('camera-stop');const result=await api(exit?'stop':'silence');render(result.game);},
  ready:window.StarBuddyMode.ready,
  resume(){left=false;busy=false;controls();status('声音 / 收音已结束，可以继续点选。');}});
