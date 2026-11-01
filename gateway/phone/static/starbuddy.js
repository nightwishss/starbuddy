const face=document.getElementById('face');
const mood=document.getElementById('mood');
const state=document.getElementById('state');
const form=document.getElementById('talk');
const input=document.getElementById('message');
const send=document.getElementById('send');
const mic=document.getElementById('mic');
const boardMic=document.getElementById('board-mic');
const observe=document.getElementById('observe');
const monitor=document.getElementById('monitor');
const speaker=document.getElementById('speaker');
const replay=document.getElementById('replay');
const connection=document.getElementById('connection');
let busy=false;
let recorder=null;
let audioChunks=[];
let outputMode='browser';
let currentAudio=null;
let currentPlayback=null;
let monitoring=false;

function expression(name){face.className=`face ${name||'idle'}`;}
function inferMood(text){if(/难过|伤心|害怕|不开心|哭/.test(text))return'gentle';if(/抱抱|喜欢|爱/.test(text))return'love';if(/错误|失败|不能|无法/.test(text))return'worried';return'happy';}
async function readJson(response){let data={};try{data=await response.json();}catch{}if(!response.ok)throw new Error(data.error||`请求失败（${response.status}）`);return data;}
async function emitEvent(event,affect){try{return await readJson(await fetch('/api/interaction/event',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({event,...(affect?{affect}:{})})}));}catch(error){console.warn('interaction event rejected',event,error.message);return null;}}
function recordingMime(){for(const type of ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus']){if(MediaRecorder.isTypeSupported(type))return type;}return '';}
function updateOutputLabel(){const labels={browser:'声音：手机',device:'声音：显示器',off:'声音：关'};speaker.textContent=labels[outputMode];speaker.setAttribute('aria-pressed',String(outputMode==='off'));}
async function playBoardAudio(media){
  if(!media){emitEvent('speech_finished');return;}
  if(outputMode==='off'){emitEvent('speech_finished');return;}
  if(outputMode==='device'){
    replay.hidden=true;
    currentAudio=null;
    try{
      const result=await readJson(await fetch('/api/device/audio/play',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_id:media.media_id,volume_percent:70})}));
      currentPlayback=result.playback_id;
      expression('speaking');
      state.textContent='星伴正在通过显示器说话…';
      watchPhysicalPlayback(currentPlayback);
      return;
    }catch(error){
      // A lost response is not proof that the board did not start playing.
      state.textContent=`显示器播放未确认，请稍后重试：${error.message}`;
      return;
    }
  }
  currentAudio=new Audio(`/api/media/${encodeURIComponent(media.media_id)}`);
  replay.hidden=false;
  currentAudio.addEventListener('play',()=>{expression('speaking');state.textContent='星伴正在通过手机说话…';});
  const audio=currentAudio;
  currentAudio.addEventListener('ended',()=>{if(currentAudio!==audio)return;expression('happy');state.textContent='我说完啦';emitEvent('speech_finished');});
  currentAudio.addEventListener('error',()=>{if(currentAudio!==audio)return;state.textContent='手机音频播放失败';emitEvent('failed');});
  currentAudio.play().catch(()=>{const button=document.createElement('button');button.className='playback';button.textContent='点击播放星伴的语音';button.addEventListener('click',()=>currentAudio.play());state.replaceChildren(button);});
}
function playbackSettled(result){return result.settled===true;}
async function playbackRequest(action,playbackId){
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),3000);
  const path=action==='stop'?'/api/device/audio/stop':'/api/device/audio/status';
  try{return await readJson(await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({playback_id:playbackId}),signal:controller.signal}));}
  finally{clearTimeout(timer);}
}
async function stopPhysicalPlayback(){
  if(!currentPlayback)return;
  const playbackId=currentPlayback;
  let result=await playbackRequest('stop',playbackId);
  const deadline=Date.now()+5000;
  while(!playbackSettled(result)&&Date.now()<deadline){
    await new Promise(resolve=>setTimeout(resolve,300));
    result=await playbackRequest('status',playbackId);
  }
  if(!playbackSettled(result))throw new Error('扬声器仍在停止，请稍后重试');
  if(currentPlayback===playbackId)currentPlayback=null;
}
async function stopSpeech(){
  if(currentAudio){currentAudio.pause();currentAudio=null;await emitEvent('speech_finished');}
  await stopPhysicalPlayback();
}
async function watchPhysicalPlayback(playbackId,started=Date.now(),failures=0){
  if(currentPlayback!==playbackId)return;
  if(Date.now()-started>190000){state.textContent='扬声器状态未确认，请稍后重试停止';return;}
  try{
    const result=await playbackRequest('status',playbackId);
    if(currentPlayback!==playbackId)return;
    if(playbackSettled(result)){
      currentPlayback=null;
      expression(result.state==='completed'?'happy':'worried');
      state.textContent=result.state==='completed'?'显示器扬声器回答完成':'显示器扬声器已停止';
      return;
    }
    failures=0;
  }catch(error){if(currentPlayback!==playbackId)return;failures++;state.textContent=`暂时无法确认扬声器状态：${error.message}`;}
  setTimeout(()=>watchPhysicalPlayback(playbackId,started,failures),Math.min(2000,500*(1+failures)));
}
async function physicalVoiceTurn(){if(busy)return;busy=true;boardMic.disabled=true;mic.disabled=true;send.disabled=true;try{await stopSpeech();expression('gentle');state.textContent='请靠近板载麦克风自然说话，说完后会自动结束…';const data=await readJson(await fetch('/api/starbuddy/device-turn',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}));if(!data.detected){const floor=data.capture?.turn?.noise_floor_dbfs;throw new Error(`没有检测到清晰声音${floor===undefined?'':`（噪声底 ${floor} dBFS）`}，请靠近麦克风重试。`);}input.value=data.transcript;mood.textContent=data.answer;expression(inferMood(data.transcript+data.answer));connection.textContent='全板端自然语音';currentPlayback=data.playback.playback_id;state.textContent=`听见了：${data.transcript} · 实体扬声器正在回答…`;watchPhysicalPlayback(currentPlayback);}catch(error){mood.textContent='暂时不能开始下一轮，请检查提示。';expression('worried');state.textContent=error.message;}finally{busy=false;boardMic.disabled=false;mic.disabled=false;send.disabled=false;}}
function observationText(data){const counts=(data?.stable_summary||data?.summary)?.counts||{};const labels=Object.entries(counts).slice(0,5).map(([label,count])=>`${label}×${count}`);return labels.length?`我看到了：${labels.join('、')}`:'暂时没有稳定识别出已知物体';}
async function observeOnce(){if(busy)return;busy=true;observe.disabled=true;expression('thinking');state.textContent='环境摄像头正在板端观察…';try{const data=await readJson(await fetch('/api/device/vision/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({role:'camera.environment',profile:'fast',confidence:0.35,stability_samples:2})}));mood.textContent=observationText(data);expression('happy');state.textContent=data.change?.provisional?'正在确认环境变化':(data.change?.changed?'检测到稳定环境变化':'结构化观察完成');connection.textContent='板端 QNN 视觉';}catch(error){expression('worried');state.textContent=error.message;}finally{busy=false;observe.disabled=false;}}
async function refreshMonitor(){if(!monitoring)return;try{const data=await readJson(await fetch('/api/device/vision/monitor/status',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}));if(data.last_observation){mood.textContent=observationText(data.last_observation);state.textContent=`持续观察：${data.samples} 帧，${data.changes} 次变化`;}if(!data.active){monitoring=false;monitor.textContent='持续观察：关';monitor.setAttribute('aria-pressed','false');return;}}catch(error){state.textContent=error.message;}setTimeout(refreshMonitor,1800);}
async function toggleMonitor(){monitor.disabled=true;try{if(monitoring){await readJson(await fetch('/api/device/vision/monitor/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}));monitoring=false;monitor.textContent='持续观察：关';monitor.setAttribute('aria-pressed','false');state.textContent='持续观察已停止';}else{await readJson(await fetch('/api/device/vision/monitor/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({role:'camera.environment',profile:'fast',confidence:0.35,stability_samples:2,interval_seconds:5})}));monitoring=true;monitor.textContent='持续观察：开';monitor.setAttribute('aria-pressed','true');expression('thinking');state.textContent='持续观察已启动';refreshMonitor();}}catch(error){state.textContent=error.message;}finally{monitor.disabled=false;}}
async function processRecording(stream,mimeType){stream.getTracks().forEach(track=>track.stop());mic.classList.remove('recording');mic.setAttribute('aria-label','开始录音');busy=true;send.disabled=true;mic.disabled=true;expression('thinking');state.textContent='板端正在识别声音…';await emitEvent('input_ready');try{const baseMime=mimeType.startsWith('audio/ogg')?'audio/ogg':'audio/webm';const blob=new Blob(audioChunks,{type:baseMime});const media=await readJson(await fetch('/api/media/audio',{method:'POST',headers:{'Content-Type':baseMime},body:blob}));state.textContent='本地 Agent 正在思考和合成声音…';const data=await readJson(await fetch('/api/starbuddy/voice',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_id:media.media_id})}));input.value=data.transcript;mood.textContent=data.answer;expression(inferMood(data.transcript+data.answer));state.textContent=`听见了：${data.transcript}`;connection.textContent='板端语音 AI';await emitEvent('response_ready');playBoardAudio(data.audio);}catch(error){mood.textContent='我刚刚没有听清，我们再试一次吧。';expression('worried');state.textContent=error.message;await emitEvent('failed');}finally{audioChunks=[];recorder=null;busy=false;send.disabled=false;mic.disabled=false;}}
async function startRecording(){if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder)throw new Error('当前浏览器无法采集原始音频。');await stopSpeech();await emitEvent('reset');const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true},video:false});await emitEvent('listen_started');const mimeType=recordingMime();recorder=new MediaRecorder(stream,mimeType?{mimeType}:undefined);audioChunks=[];recorder.ondataavailable=event=>{if(event.data.size)audioChunks.push(event.data);};recorder.onstop=()=>processRecording(stream,mimeType||recorder.mimeType);recorder.start();mic.classList.add('recording');mic.setAttribute('aria-label','停止并发送');state.textContent='手机只在采集原始声音，再点一次发送';expression('gentle');}
async function talk(message,startingMood='thinking'){
  if(busy||!message.trim())return;busy=true;send.disabled=true;input.disabled=true;expression(startingMood==='gentle'?'gentle':'thinking');state.textContent='正在认真想…';mood.textContent='嗯，我听见了。';
  try{await stopSpeech();await emitEvent('reset');await emitEvent('input_ready');const data=await readJson(await fetch('/api/starbuddy/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message})}));mood.textContent=data.answer;expression(inferMood(message+data.answer));state.textContent=data.audio?'板端已生成语音':'文字回应完成';connection.textContent=data.audio?'板端语音 AI':'本地 AI';await emitEvent('response_ready');await playBoardAudio(data.audio);}
  catch(error){mood.textContent='我刚刚走神了一下。我们可以再试一次。';expression('worried');state.textContent=error.message;connection.textContent='暂时离线';await emitEvent('failed');}
  finally{busy=false;send.disabled=false;input.disabled=false;input.focus();}
}
document.querySelectorAll('.moments button').forEach(button=>button.addEventListener('click',()=>talk(button.dataset.prompt,button.dataset.mood)));
form.addEventListener('submit',event=>{event.preventDefault();const message=input.value.trim();input.value='';talk(message);});
face.addEventListener('click',()=>talk('我轻轻碰了碰你，请用一句俏皮又温暖的话回应我。','happy'));
mic.addEventListener('click',async()=>{try{if(recorder?.state==='recording'){recorder.stop();return;}if(busy)return;await startRecording();}catch(error){mood.textContent='麦克风还没有准备好。';expression('worried');state.textContent=`无法录音：${error.message}`;}});
boardMic.addEventListener('click',physicalVoiceTurn);
observe.addEventListener('click',observeOnce);
monitor.addEventListener('click',toggleMonitor);
speaker.addEventListener('click',async()=>{if(busy)return;try{await stopSpeech();outputMode=outputMode==='browser'?'device':outputMode==='device'?'off':'browser';updateOutputLabel();state.textContent=outputMode==='device'?'回复将从显示器扬声器播放':outputMode==='browser'?'回复将从手机播放':'语音播放已关闭';}catch(error){state.textContent=error.message;}});
replay.addEventListener('click',()=>{if(!currentAudio)return;currentAudio.currentTime=0;currentAudio.play().catch(()=>{state.textContent='浏览器暂时没有允许播放，请再点一次。';});});
updateOutputLabel();
