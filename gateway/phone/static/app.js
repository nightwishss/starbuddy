const messages=document.getElementById('messages');
const form=document.getElementById('chatForm');
const prompt=document.getElementById('prompt');
const send=document.getElementById('send');
const mic=document.getElementById('mic');
const statusNode=document.getElementById('status');
const chatMode=document.getElementById('chatMode');
const cameraPanel=document.getElementById('cameraPanel');
const video=document.getElementById('camera');
let stream=null;
let recorder=null;
let audioChunks=[];

function addMessage(kind,text){const item=document.createElement('article');item.className=`message ${kind}`;const badge=document.createElement('span');badge.textContent=kind==='agent'?'H':'你';const body=document.createElement('div');body.textContent=text;item.append(badge,body);messages.append(item);item.scrollIntoView({behavior:'smooth',block:'end'});}
function setBusy(busy,label='Hermes 思考中…'){send.disabled=busy;prompt.disabled=busy;mic.disabled=busy;statusNode.textContent=busy?label:'已连接';statusNode.classList.toggle('busy',busy);}
async function readJson(response){let data={};try{data=await response.json();}catch{}if(!response.ok)throw new Error(data.error||`请求失败（${response.status}）`);return data;}
function playBoardAudio(media){if(!document.getElementById('autoSpeak').checked)return;const audio=new Audio(`/api/media/${encodeURIComponent(media.media_id)}`);audio.play().catch(()=>{const button=document.createElement('button');button.className='playback';button.textContent='播放板端语音回复';button.addEventListener('click',()=>audio.play());messages.append(button);});}

function backendLabel(backend){if(backend==='local-qwen')return '板端 Qwen';if(backend==='local-agent')return '本地 Agent';return 'Hermes Agent';}
chatMode.addEventListener('change',()=>{prompt.placeholder=chatMode.value==='local'?'和板端 Qwen 聊聊…':chatMode.value==='local-agent'?'让本地 Agent 检查板卡状态…':'让 Hermes 检查或操作设备…';});
form.addEventListener('submit',async(event)=>{event.preventDefault();const message=prompt.value.trim();if(!message)return;addMessage('user',message);prompt.value='';const busyLabel=chatMode.value==='local'?'板端 Qwen 推理中…':chatMode.value==='local-agent'?'本地 Agent 调用板卡能力中…':'Hermes 思考中…';setBusy(true,busyLabel);try{const data=await readJson(await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,mode:chatMode.value})}));addMessage('agent',`[${backendLabel(data.backend)}] ${data.answer}`);}catch(error){addMessage('error',error.message);}finally{setBusy(false);prompt.focus();}});

function recordingMime(){for(const type of ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus']){if(MediaRecorder.isTypeSupported(type))return type;}return '';}
async function startRecording(){if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder)throw new Error('当前浏览器无法采集原始音频。');const audioStream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true},video:false});const mimeType=recordingMime();recorder=new MediaRecorder(audioStream,mimeType?{mimeType}:undefined);audioChunks=[];recorder.ondataavailable=(event)=>{if(event.data.size)audioChunks.push(event.data);};recorder.onstop=()=>processRecording(audioStream,mimeType||recorder.mimeType);recorder.start();mic.textContent='停止并发送';mic.classList.add('recording');statusNode.textContent='手机正在采集原始音频';}
async function processRecording(audioStream,mimeType){audioStream.getTracks().forEach(track=>track.stop());mic.textContent='麦克风';mic.classList.remove('recording');setBusy(true,'板端正在识别语音…');try{const baseMime=mimeType.startsWith('audio/ogg')?'audio/ogg':'audio/webm';const blob=new Blob(audioChunks,{type:baseMime});const media=await readJson(await fetch('/api/media/audio',{method:'POST',headers:{'Content-Type':baseMime},body:blob}));statusNode.textContent=chatMode.value==='local'?'板端 Qwen 思考并合成语音…':'Hermes 思考并合成语音…';const result=await readJson(await fetch('/api/voice/turn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_id:media.media_id,mode:chatMode.value})}));addMessage('user',result.transcript);addMessage('agent',`[${backendLabel(result.backend)}] ${result.answer}`);playBoardAudio(result.audio);}catch(error){addMessage('error',`板端语音处理失败：${error.message}`);}finally{audioChunks=[];recorder=null;setBusy(false);}}
mic.addEventListener('click',async()=>{try{if(recorder?.state==='recording'){recorder.stop();return;}await startRecording();}catch(error){addMessage('error',`无法录音：${error.message}`);}});

document.getElementById('cameraButton').addEventListener('click',async()=>{try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});video.srcObject=stream;cameraPanel.hidden=false;}catch(error){addMessage('error',`无法打开摄像头：${error.message}`);}});
function closeCamera(){if(stream){stream.getTracks().forEach(track=>track.stop());stream=null;}cameraPanel.hidden=true;video.srcObject=null;}
document.getElementById('closeCamera').addEventListener('click',closeCamera);
document.getElementById('snapshot').addEventListener('click',()=>{const canvas=document.getElementById('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);canvas.toBlob(async(blob)=>{try{statusNode.textContent='板端 NPU 正在识别…';const media=await readJson(await fetch('/api/camera/snapshot',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob}));const result=await readJson(await fetch('/api/vision/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_id:media.media_id,confidence:0.35})}));if(!result.count){addMessage('agent','板端 NPU 没有检测到置信度足够的物体。');}else{const summary=result.detections.map(item=>`${item.label} ${(item.confidence*100).toFixed(1)}%`).join('、');addMessage('agent',`板端 NPU 检测到 ${result.count} 个物体：${summary}`);}}catch(error){addMessage('error',`板端视觉识别失败：${error.message}`);}finally{statusNode.textContent='已连接';}},'image/jpeg',0.86);});
