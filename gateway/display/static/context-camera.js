/* Visible local camera + semantic context. No microphone, profile or cloud IO. */
(() => {
  'use strict';
  const button = document.getElementById('context-camera-toggle');
  if (!button) return;
  const status = document.getElementById('context-camera-state');
  const facts = document.getElementById('context-camera-facts');
  const images = [['camera.front', document.getElementById('context-front')],
    ['camera.environment', document.getElementById('context-environment')]];
  let enabled = false, owned = false, away = false, busy = false, poll, epoch = 0, refreshGeneration = 0;
  const workers = [];
  const active = () => !away && !document.hidden;
  async function post(path, keepalive = false) {
    const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
      body:'{}', keepalive});
    if (!r.ok) throw Error('视觉请求未完成');
    return r.json();
  }
  function stopFrames() {
    epoch++;
    for (const w of workers.splice(0)) {clearTimeout(w.timer); clearTimeout(w.deadline);}
    for (const [,img] of images) {img.onload=null; img.onerror=null; img.removeAttribute('src'); img.dataset.live='false';}
  }
  function startFrames() {
    stopFrames();
    if (!enabled || !active()) return;
    const token=epoch;
    for (const [source,img] of images) {
      const w={}; workers.push(w);
      const schedule=delay => {clearTimeout(w.deadline); if(active()&&enabled&&token===epoch) w.timer=setTimeout(pump,delay);};
      const pump=() => {
        if (!active() || !enabled || token!==epoch) return;
        img.onload=()=>{if(token!==epoch)return;img.dataset.live='true';schedule(700);};
        img.onerror=()=>{if(token!==epoch)return;img.dataset.live='false';img.removeAttribute('src');schedule(1300);};
        img.src='/api/camera/source-frame.jpg?source='+encodeURIComponent(source)+'&t='+Date.now();
        w.deadline=setTimeout(()=>{if(token!==epoch)return;img.onload=null;img.onerror=null;img.dataset.live='false';img.removeAttribute('src');schedule(1300);},3500);
      };pump();
    }
  }
  function render(value, identity) {
    const next=value.vision?.enabled===true;
    if (next!==enabled) {enabled=next; enabled?startFrames():stopFrames();}
    button.setAttribute('aria-pressed',String(enabled));
    button.textContent=enabled?'暂停视觉':'开启视觉';
    status.textContent=enabled?(value.vision.error?'视觉已开启，识别暂未完成':'双路预览 · 识别仅更新 Fusion，不自动播报'):'画面未开启；已有后台线索仍会显示，不代表视觉全部关闭。';
    const people=value.people||[], objects=value.objects||[], sources=value.sources||[];
    const front=people.filter(p=>p.source==='camera.front').length;
    const frontObserved=sources.some(s=>s.kind==='face_presence'&&s.source==='camera.front');
    const labels={cup:'杯子',bottle:'瓶子',person:'人',book:'书',chair:'椅子',laptop:'电脑','cell phone':'手机'};
    const summary=objects.slice(0,4).map(o=>(labels[o.label]||o.label)+' ×'+o.count).join('、');
    const label=identity?.status==='known'&&typeof identity.label==='string'?identity.label.slice(0,24):'';
    facts.textContent=(frontObserved?'前置人脸：'+front+' 人':'前置人脸：暂无新鲜线索')+
      ' · '+(label?'镜头里认出：'+label:'当前称呼未确认')+
      ' · '+(summary?'物体：'+summary:'物体：暂无新鲜线索');
  }
  async function refresh() {
    clearTimeout(poll); if(!active())return;
    const generation=++refreshGeneration;
    try {
      const [value,identity]=await Promise.all([
        fetch('/api/companion/perception-status',{cache:'no-store',signal:AbortSignal.timeout(2500)})
          .then(r=>{if(!r.ok)throw Error();return r.json();}),
        fetch('/api/companion/identity-current',{cache:'no-store',signal:AbortSignal.timeout(2500)})
          .then(r=>r.ok?r.json():null).catch(()=>null)
      ]);
      if(active()&&generation===refreshGeneration)render(value,identity);
    } catch (_) {if(active()&&generation===refreshGeneration) facts.textContent='情境连接暂不可用；不会把旧结果当作当前观察。';}
    if(active()&&generation===refreshGeneration)poll=setTimeout(refresh,1600);
  }
  button.addEventListener('click',async()=>{
    if(busy||!active())return;busy=true;button.disabled=true;
    refreshGeneration++;clearTimeout(poll);
    try {const value=await post('/api/companion/vision/'+(enabled?'stop':'start'));
      if(away){
        // A successful Start can arrive after pagehide, before we learned
        // that this page owns the newly started monitor. Do not leave it on.
        if(value.enabled===true&&value.accepted===true)
          post('/api/companion/vision/stop',true).catch(()=>{});
        return;
      }
      if(value.enabled&&value.accepted)owned=true;
      if(!value.enabled)owned=false;
      enabled=value.enabled===true;enabled?startFrames():stopFrames();await refresh();
    }catch(_){status.textContent='视觉切换未确认，请稍后重试。';}
    finally{busy=false;button.disabled=false;}
  });
  document.addEventListener('visibilitychange',()=>{refreshGeneration++;clearTimeout(poll);stopFrames();if(active()){startFrames();refresh();}});
  window.addEventListener('pagehide',()=>{away=true;refreshGeneration++;clearTimeout(poll);stopFrames();if(owned)post('/api/companion/vision/stop',true).catch(()=>{});});
  refresh();
})();
