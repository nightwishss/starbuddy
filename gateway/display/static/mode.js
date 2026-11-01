/* Small navigation barrier, not another session manager or audio owner. */
(() => {
  'use strict';
  function createGate({sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms)), now=()=>Date.now()}={}) {
    let pending=0, running=false;
    return {
      async track(work) {pending++;try{return await work();}finally{pending--; }},
      async settle(adapter, exit=false) {
        if(running)return false;
        running=true;
        try {
          adapter.freeze(exit);
          const deadline=now()+20000;
          await adapter.stop(exit);
          // A start request already in flight can reach the server after stop.
          // Wait for its reply, then stop again before accepting readiness.
          // Always reassert once after drain: an older request may have finished
          // while the first stop itself was in flight.
          let drained=false;
          while(now()<deadline) {
            if(pending===0) {
              if(!drained){await adapter.stop(exit);drained=true;}
              if(await adapter.ready())return true;
            } else drained=false;
            await sleep(500);
          }
          return false;
        } finally {running=false;}
      }
    };
  }
  if(typeof module==='object' && module.exports){module.exports={createGate};return;}
  const gate=createGate();
  let adapter=null, running=false, locked=false;
  const destinations=[['/','主页'],['/companion','聊天陪伴'],['/story','共创故事'],['/quiet','晚安 / 电台'],['/masks','情绪面具'],['/journal','树洞 / 日记'],['/hugs','远程拥抱'],['/posture','坐姿守护'],['/preferences','我的节奏']];
  const bar=document.createElement('nav');bar.id='mode-bar';bar.setAttribute('aria-label','模式控制');
  const label=document.createElement('strong'),select=document.createElement('select');select.setAttribute('aria-label','切换模式');
  const placeholder=document.createElement('option');placeholder.value='';placeholder.textContent='切换模式…';select.append(placeholder);
  for(const [path,name] of destinations){const o=document.createElement('option');o.value=path;o.textContent=name;select.append(o);}
  const stop=document.createElement('button'),back=document.createElement('button'),message=document.createElement('span');
  stop.type=back.type='button';stop.textContent='■ 停止当前';back.textContent='退出 · 回主页';message.id='mode-message';message.setAttribute('role','status');
  bar.append(label,select,stop,back,message);
  function lock(value){locked=value;document.body.classList.toggle('mode-locked',value);for(const node of document.body.children)if(node!==bar && node.tagName!=='SCRIPT'){node.inert=value;}}
  async function act(target=null) {
    if(running || !adapter)return false;
    running=true;lock(true);stop.disabled=back.disabled=select.disabled=true;
    message.textContent='正在停止当前活动，请稍等…';
    try {
      if(!await gate.settle(adapter,Boolean(target)))throw Error('尚未确认结束。请稍后再点停止或退出；不会启动另一模式。');
      if(target){location.assign(target);return true;}
      adapter.resume?.();lock(false);message.textContent=adapter.stopped || '当前活动已停止，可以继续或切换模式。';return true;
    } catch(error){message.textContent=error.message || '连接未确认，请重试停止或退出。';return false;}
    finally {running=false;stop.disabled=back.disabled=select.disabled=false;select.value='';}
  }
  async function json(path,data) {
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),4000);
    try{const r=await fetch(path,data===undefined?{cache:'no-store',signal:controller.signal}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:controller.signal});
      if(!r.ok)throw Error('服务连接未确认，请重试停止或退出。');return await r.json();
    }finally{clearTimeout(timer);}
  }
  window.StarBuddyMode={
    track:work=>gate.track(work), json,
    ready:async()=> (await json('/api/mode/status')).ready===true,
    stop:()=>act(), exit:(path='/')=>act(path),
    configure(value){adapter=value;label.textContent=value.name;stop.textContent=value.stopLabel || '■ 停止当前';
      document.body.classList.add('has-mode-bar');document.body.prepend(bar);},
    get locked(){return locked;}
  };
  stop.addEventListener('click',()=>act());back.addEventListener('click',()=>act('/'));
  window.addEventListener('pageshow',event=>{if(event.persisted)location.reload();});
  select.addEventListener('change',()=>{if(select.value)act(select.value);});
  // Existing back links and app links use the same barrier, not sendBeacon alone.
  document.addEventListener('click',event=>{
    const link=event.target.closest?.('a[href]');
    if(!link || !adapter)return;
    const url=new URL(link.href,location.href);
    if((url.origin===location.origin && destinations.some(([path])=>path===url.pathname)) || url.href==='http://127.0.0.1:8795/?mode=manage'){
      event.preventDefault();event.stopImmediatePropagation();act(url.href);
    }
  },true);
  // Also guard old WebKit versions without native inert support.
  for(const kind of ['click','submit','keydown'])document.addEventListener(kind,event=>{
    if(locked && !bar.contains(event.target)){event.preventDefault();event.stopImmediatePropagation();}
  },true);
})();
