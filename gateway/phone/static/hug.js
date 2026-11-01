(() => {
  'use strict';
  const $=id=>document.getElementById(id),buttons=[...document.querySelectorAll('[data-kind]')];
  const labels={queued:'已接收，等待板端反馈',claimed:'星伴正在传递这份问候',completed:'设备已播完问候 ♡',silent:'设备已完成安静反馈 ♡',busy:'板端正在互动，这次没有打扰',expired:'问候已过期，没有补播',cancelled:'这次问候已取消',failed:'这次问候没有完成',unknown:'无法确认结果，不会自动重播'};
  let busy=false,pending=null,timer,deadline=0,paired=false;
  function controls(){buttons.forEach(b=>b.disabled=busy||Boolean(pending)||!paired);$('pair').disabled=busy;$('refresh').disabled=busy;}
  async function api(action,data={}){const c=new AbortController(),t=setTimeout(()=>c.abort(),5000);try{const r=await fetch('/api/hug/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:c.signal,cache:'no-store'});const j=await r.json();if(!r.ok){if(r.status===401){paired=false;$('pair-panel').hidden=false;$('send-panel').hidden=true;}throw Error(j.error||'暂时不可用');}return j;}finally{clearTimeout(t);}}
  async function connect(){if(busy)return;busy=true;controls();try{const s=await api('status');paired=true;$('pair-panel').hidden=true;$('send-panel').hidden=false;$('mode').textContent=!s.enabled?'板端已开启勿扰':s.muted?'安静模式：只显示，不说话':'屏幕 + 温柔语音';$('status').textContent='可以送一个小问候了';}catch(e){$('status').textContent=e.message;}finally{busy=false;controls();}}
  async function poll(){clearTimeout(timer);if(!pending||document.hidden)return;if(Date.now()>deadline){pending=null;$('status').textContent='等待已结束，结果未确认；不会自动重播';controls();return;}
    try{const r=await api('receipt',{request_id:pending});$('status').textContent=labels[r.state]||'状态未知';if(!['queued','claimed'].includes(r.state)){pending=null;controls();return;}}catch(e){$('status').textContent='回执暂不可用，仍在有限等待；不会重发';}
    if(!document.hidden)timer=setTimeout(poll,1000);
  }
  $('pair-form').addEventListener('submit',async e=>{e.preventDefault();if(busy)return;busy=true;controls();const code=$('code').value;$('code').value='';try{await api('pair',{code});busy=false;await connect();}catch(e){$('status').textContent=e.message;}finally{busy=false;controls();}});
  buttons.forEach(b=>b.addEventListener('click',async()=>{if(busy||pending||!paired)return;busy=true;pending=Array.from(crypto.getRandomValues(new Uint8Array(16)),n=>n.toString(16).padStart(2,'0')).join('');deadline=Date.now()+35000;controls();
    try{const r=await api('send',{request_id:pending,kind:b.dataset.kind});$('status').textContent=labels[r.state]||'已提交';if(!['queued','claimed'].includes(r.state))pending=null;}catch(e){$('status').textContent=e.message;}finally{busy=false;controls();if(pending)timer=setTimeout(poll,1000);}
  }));
  $('refresh').addEventListener('click',connect);
  document.addEventListener('visibilitychange',()=>{clearTimeout(timer);if(!document.hidden&&pending)poll();});
  window.addEventListener('pagehide',()=>{clearTimeout(timer);$('code').value='';});
  connect();
})();
