(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const labels={queued:'问候已接收，等待反馈',claimed:'一个轻轻的拥抱正在送达',completed:'这份问候已经播完',silent:'安静的拥抱已经送达',busy:'现在忙着互动，这次没有打扰',expired:'这次问候已过期，没有补播',cancelled:'这次问候已取消',failed:'这次问候没有完成',unknown:'无法确认完成，不会自动重播'};
  let timer,codeTimer,busy=false,editing=false,hiddenEpoch=0,lastReceiving=null,leaving=false;
  function api(action,data={}){return window.StarBuddyMode ? window.StarBuddyMode.track(()=>request(action,data)) : request(action,data);}
  async function request(action,data={}) {
    const c=new AbortController(),t=setTimeout(()=>c.abort(),4000);
    try {const r=await fetch('/api/hugs/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:c.signal,cache:'no-store'});const j=await r.json();if(!r.ok)throw Error(j.error||'暂时不可用');return j;}finally{clearTimeout(t);}
  }
  function clearCode(){clearTimeout(codeTimer);$('code').textContent='';$('invitation').hidden=true;}
  function text(id,value){if($(id).textContent!==value)$(id).textContent=value;}
  function render(s){if(!editing){for(const id of ['enabled','muted'])if($(id).checked!==s[id])$(id).checked=s[id];}text('paired',`已授权 ${s.paired_devices} 台`);
    const r=s.latest; text('receipt',r?(labels[r.state]||'状态未知'):'还没有新的拥抱');const receiving=Boolean(r&&r.state==='claimed');
    if(receiving!==lastReceiving){$('receipt').classList.toggle('receiving',receiving);lastReceiving=receiving;}
  }
  async function refresh(){clearTimeout(timer);if(document.hidden||leaving)return;try{render(await api('status'));}catch(e){$('status').textContent='状态连接中断，请稍后刷新';}finally{if(!document.hidden&&!leaving)timer=setTimeout(refresh,2000);}}
  async function action(name,data={}){if(busy)return;busy=true;const epoch=hiddenEpoch;
    for(const id of ['apply','invite','revoke','demo'])$(id).disabled=true;
    try{const r=await api(name,data);if(name==='invite'&&!document.hidden&&epoch===hiddenEpoch){clearCode();$('code').textContent=r.code;$('invitation').hidden=false;codeTimer=setTimeout(clearCode,r.expires_in*1000);}
      else if(name==='settings'||name==='revoke'){editing=false;clearCode();render(r);}
      $('status').textContent=name==='demo'?'试播已提交；下方回执显示实际结果。':'已更新';
    }catch(e){$('status').textContent=e.message;}finally{busy=false;for(const id of ['apply','invite','revoke','demo'])$(id).disabled=false;refresh();}}
  ['enabled','muted'].forEach(id=>$(id).addEventListener('change',()=>{editing=true;}));
  $('apply').addEventListener('click',()=>action('settings',{enabled:$('enabled').checked,muted:$('muted').checked}));
  $('invite').addEventListener('click',()=>action('invite'));
  $('revoke').addEventListener('click',()=>action('revoke'));
  $('demo').addEventListener('click',()=>action('demo',{kind:'hug',request_id:Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('')}));
  document.addEventListener('visibilitychange',()=>{hiddenEpoch++;clearCode();clearTimeout(timer);if(!document.hidden)refresh();});
  window.addEventListener('pagehide',()=>{hiddenEpoch++;clearCode();clearTimeout(timer);});
  refresh();
  window.StarBuddyMode?.configure({name:'远程拥抱',stopLabel:'开启勿扰 / 停止接收',
    freeze(){leaving=true;hiddenEpoch++;clearCode();clearTimeout(timer);},
    async stop(exit){if(!exit){editing=false;render(await api('settings',{enabled:false,muted:true}));}},
    async ready(){return true;},resume(){leaving=false;refresh();},
    stopped:'已开启勿扰。在途问候正在撤回，声音停止可能稍有延迟；退出页面不会撤销授权。'});
})();
