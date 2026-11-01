/* Explicit local choices only. No periodic polling, browser storage or model calls. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  let state=null,busy=false,controller=null;
  const describe=v=>`${v.speed.toFixed(1)}× 语速 · ${v.reply_length==='brief'?'简短回答':'正常说明'}`;
  function render(value){
    state=value;
    $('speed').value=String(value.values.speed);$('reply-length').value=value.values.reply_length;
    $('remember').checked=value.remember;
    $('effective').textContent=describe(value.values);
    $('retention').textContent=value.expires_at?`${value.remember?'已同意在本机记住':'临时使用'}，${new Date(value.expires_at*1000).toLocaleString()} 到期恢复默认。`:'当前为默认值，没有记住偏好。';
    $('change').textContent=value.last_change?`这次变化：${describe(value.last_change.before)} → ${describe(value.last_change.after)}。依据：你的明确选择。`:value.source==='expired'?'上次选择已到期；没有重新猜测你的偏好。':'不根据表情、声音或身份自动修改。';
    $('notice').textContent=value.storage_error?'本机旧设置不可用，已使用默认值；可以恢复默认并忘记，或请管理员检查存储。':'设置已确认；后续输出按此节奏。可以去聊天试试。';
  }
  async function request(action,data){
    controller=new AbortController();const timer=setTimeout(()=>controller.abort(),4000);
    try{const response=await fetch('/api/preferences/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:controller.signal});
      const result=await response.json();if(!response.ok)throw Error(result.error||'设置未确认');return result;
    }finally{clearTimeout(timer);controller=null;}
  }
  async function run(action){
    if(busy || (action!=='status' && !state))return;
    const data=action==='status'?{}:action==='reset'?{revision:state.revision}:{revision:state.revision,speed:Number($('speed').value),reply_length:$('reply-length').value,remember:$('remember').checked};
    busy=true;$('controls').disabled=true;$('refresh').disabled=true;
    $('notice').textContent='正在确认设置…';
    try{render(await request(action,data));}
    catch(error){state=null;$('effective').textContent='状态尚未确认';$('retention').textContent='';$('change').textContent='';$('notice').textContent=(error.name==='AbortError'?'连接超时，结果未确认':error.message)+'；请刷新状态，不会自动重复保存。';}
    finally{busy=false;$('controls').disabled=!state;$('refresh').disabled=false;}
  }
  $('preferences-form').addEventListener('submit',event=>{event.preventDefault();run('update');});
  $('reset').addEventListener('click',()=>run('reset'));
  $('refresh').addEventListener('click',()=>run('status'));
  document.addEventListener('click',event=>{if(busy && event.target.closest?.('a[href]')){event.preventDefault();$('notice').textContent='正在确认设置，请稍等；确认后即可离开。';}},true);
  window.addEventListener('pagehide',()=>controller?.abort());
  window.addEventListener('pageshow',event=>{if(event.persisted)run('status');});
  run('status');
})();
