/* Private input never goes to ordinary chat, TTS, analytics or browser storage. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const moods = {unknown:'没有选择',happy:'开心',calm:'平静',sad:'难过',worried:'担心',angry:'生气',mixed:'说不清 / 混合'};
  let mode='tree', token=null, draft=null, configured=null, epoch=0, busy=false, idleTimer, lockTimer;
  let composing=false;
  ['secret','note','passphrase','confirm-passphrase'].forEach(id=>{
    $(id).addEventListener('compositionstart',()=>{composing=true;});
    $(id).addEventListener('compositionend',()=>{composing=false;});
  });
  function api(action,data,keepalive=false){return window.StarBuddyMode ? window.StarBuddyMode.track(()=>request(action,data,keepalive)) : request(action,data,keepalive);}
  async function request(action, data, keepalive=false) {
    const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),10000);
    try {
      const response=await fetch('/api/journal/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),cache:'no-store',signal:controller.signal,keepalive});
      const result=await response.json();
      if (!response.ok) throw new Error(result.error || '暂时不可用');
      return result;
    } finally { clearTimeout(timer); }
  }
  function controls() {
    document.querySelectorAll('[data-action]').forEach(el=>{el.disabled=busy;});
    $('save').disabled=busy || !draft || !$('consent').checked;
    $('unlock').disabled=busy || configured===null;
  }
  async function run(work) {
    if(busy)return;
    const version=epoch; busy=true; controls(); $('status').textContent='';
    try { await work(version); }
    catch(error) { if(version===epoch) $('status').textContent=error.name==='AbortError'?'响应超时。若刚在保存，请刷新日记确认是否已保存。':error.message; }
    finally { if(version===epoch){busy=false;controls();} }
  }
  let lockConfirmed=true;
  const revocations=new Set();
  const revoke=t=>{if(!t)return Promise.resolve();const p=api('lock',{token:t},true).then(()=>revocations.delete(t),()=>{lockConfirmed=false;});revocations.add(t);return p;};
  function eraseDraft() {draft=null;$('preview-panel').hidden=true;$('preview-text').textContent='';$('preview-mood').textContent='';$('consent').checked=false;controls();}
  function clearAll() {
    epoch++;busy=false;composing=false;clearTimeout(idleTimer);clearTimeout(lockTimer);revoke(token);token=null;
    $('secret').value='';$('answer').textContent='临时文字已清空。你可以慢慢来。';
    $('note').value='';$('mood').value='unknown';$('passphrase').value='';$('confirm-passphrase').value='';
    $('entries').replaceChildren();eraseDraft();$('notebook').hidden=true;$('unlock-form').hidden=false;
    $('status').textContent='已清屏并锁定。';controls();
  }
  function activity(){clearTimeout(idleTimer);idleTimer=setTimeout(clearAll,90000);}
  function switchMode(next){
    clearAll();mode=next;$('tree-panel').hidden=mode!=='tree';$('diary-panel').hidden=mode!=='diary';
    $('tree-tab').setAttribute('aria-pressed',mode==='tree');$('diary-tab').setAttribute('aria-pressed',mode==='diary');
    activity(); if(mode==='diary') run(async version=>{
      configured=null;controls();const result=await api('status',{});if(version!==epoch)return;
      configured=result.configured;$('unlock-title').textContent=configured?'打开这台板子的日记本':'先为这台板子设置日记口令';
      $('confirm-wrap').hidden=configured;$('unlock').textContent=configured?'解锁':'设置口令并打开';
    });
  }
  function showEntries(entries){
    $('entries').replaceChildren();
    if(!entries.length){$('entries').textContent='还没有记录。只有你确认保存的内容才会出现在这里。';return;}
    entries.forEach(entry=>{
      const item=document.createElement('article');item.className='entry';
      const meta=document.createElement('div');meta.className='small';
      meta.textContent=new Date(entry.created*1000).toLocaleDateString('zh-CN')+' · 自己写下 · 自选心情：'+(moods[entry.mood]||'没有选择');
      const body=document.createElement('p');body.textContent=entry.text;
      const remove=document.createElement('button');remove.type='button';remove.textContent='删除这条';
      let armed=false;
      remove.addEventListener('click',()=>{
        if(!armed){armed=true;remove.textContent='确认删除？再点一次';return;}
        run(async version=>{await api('delete',{token,entry_id:entry.id,consent:true});if(version!==epoch)return;await list(version);});
      });
      item.append(meta,body,remove);$('entries').append(item);
    });
  }
  async function list(version){const result=await api('list',{token});if(version===epoch)showEntries(result.entries);}
  $('tree-tab').addEventListener('click',()=>switchMode('tree'));
  $('diary-tab').addEventListener('click',()=>switchMode('diary'));
  $('lock').addEventListener('click',clearAll);$('clear').addEventListener('click',clearAll);
  $('back').addEventListener('click',clearAll);
  $('tree-form').addEventListener('submit',event=>{
    event.preventDefault();if(busy||composing)return;
    const text=$('secret').value.trim();if(!text)return;$('secret').value='';
    run(async version=>{const result=await api('listen',{text,mode:$('support').value});if(version===epoch)$('answer').textContent=result.answer;});
  });
  $('unlock-form').addEventListener('submit',event=>{
    event.preventDefault();if(busy||composing||configured===null)return;
    const passphrase=$('passphrase').value, confirmation=$('confirm-passphrase').value;
    $('passphrase').value='';$('confirm-passphrase').value='';
    if(!configured && passphrase!==confirmation){$('status').textContent='两次口令不同，请重新输入。';return;}
    run(async version=>{
      const result=await api(configured?'unlock':'create',{passphrase});
      if(version!==epoch){await revoke(result.token);return;}
      token=result.token;configured=true;$('unlock-form').hidden=true;$('notebook').hidden=false;
      $('unlock-title').textContent='打开这台板子的日记本';$('confirm-wrap').hidden=true;$('unlock').textContent='解锁';
      clearTimeout(lockTimer);lockTimer=setTimeout(clearAll,result.expires_in*1000);
      await list(version);
    });
  });
  function edit(){if(!draft)return;eraseDraft();api('discard',{token}).catch(()=>{});}
  $('note').addEventListener('input',edit);$('mood').addEventListener('change',edit);
  $('note-form').addEventListener('submit',event=>{
    event.preventDefault();if(composing)return;const text=$('note').value.trim(),mood=$('mood').value;if(!text)return;
    run(async version=>{const result=await api('preview',{token,text,mood});
      if(version!==epoch || $('note').value.trim()!==text || $('mood').value!==mood)return;
      draft=result.draft;$('preview-panel').hidden=false;$('preview-text').textContent=draft.text;
      $('preview-mood').textContent='来源：自己写下；心情：'+moods[draft.mood]+'（自选，不是模型判断）';$('consent').checked=false;
    });
  });
  $('consent').addEventListener('change',controls);
  $('discard').addEventListener('click',()=>{epoch++;busy=false;eraseDraft();api('discard',{token}).catch(()=>{});});
  $('save').addEventListener('click',()=>{
    if(!draft||!$('consent').checked)return;const id=draft.id;
    run(async version=>{await api('save',{token,draft_id:id,consent:true});if(version!==epoch)return;
      eraseDraft();$('note').value='';$('mood').value='unknown';$('status').textContent='这条日记已保存在本机。树洞内容没有加入。';await list(version);
    });
  });
  $('refresh').addEventListener('click',()=>run(list));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)clearAll();});
  document.addEventListener('input',activity);document.addEventListener('pointerdown',activity);
  window.addEventListener('pagehide',clearAll);window.addEventListener('pageshow',clearAll);
  clearAll();activity();
  window.StarBuddyMode?.configure({name:'树洞 / 日记',stopLabel:'清屏并锁定',
    freeze(){clearAll();},
    async stop(){for(const t of [...revocations]){await api('lock',{token:t},true);revocations.delete(t);}lockConfirmed=true;},
    async ready(){return lockConfirmed&&revocations.size===0;},
    resume(){activity();},stopped:'临时文字已清空，日记已锁定；已保存日记不删除。'});
})();
