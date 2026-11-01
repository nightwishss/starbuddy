document.getElementById('pairForm').addEventListener('submit',async(event)=>{
  event.preventDefault();
  const error=document.getElementById('error');
  error.textContent='';
  try{
    const response=await fetch('/api/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:document.getElementById('code').value})});
    const data=await response.json();
    if(response.ok){location.href='/';return;}
    error.textContent=data.error||'连接失败';
  }catch(_error){error.textContent='无法连接板卡，请检查Wi-Fi。';}
});
