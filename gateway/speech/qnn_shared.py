"""Authenticated loopback QNN transport in the existing Pose HTTP worker.

Only fixed TTS graph operations and owned scratch paths are accepted. No user
text, media, device control or arbitrary model path crosses this interface.
"""
from __future__ import annotations

import ctypes
import hmac
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time

MAX_RESPONSE = 512*1024
OWNER = re.compile(r'tts-qnn-smoke-[a-f0-9]{12}')
COMPONENTS = {'bert_wrapper', 'encoder', 'flow', 'decoder'}


def exchange(root, request, timeout=15):
    body=(json.dumps(request,separators=(',',':'))+'\n').encode()
    if len(body)>2048: raise ValueError('shared request too large')
    fd=os.open(str(Path(root)/'auth.token'),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'r') as stream:
        detail=os.fstat(stream.fileno())
        if not stat.S_ISREG(detail.st_mode) or detail.st_uid!=os.getuid() or detail.st_mode&0o077:
            raise PermissionError('private transport credential required')
        token=stream.read(65)
    if not re.fullmatch(r'[a-f0-9]{64}',token):raise ValueError('invalid transport credential')
    # Fixed existing loopback endpoint: no proxy, redirect, URL/model selection.
    connection=http.client.HTTPConnection('127.0.0.1',8793,timeout=timeout)
    try:
        connection.request('POST','/_internal/qnn-tts',body,
                           {'Content-Type':'application/json','Authorization':'Bearer '+token})
        response=connection.getresponse();data=response.read(MAX_RESPONSE+1)
        if response.status!=200 or len(data)>MAX_RESPONSE:raise RuntimeError('invalid shared response')
    finally:connection.close()
    result=json.loads(data)
    if not isinstance(result,dict) or result.get('ok') is not True:
        raise RuntimeError('shared QNN request failed')
    return result


class SharedCore:
    def __init__(self, root, models, library, pose_lock, uid, *, clock=time.monotonic):
        self.root,self.models,self.uid=Path(root).resolve(),Path(models).resolve(),uid
        self.clock,self.lock=clock,pose_lock
        self.lib=ctypes.CDLL(str(library))
        self.lib.vq_qnn_open.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_size_t]
        self.lib.vq_qnn_open.restype=ctypes.c_int
        self.lib.vq_qnn_run.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.vq_qnn_run.restype=ctypes.c_int
        self.lib.vq_qnn_close.argtypes=[ctypes.c_void_p];self.lib.vq_qnn_close.restype=ctypes.c_int
        self.handle,self.owner,self.last,self.failed=ctypes.c_void_p(),None,0,False
        self.opens,self.runs=0,0
        self.last_error=None

    def state_path(self, value):
        if not isinstance(value,str) or len(value)>1024:raise ValueError('state path')
        requested=Path(value)
        root=self.root/'clients'
        if not requested.is_absolute() or not requested.resolve().is_relative_to(root):raise ValueError('state scope')
        for path in (requested,*requested.parents):
            if path==root:break
            detail=path.lstat()
            if not stat.S_ISDIR(detail.st_mode) or detail.st_uid!=self.uid or detail.st_mode&0o077:
                raise ValueError('private owned directories required')
        return requested

    def close(self):
        if self.handle.value:
            result=self.lib.vq_qnn_close(self.handle)
            self.handle=ctypes.c_void_p()
            if result:
                self.failed=True
                raise RuntimeError('shared native close failed')
        self.owner=None

    def sweep(self):
        with self.lock:
            if self.owner and self.clock()-self.last>20:self.close()

    def status(self):
        return {'transport':'shared-pose-process','ready':not self.failed,
                'tts_resident':self.owner is not None,'context_opens':self.opens,'graph_runs':self.runs,
                'last_error':self.last_error}

    def dispatch(self, request, uid):
        if uid!=self.uid:raise PermissionError('peer uid')
        if not isinstance(request,dict):raise ValueError('request')
        op,owner=request.get('op'),request.get('owner')
        if not isinstance(owner,str) or not OWNER.fullmatch(owner):raise ValueError('owner')
        with self.lock:
            if self.failed:raise RuntimeError('shared native quarantined')
            if op=='close' and owner!=self.owner:
                if set(request)!={'op','owner'}:raise ValueError('extra fields')
                return {'ok':True,'event':'CLOSED','resources_released':True}
            if op=='open':
                if set(request)!={'op','owner','state'} or self.owner is not None:raise ValueError('busy/invalid open')
                path=self.state_path(request['state'])
                buffer=ctypes.create_string_buffer(MAX_RESPONSE)
                code=self.lib.vq_qnn_open(str(self.models).encode(),str(path).encode(),self.uid,
                    ctypes.byref(self.handle),buffer,len(buffer))
                if code:raise RuntimeError('shared native open failed')
                self.owner,self.last=owner,self.clock();self.opens+=1
                return json.loads(buffer.value)
            if owner!=self.owner:raise PermissionError('owner mismatch')
            self.last=self.clock()
            if op in ('ping','close'):
                if set(request)!={'op','owner'}:raise ValueError('extra fields')
                if op=='close':self.close()
                return {'ok':True,'event':'CLOSED' if op=='close' else 'ALIVE','resources_released':op=='close'}
            if op!='run' or set(request)!={'op','owner','component','token'}:raise ValueError('operation')
            component,token=request['component'],request['token']
            if component not in COMPONENTS or not isinstance(token,str) or not re.fullmatch(r'r[0-9]{1,6}',token):
                raise ValueError('graph/token')
            buffer=ctypes.create_string_buffer(MAX_RESPONSE)
            code=self.lib.vq_qnn_run(self.handle,component.encode(),token.encode(),buffer,len(buffer))
            if code:
                self.close();raise RuntimeError('shared native run failed')
            self.runs+=1
            return json.loads(buffer.value)


def handle_http(handler,core):
    if core is None:return handler._json(404,{'ok':False})
    auth=handler.headers.get('Authorization','')
    if not hmac.compare_digest(auth.encode('utf-8'),('Bearer '+core.token).encode('ascii')):
        return handler._json(403,{'ok':False})
    try:
        length=int(handler.headers.get('Content-Length','0'))
        if handler.headers.get('Transfer-Encoding') or not 1<=length<=2048:raise ValueError('request length')
        handler.connection.settimeout(2)
        data=handler.rfile.read(length)
        if len(data)!=length:raise ValueError('incomplete request')
        result=core.dispatch(json.loads(data),core.uid)
    except Exception as exc:
        core.last_error=type(exc).__name__
        return handler._json(400,{'ok':False,'error':'shared_qnn_unavailable'})
    return handler._json(200,result)


def start_shared(root, models, library, pose_lock, uid=1000):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('shared root required')
    detail=root.stat()
    if detail.st_uid!=uid or detail.st_mode&0o077:raise ValueError('private shared root required')
    core=SharedCore(root,models,library,pose_lock,uid)
    core.token=secrets.token_hex(32)
    fd=os.open(str(root/'auth.token'),os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:
        detail=os.fstat(stream.fileno())
        if not stat.S_ISREG(detail.st_mode) or detail.st_nlink!=1 or detail.st_mode&0o077 or detail.st_uid not in {uid,os.getuid()}:
            raise PermissionError('unsafe credential file')
        if os.getuid()!=uid:os.fchown(stream.fileno(),uid,-1)
        stream.write(core.token);stream.truncate()
    def serve():
        try:
            while True:
                time.sleep(1);core.sweep()
        except Exception:
            core.failed=True
        finally:
            with pose_lock:core.close()
    threading.Thread(target=serve,name='qnn-tts-shared',daemon=True).start()
    return core
