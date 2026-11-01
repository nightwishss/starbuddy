"""Use the checked graph/tensor adapter through the existing Pose process."""
from pathlib import Path
from qnn_tts_chain import QnnChainClient, validate_components
from gateway.speech.qnn_shared import exchange


class SharedQnnClient(QnnChainClient):
    def __init__(self, *args, shared_root, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_root,self.answer,self.opened=Path(shared_root),None,False

    def start(self):
        import time
        began=time.monotonic()
        self.opened=True  # A timed-out open may still have reached the server.
        self.ready=exchange(self.shared_root,{'op':'open','owner':self.owner,'state':str(self.root)})
        if self.ready.get('backend')!='QNN HTP' or self.ready.get('cpu_fallback') is not False:
            raise ValueError('unexpected shared backend')
        validate_components(self.ready['components'])
        self.process=True
        self.initialization_ms=(time.monotonic()-began)*1000
        return self.ready

    def _send(self, command):
        _,component,token=command.strip().split('\t')
        self.answer=exchange(self.shared_root,{'op':'run','owner':self.owner,'component':component,'token':token})

    def _response(self, timeout):
        return self.answer

    def heartbeat(self):
        if self.opened:exchange(self.shared_root,{'op':'ping','owner':self.owner},timeout=2)

    def close(self):
        if self._cleanup is not None:return dict(self._cleanup)
        if self.opened:
            result=exchange(self.shared_root,{'op':'close','owner':self.owner},timeout=8)
            if result.get('resources_released') is not True:raise RuntimeError('shared context close unconfirmed')
        self.opened=False;self.process=None;self.log.close()
        self._cleanup={'native_resources_released':True,'context_release_confirmed':True,
                       'transport':'shared-pose-process'}
        return dict(self._cleanup)
