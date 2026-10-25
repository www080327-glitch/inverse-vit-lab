"""Persistent cumulative budget. OS lock prevents simultaneous GPU jobs across commands."""
from pathlib import Path
import contextlib
import fcntl
import json
import os
import time


class Budget:
    def __init__(self, root='evidence/p1', train_limit=18000, eval_limit=18000):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'budget.json'
        self.lock = (self.root/'gpu.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'train_seconds':0.,'eval_seconds':0.,'train_limit':train_limit,'eval_limit':eval_limit,'events':[]}
        if self.state.get('active'):
            active = self.state.pop('active')
            # Crash recovery conservatively bills elapsed wall time to avoid resetting the budget.
            self.state[active['kind']+'_seconds'] += max(0,time.time()-active['start_unix'])
            self.state['events'].append(dict(active, recovered_after_crash=True, end_unix=time.time()))
        self.save()

    def save(self):
        tmp = self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(self.state, indent=2)); tmp.replace(self.path)

    def remaining(self, kind):
        used = self.state[kind+'_seconds']
        if self.state.get('active',{}).get('kind') == kind:
            used += time.time()-self.state['active']['start_unix']
        return self.state[kind+'_limit']-used

    @contextlib.contextmanager
    def charge(self, kind, label):
        if self.remaining(kind) <= 5:
            raise TimeoutError(kind+' budget exhausted')
        active = dict(kind=kind,label=label,pid=os.getpid(),start_unix=time.time())
        self.state['active']=active; self.save()
        try:
            yield
        finally:
            seconds=time.time()-active['start_unix']
            self.state[kind+'_seconds'] += seconds
            self.state.pop('active',None)
            self.state['events'].append(dict(active,seconds=seconds,end_unix=time.time()))
            self.save()

    def close(self):
        self.lock.close()
