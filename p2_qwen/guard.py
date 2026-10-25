"""Persistent aggregate wall-clock cap and independent child-process watchdog.

All local accelerated invocations MUST run under this entry point. Local timers charge
whole subprocess wall time (including load/IO), conservatively exceeding GPU busy time.
Remote ledger charges wall clock from provider billing start to verified release.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)


def check_compute_deadline():
    """Final evaluation uses its compute reserve, not the earlier training stop."""
    if not os.environ.get('P2_GUARDED') or 'P2_DEADLINE_UNIX' not in os.environ:
        raise RuntimeError('External cumulative budget guard required')
    if time.time()>float(os.environ['P2_DEADLINE_UNIX']):
        raise TimeoutError('Final compute reserve reached; save and release')


def main():
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    p=argparse.ArgumentParser();p.add_argument('--ledger',default='evidence/p2_qwen/local_budget.json')
    p.add_argument('--seconds',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();path=Path(a.ledger);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(path.read_text()) if path.exists() else dict(cap_seconds=1800.,used_seconds=0.,events=[])
        if state.get('active'):
            # Crash is charged conservatively until now; no restart reset.
            state['used_seconds']+=max(0,time.time()-state['active']['start_unix'])
            state['events'].append(dict(state.pop('active'),status='crash_reconciled'))
        remaining=state['cap_seconds']-state['used_seconds']
        allowance=min(a.seconds,remaining-10)
        if allowance<=0: raise TimeoutError('Aggregate local budget exhausted')
        cmd=a.command[1:] if a.command[:1]==['--'] else a.command
        if not cmd: raise ValueError('Command required')
        start=time.time();state['active']=dict(start_unix=start,allowance=allowance,command=cmd)
        atomic_json(path,state)
        reserve=min(60,allowance/4)
        env=dict(os.environ,P2_GUARDED='1',P2_DEADLINE_UNIX=str(start+allowance-reserve),PYTORCH_ENABLE_MPS_FALLBACK='0',P2_BUDGET_LEDGER=str(path.resolve()))
        proc=subprocess.Popen(cmd,env=env,start_new_session=True)
        try:
            rc=proc.wait(timeout=allowance)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            try:rc=proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
        finally:
            elapsed=time.time()-start
            state['used_seconds']+=elapsed
            state['events'].append(dict(state.pop('active'),elapsed_seconds=elapsed,returncode=proc.returncode))
            atomic_json(path,state)
        print(json.dumps(dict(local_used_seconds=state['used_seconds'],remaining_seconds=state['cap_seconds']-state['used_seconds'])),flush=True)
        raise SystemExit(rc)


if __name__=='__main__':main()
