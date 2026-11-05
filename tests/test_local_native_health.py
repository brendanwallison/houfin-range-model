import json
from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/exp'))
import local_overnight_worker as w


def test_repeated_native_crashes_open_persistent_hold(tmp_path,monkeypatch):
 monkeypatch.setattr(w,'STATE',tmp_path)
 monkeypatch.setattr(w.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='native kernel evidence',stderr=''))
 state={};record={'step':'fixture'}
 assert not w.record_native_failure(state,record,RuntimeError('MAP exit 1; nonfinite objective'))
 assert not state
 assert w.record_native_failure(state,record,RuntimeError('MAP exit -11; native crash'))
 assert state['consecutive_native_failures']==1
 assert not (tmp_path/'NATIVE_CRASH_HOLD.json').exists()
 assert w.record_native_failure(state,record,RuntimeError('MAP exit 139; compiler crash'))
 hold=json.loads((tmp_path/'NATIVE_CRASH_HOLD.json').read_text())
 assert hold['consecutive_native_failures']==2
 assert hold['kernel_tail']=='native kernel evidence'
 assert len(list(tmp_path.glob('native_failure_*.json')))==2


def test_diagnostic_capture_failure_does_not_hide_native_crash(tmp_path,monkeypatch):
 monkeypatch.setattr(w,'STATE',tmp_path)
 def unavailable(*a,**k):raise OSError('kernel log inaccessible')
 monkeypatch.setattr(w.subprocess,'run',unavailable)
 assert w.record_native_failure({},dict(step='fixture'),RuntimeError('MAP exit -6; native abort'))
 receipt=json.loads(next(tmp_path.glob('native_failure_*.json')).read_text())
 assert receipt['kernel_read_error']=='kernel log inaccessible'