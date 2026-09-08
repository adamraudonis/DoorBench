import importlib.util
from pathlib import Path
import threading


spec = importlib.util.spec_from_file_location('run_center_server',
    Path(__file__).resolve().parents[1]/'scripts/gpu_dashboard/server.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run(id):
    return dict(id=id, name=id, results='/'+id)


def test_slow_archive_does_not_delay_publication_or_refresh_of_live_run():
    release, began, fast = threading.Event(), threading.Event(), threading.Event()
    cache = {}; now = [100.]
    def fetch(item):
        if item['id']=='archive':
            began.set(); assert release.wait(3)
            return dict(complete=True, status='completed')
        fast.set()
        return dict(complete=False, status='running')
    monitor = module.RunMonitor(cache, threading.Lock(), fetch=fetch, workers=2, clock=lambda:now[0])
    runs = [run('archive'), run('live')]
    try:
        monitor.poll(runs)
        assert began.wait(1) and fast.wait(1)
        monitor.pending['live'][1].result(timeout=1)
        monitor.poll(runs)
        assert cache['live']['data']['status']=='running' and 'archive' not in cache
        now[0] += 5
        fast.clear(); monitor.poll(runs)
        assert fast.wait(1)
        monitor.pending['live'][1].result(timeout=1)
        monitor.poll(runs)
        assert cache['live']['fetched_at']==105 and 'archive' not in cache
    finally:
        release.set(); monitor.pool.shutdown()


def test_changed_or_removed_registry_entry_cannot_receive_stale_result():
    release = threading.Event(); cache = {}
    def fetch(item):
        assert release.wait(3)
        return dict(complete=True, status='completed', root=item['results'])
    monitor = module.RunMonitor(cache, threading.Lock(), fetch=fetch, workers=1)
    try:
        monitor.poll([run('old')])
        release.set(); monitor.pending['old'][1].result(timeout=1)
        monitor.poll([])
        assert cache=={} and monitor.pending=={}
    finally: monitor.pool.shutdown()
