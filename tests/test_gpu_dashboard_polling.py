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


def test_registry_order_updates_without_waiting_for_archives_to_refresh():
    cache = {name: dict(id=name, last_poll=100., data=dict(status='completed'))
             for name in ('old','new')}
    monitor = module.RunMonitor(cache, threading.Lock(), clock=lambda:100.)
    try:
        monitor.poll([run('new'),run('old')])
        assert cache['new']['registry_index']==0 and cache['old']['registry_index']==1
        assert not monitor.pending
        monitor.poll([run('old'),run('new')])
        assert cache['old']['registry_index']==0 and cache['new']['registry_index']==1
    finally: monitor.pool.shutdown()


def test_duplicate_registry_ids_are_rejected_before_polling():
    import pytest
    monitor=module.RunMonitor({},threading.Lock(),fetch=lambda r:None,workers=1)
    try:
        with pytest.raises(ValueError,match='unique'):
            monitor.poll([run('same'),dict(run('same'),results='/different')])
        assert not monitor.pending
    finally:monitor.pool.shutdown()


def test_newest_unknown_run_is_not_starved_by_old_archives():
    started=[];done=threading.Event()
    def fetch(item):
        started.append(item['id']);done.set();return dict(complete=False,status='running')
    monitor=module.RunMonitor({},threading.Lock(),fetch=fetch,workers=1)
    try:
        monitor.poll([run('old-'+str(i)) for i in range(100)]+[run('new-live')])
        assert done.wait(1) and started==['new-live']
    finally:monitor.pool.shutdown()


def test_dead_archive_backoff_keeps_capacity_for_live_refresh():
    cache={'dead':dict(id='dead',last_poll=100.,error='unavailable',consecutive_errors=5),
           'live':dict(id='live',last_poll=100.,error='temporary',consecutive_errors=5,data=dict(status='running'))}
    monitor=module.RunMonitor(cache,threading.Lock(),fetch=lambda r:dict(complete=False,status='running'),workers=1,clock=lambda:110.)
    try:
        monitor.poll([run('dead'),run('live')]);assert set(monitor.pending)=={'live'}
    finally:monitor.pool.shutdown()
