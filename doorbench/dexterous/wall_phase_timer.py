"""Main-thread wall timing; never a GPU kernel-time or policy observation."""
import time


class WallPhaseTimer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.previous = None
        self.totals = {}
        self.steps = 0

    def start(self):
        if self.previous is not None:
            raise ValueError('Previous timing interval must finish')
        self.previous = self.clock()

    def mark(self, name):
        if self.previous is None:
            raise ValueError('Timing interval has not started')
        now = self.clock()
        elapsed = now - self.previous
        if elapsed < 0:
            raise ValueError('Monotonic wall clock required')
        self.totals[name] = self.totals.get(name, 0.) + elapsed
        self.previous = now

    def finish(self, name):
        self.mark(name)
        self.previous = None
        self.steps += 1

    def receipt(self):
        return dict(schema='doorbench.main-thread-wall-timing.v1',
                    completed_steps=self.steps, partial_interval=self.previous is not None, phase_seconds=dict(self.totals),
                    total_seconds=sum(self.totals.values()),
                    scope='Main-thread wall time including waits; asynchronous GPU work may be charged to later synchronization. Excludes startup and final export. Not a batch-size benchmark.')
