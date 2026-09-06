"""Atomic, low-frequency run heartbeat for the local GPU dashboard."""

import atexit
import json
import threading
from datetime import datetime, timezone


class RunProgress:
    def __init__(self, directory, total, batch_size):
        self.path = directory / "progress.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.data = dict(
            phase="preparing",
            eligible_doors=total,
            batch_size=batch_size,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._write()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def _write(self):
        with self.lock:
            payload = {
                **self.data,
                "heartbeat_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, indent=2) + "\n")
            temp.replace(self.path)

    def _loop(self):
        while not self.stop.wait(5):
            self._write()

    def update(self, phase, **fields):
        with self.lock:
            self.data.update(phase=phase, **fields)
        self._write()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        if self.data["phase"] != "completed":
            self.update("stopped")
        else:
            self._write()
