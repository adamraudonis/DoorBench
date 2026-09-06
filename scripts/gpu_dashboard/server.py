#!/usr/bin/env python3
"""DoorBench local Run Center. No credentials or remote control exposed to the browser."""

import argparse
import json
import shlex
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def snapshot(run):
    command = ["python3", "-", run["results"]]
    if run.get("ssh_host"):
        if run["ssh_host"].startswith("-"):
            raise ValueError("Invalid SSH host")
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-p",
            str(run.get("ssh_port", 22)),
        ]
        if run.get("ssh_key"):
            args += ["-i", str(Path(run["ssh_key"]).expanduser())]
        # Quote every remote argument; none is interpreted as shell code.
        args += [run["ssh_host"], shlex.join(command + ["--gpu"])]
    else:
        args = command
    proc = subprocess.run(
        args,
        input=(HERE / "collect.py").read_text(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode:
        # Don't send SSH paths, config or arbitrary stderr to the browser.
        raise RuntimeError(
            "Cannot read remote run (SSH unavailable or path missing)"
            if run.get("ssh_host")
            else "Cannot read local run (path missing or ledger being written)"
        )
    return json.loads(proc.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("out/gpu-dashboard/runs.json"))
    p.add_argument("--port", type=int, default=5183)
    p.add_argument(
        "--results", help="Register a local directory or remote result directory"
    )
    p.add_argument(
        "--register-only",
        action="store_true",
        help="Update the registry without starting a second server",
    )
    p.add_argument("--name", default="G1 catalogue")
    p.add_argument("--id", default="g1-catalogue")
    p.add_argument("--ssh-host")
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--ssh-key")
    p.add_argument(
        "--teardown", help="Local confirmed teardown receipt for an archived run"
    )
    a = p.parse_args()
    a.config = a.config.resolve()
    a.config.parent.mkdir(parents=True, exist_ok=True)
    if a.results:
        runs = json.loads(a.config.read_text()) if a.config.exists() else []
        item = dict(
            id=a.id,
            name=a.name,
            results=a.results if a.ssh_host else str(Path(a.results).resolve()),
        )
        if a.ssh_host:
            item.update(ssh_host=a.ssh_host, ssh_port=a.ssh_port, ssh_key=a.ssh_key)
        if a.teardown:
            item["teardown"] = str(Path(a.teardown).resolve())
        runs = [r for r in runs if r["id"] != a.id] + [item]
        temp = a.config.with_suffix(".tmp")
        temp.write_text(json.dumps(runs, indent=2) + "\n")
        temp.replace(a.config)
    if not a.config.exists():
        a.config.write_text("[]\n")
    if a.register_only:
        print(f"Registry updated: {a.config}", flush=True)
        return
    cache = {}
    lock = threading.Lock()

    def refresh():
        while True:
            try:
                runs = json.loads(a.config.read_text())
                for run in runs:
                    id = run["id"]
                    old = cache.get(id, {})
                    try:
                        data = snapshot(run)
                        terminated = False
                        if run.get("teardown"):
                            try:
                                terminated = (
                                    json.loads(Path(run["teardown"]).read_text()).get(
                                        "confirmed_absent"
                                    )
                                    is True
                                )
                            except (OSError, ValueError):
                                pass
                        item = dict(
                            id=id,
                            name=run["name"],
                            source="SSH"
                            if run.get("ssh_host")
                            else "Local archive"
                            if data["complete"]
                            else "Local files",
                            fetched_at=time.time(),
                            error=None,
                            terminated=terminated,
                            data=data,
                        )
                    except Exception as e:
                        item = {
                            **old,
                            "id": id,
                            "name": run.get("name", id),
                            "error": str(e),
                            "last_poll": time.time(),
                        }
                    with lock:
                        cache[id] = item
                with lock:
                    for id in list(cache):
                        if id not in {r["id"] for r in runs}:
                            del cache[id]
            except (OSError, ValueError, TypeError):
                pass
            time.sleep(5)

    threading.Thread(target=refresh, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") not in (
                f"127.0.0.1:{a.port}",
                f"localhost:{a.port}",
            ):
                self.send_error(403)
                return
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/runs":
                with lock:
                    body = json.dumps(
                        {"runs": list(cache.values()), "server_time": time.time()},
                        allow_nan=False,
                    ).encode()
                kind = "application/json"
            elif path in ("/", "/index.html"):
                body = (HERE / "index.html").read_bytes()
                kind = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(
        f"DoorBench Run Center: http://127.0.0.1:{a.port}  (config: {a.config})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
