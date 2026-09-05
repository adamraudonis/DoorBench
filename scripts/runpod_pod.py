#!/usr/bin/env python3
"""Create / inspect / connect to / terminate the DoorBench Isaac Lab GPU pod on RunPod (REST API v1).

Replicable one-command flow (see docs/RUNPOD.md):

    export RUNPOD_API_KEY=rpa_...                 # Settings -> API Keys (Read & Write), never commit it
    python scripts/runpod_pod.py create           # L40S / RTX 6000 Ada / A40 / 4090, CUDA 12.8+, 150 GB volume
    python scripts/runpod_pod.py wait             # blocks until SSH is reachable, prints the ssh command
    python scripts/runpod_pod.py bootstrap        # copies scripts/pod_bootstrap.sh and runs it in tmux (~25 min)
    python scripts/runpod_pod.py ssh              # interactive shell
    python scripts/runpod_pod.py status           # GPU, cost/h, uptime, spend so far
    python scripts/runpod_pod.py tensorboard      # live training curves at http://localhost:6006 (SSH tunnel)
    python scripts/runpod_pod.py watch            # live stage / iteration / GPU status of isaaclab/cloud/run_all.sh
    python scripts/runpod_pod.py stop / start     # pause GPU billing keeping /workspace, resume later
    python scripts/runpod_pod.py terminate        # stops billing (volume is deleted too)

The pod id is remembered in ~/.runpod/doorbench_pod.json.  Only uses the standard library.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
STATE = Path.home() / ".runpod" / "doorbench_pod.json"
KEY_FILE = Path.home() / ".ssh" / "runpod_doorbench"
HERE = Path(__file__).resolve().parent.parent

DEFAULT_POD = {
    "name": "doorbench-isaaclab",
    # Ubuntu 22.04 + CUDA 12.8 + sshd; the bootstrap creates a Python 3.11 uv venv for Isaac Sim 5.1 / Isaac Lab v2.3.2
    "imageName": "runpod/pytorch:1.2.0-rc.162-cu1281-torch271-ubuntu2204",
    "computeType": "GPU",
    "cloudType": "SECURE",
    # Isaac Sim needs RT cores (A100/H100 are NOT supported); priority order = best supported first
    "gpuTypeIds": ["NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA A40", "NVIDIA GeForce RTX 4090"],
    "gpuTypePriority": "custom",
    "gpuCount": 1,
    "allowedCudaVersions": ["12.8", "12.9", "13.0"],   # host driver >= 570 (Isaac Sim 5.1 wheels)
    "minVCPUPerGPU": 8,
    "minRAMPerGPU": 32,
    "volumeInGb": 150,          # Isaac Sim + extension cache ~ 40 GB, Isaac Lab, DoorBench, checkpoints, videos
    "containerDiskInGb": 60,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],        # direct SSH (public IP + mapped port); scp/rsync work
    "interruptible": False,
}


def _key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        cfg = Path.home() / ".runpod" / "config.toml"
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.strip().startswith("apikey"):
                    k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        sys.exit("RUNPOD_API_KEY not set (Settings -> API Keys on runpod.io)")
    return k


def _req(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API {method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}")


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(d: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, indent=1))


def _ssh_key():
    if not KEY_FILE.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "doorbench-runpod", "-f", str(KEY_FILE)], check=True)
    return KEY_FILE.with_suffix(".pub").read_text().strip()


def cmd_create(a):
    body = dict(DEFAULT_POD)
    body["env"] = {"PUBLIC_KEY": _ssh_key(), "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"}
    if a.gpu:
        body["gpuTypeIds"] = [a.gpu]
    pod = _req("POST", "/pods", body)
    _save({"id": pod["id"], "created": time.time(), "costPerHr": pod.get("costPerHr")})
    print(f"created pod {pod['id']} ({pod.get('costPerHr')} $/h) - run `wait` next")


def _pod():
    st = _state()
    if not st.get("id"):
        sys.exit("no pod recorded; run `create` first")
    return _req("GET", f"/pods/{st['id']}")


def cmd_wait(a):
    for _ in range(90):
        pod = _pod()
        ip, pm = pod.get("publicIp"), pod.get("portMappings") or {}
        if ip and pm.get("22"):
            st = _state(); st.update({"ip": ip, "port": pm["22"], "gpu": (pod.get("gpu") or {}).get("displayName")}); _save(st)
            (STATE.parent / "ssh").write_text(f"{ip} {pm['22']}\n")      # `read IP PORT < ~/.runpod/ssh` for shell one-liners
            print(f"ssh -i {KEY_FILE} -p {pm['22']} root@{ip}")
            return
        print("waiting for public IP / port 22 ...", pod.get("desiredStatus"), flush=True)
        time.sleep(10)
    sys.exit("pod did not expose SSH in 15 min")


def _ssh_args():
    st = _state()
    if not st.get("ip"):
        cmd_wait(None)
        st = _state()
    return ["ssh", "-i", str(KEY_FILE), "-o", "StrictHostKeyChecking=accept-new", "-p", str(st["port"]), f"root@{st['ip']}"]


def cmd_ssh(a):
    os.execvp("ssh", _ssh_args() + (a.cmd or []))


def cmd_bootstrap(a):
    st = _state()
    script = HERE / "scripts" / "pod_bootstrap.sh"
    subprocess.run(["scp", "-i", str(KEY_FILE), "-o", "StrictHostKeyChecking=accept-new", "-P", str(st["port"]), str(script), f"root@{st['ip']}:/workspace/pod_bootstrap.sh"], check=True)
    subprocess.run(_ssh_args() + ["(command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)); "
                                   "tmux new-session -d -s boot 'bash /workspace/pod_bootstrap.sh > /workspace/bootstrap.log 2>&1'; "
                                   "echo started; echo 'follow with: python scripts/runpod_pod.py ssh tail -f /workspace/bootstrap.log'"], check=True)


def cmd_status(a):
    pod = _pod(); st = _state()
    hours = (time.time() - st.get("created", time.time())) / 3600
    print(json.dumps({"id": pod["id"], "status": pod.get("desiredStatus"), "gpu": (pod.get("gpu") or {}).get("displayName"),
                      "ip": pod.get("publicIp"), "ssh_port": (pod.get("portMappings") or {}).get("22"), "costPerHr": pod.get("costPerHr"),
                      "hours_since_create": round(hours, 2), "spend_estimate_usd": round(hours * float(pod.get("costPerHr") or 0), 2)}, indent=1))


def cmd_stop(a):
    """Stop the pod: GPU billing stops, the /workspace volume (and its install) is kept (volume storage is still billed)."""
    st = _state()
    _req("POST", f"/pods/{st['id']}/stop")
    print(f"stopped pod {st['id']} (GPU billing paused; `start` resumes with the same /workspace)")


def cmd_start(a):
    """Resume a stopped pod (the public IP / SSH port usually change: run `wait` afterwards)."""
    st = _state()
    _req("POST", f"/pods/{st['id']}/start")
    st.pop("ip", None); st.pop("port", None); _save(st)
    print(f"start requested for pod {st['id']} - run `wait` next")


def cmd_tensorboard(a):
    """Live training curves: start TensorBoard on the pod (rsl_rl writes logs/rsl_rl/<exp>/<run>) and tunnel it to http://localhost:6006."""
    st = _state()
    remote = ("source /workspace/DoorBench/isaaclab/cloud/env.sh 2>/dev/null; cd /workspace/DoorBench; "
              "pgrep -f 'tensorboard --logdir' >/dev/null || nohup tensorboard --logdir logs/rsl_rl --port 6006 --bind_all >/workspace/tensorboard.log 2>&1 & "
              "sleep 2; echo 'TensorBoard on the pod: http://localhost:6006 (through this tunnel). Ctrl-C closes the tunnel.'; sleep 100000000")
    os.execvp("ssh", _ssh_args() + ["-L", f"{a.port}:localhost:6006", remote])


def cmd_watch(a):
    """Follow the pipeline: stage markers of logs/run_all.log, latest training iteration/reward, GPU utilisation."""
    remote = ("cd /workspace/DoorBench; while true; do clear; date -u; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; "
              "echo; grep -E '^== |STAGE_|isaacsim-validate\\]|RUN_ALL DONE|checkpoint:' logs/run_all.log 2>/dev/null | tail -8; echo; "
              "grep -E 'Learning iteration|Mean reward|Mean episode length' logs/run_all.log 2>/dev/null | tail -3; "
              "grep -E 'Traceback|Error:' logs/run_all.log 2>/dev/null | grep -v omni | tail -2; sleep 15; done")
    os.execvp("ssh", _ssh_args() + [remote])


def cmd_terminate(a):
    st = _state()
    _req("DELETE", f"/pods/{st['id']}")
    print(f"terminated pod {st['id']} (billing stopped)")
    STATE.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create"); p.add_argument("--gpu", help="force one gpuTypeId, e.g. 'NVIDIA L40S'"); p.set_defaults(f=cmd_create)
    sub.add_parser("wait").set_defaults(f=cmd_wait)
    p = sub.add_parser("ssh"); p.add_argument("cmd", nargs="*"); p.set_defaults(f=cmd_ssh)
    sub.add_parser("bootstrap").set_defaults(f=cmd_bootstrap)
    sub.add_parser("status").set_defaults(f=cmd_status)
    sub.add_parser("stop").set_defaults(f=cmd_stop)
    p = sub.add_parser("tensorboard"); p.add_argument("--port", type=int, default=6006); p.set_defaults(f=cmd_tensorboard)
    sub.add_parser("watch").set_defaults(f=cmd_watch)
    sub.add_parser("start").set_defaults(f=cmd_start)
    sub.add_parser("terminate").set_defaults(f=cmd_terminate)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
