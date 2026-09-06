#!/usr/bin/env python3
"""Check installed dependencies, retaining one explicit upstream Isaac pin exception."""

import importlib.metadata as metadata
import json
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check():
    packages = {
        canonicalize_name(d.metadata["Name"]): d
        for d in metadata.distributions()
        if d.metadata["Name"]
    }
    errors, exceptions = [], []
    for name, dist in sorted(packages.items()):
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(req.name)
            installed = packages.get(dependency)
            if installed is not None and (
                not req.specifier or installed.version in req.specifier
            ):
                continue
            row = {
                "package": name,
                "requirement": str(req),
                "installed": installed.version if installed else None,
            }
            # Isaac Lab 2.3.2 pins Starlette 0.49.1, incompatible with the FastAPI
            # 0.115.7 pin in Isaac Sim 5.1. FastAPI 0.121.0 supports Starlette <0.50.
            accepted = (
                name == "isaacsim-kernel"
                and dist.version in ("5.1.0", "5.1.0.0")
                and dependency == "fastapi"
                and str(req.specifier) == "==0.115.7"
                and installed is not None
                and installed.version == "0.121.0"
                and packages.get("starlette") is not None
                and packages["starlette"].version == "0.49.1"
            )
            (exceptions if accepted else errors).append(row)
    report = {
        "packages": len(packages),
        "errors": errors,
        "documented_upstream_exceptions": exceptions,
    }
    print(json.dumps(report, indent=2))
    if not errors:
        print("RUNTIME_CHECK_OK", flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(check())
