#!/usr/bin/env python3
"""Record selected G1 trials in a compact native Isaac grid; selection is not a score."""

from grid_g1 import main, app, traceback, sys

if __name__ == "__main__":
    try:
        main(spacing=7.5, presentation=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        app.close()
        raise
    else:
        app.close()
