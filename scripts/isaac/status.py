#!/usr/bin/env python3
"""Persist preparation stage so verbose Kit logs cannot hide current progress."""
import json
import sys
import time
from pathlib import Path
path=Path(sys.argv[1])
if path.exists():
    data=json.loads(path.read_text());data.update(stage=sys.argv[2],stage_updated_unix=time.time())
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data)+'\n');tmp.replace(path)
