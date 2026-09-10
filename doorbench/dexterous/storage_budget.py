"""Keep simulation evidence from consuming the host's swap/workspace reserve."""
import math
from pathlib import Path
import shutil

RESERVE_BYTES = 10 * 1024**3
# Conservative initial allowance from observed full-state/contact diagnostics.
# This is storage admission, not a claim about every robot's evidence size.
EVIDENCE_BYTES_PER_SECOND = 16 * 1024**2


def check_storage(path, *, seconds_remaining=0):
    if not math.isfinite(seconds_remaining) or seconds_remaining < 0:
        raise ValueError('Finite remaining duration required')
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    required = RESERVE_BYTES + math.ceil(seconds_remaining * EVIDENCE_BYTES_PER_SECOND)
    if free < required:
        raise OSError(f'Insufficient evidence storage: {free/1024**3:.2f} GiB free, '
                      f'{required/1024**3:.2f} GiB required including 10 GiB host reserve. '
                      'Archive completed runs before starting or continuing physics.')
    return dict(free_bytes=free, required_bytes=required, reserve_bytes=RESERVE_BYTES)


RETAINED_LIMIT_BYTES = 20 * 1024**3


def check_retained_budget(roots, *, incoming_bytes=0):
    """Bound retained evidence across roots; count hard-linked files once.

    APFS clone sharing is not exposed here, so this is a conservative logical
    byte budget rather than a promise of physical space reclaimed by deletion.
    """
    if incoming_bytes < 0:
        raise ValueError('Nonnegative incoming evidence size required')
    seen=set();total=0
    for root in roots:
        root=Path(root)
        if not root.exists():continue
        for p in root.rglob('*'):
            if p.is_symlink() or not p.is_file():continue
            try: st=p.stat()
            except FileNotFoundError: continue
            key=(st.st_dev,st.st_ino)
            if key in seen:continue
            seen.add(key);total+=st.st_size
    if total+incoming_bytes > RETAINED_LIMIT_BYTES:
        raise OSError(f'Retained evidence budget exceeded: {total/1024**3:.2f} GiB retained '
                      f'plus {incoming_bytes/1024**3:.2f} GiB reserved; limit 20 GiB. '
                      'Archive or prune completed development runs before dispatch.')
    return dict(retained_bytes=total,retained_limit_bytes=RETAINED_LIMIT_BYTES)
