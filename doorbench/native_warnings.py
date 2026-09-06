"""Capture MuJoCo messages that have no corresponding MjData warning counter.

The native callback is process-global. Nested scopes are supported and forward
to the previous callback; concurrent native simulations belong in separate
processes, as in the dataset and benchmark workers.
"""
from contextlib import contextmanager
from threading import RLock

_callback_lock=RLock()


@contextmanager
def capture_native_warnings():
    """Yield message strings, forward prior handlers, and restore on exit."""
    import mujoco
    with _callback_lock:
        previous=mujoco.get_mju_user_warning()
        messages=[]
        def warning(message):
            messages.append(str(message))
            if previous is not None:
                previous(message)
        mujoco.set_mju_user_warning(warning)
        try:
            yield messages
        finally:
            mujoco.set_mju_user_warning(previous)
