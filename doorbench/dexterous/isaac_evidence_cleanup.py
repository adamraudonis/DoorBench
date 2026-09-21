"""Best-effort evidence finalization that preserves the original failure.

This module performs file cleanup only. It never repairs existing evidence,
changes qualification, or suppresses a failed controller/backend run.
"""
import json
from pathlib import Path
import traceback


def export_once(writer, target):
    """Skip only this writer's successful export to this exact destination."""
    target = Path(target)
    exported = writer.exported_path
    if exported is None:
        writer.export(target)  # Retain BoundedEvidence's fresh-file protection.
        return 'exported'
    if Path(exported).resolve() != target.resolve() or not target.is_file():
        raise ValueError('Already finalized evidence has a different or missing destination')
    return 'already_exported_by_this_writer'


def write_new_json(target, value):
    """A cleanup observation must not overwrite an unrelated existing file."""
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(target).open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def _error(error):
    return dict(type=type(error).__name__, message=str(error),
        traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)))


def preserve_core_evidence(cleanup, out, rows, acquisition_states, pad_steps, physics_enabled):
    """Attempt core outputs once, before report or finalizer errors can escape.

    Failed actions are retained as failures, never retried over partial files.
    The normal producer and exception path share this same preservation point.
    """
    if cleanup.core_preservation_attempted:
        return
    cleanup.core_preservation_attempted = True
    out = Path(out)
    cleanup.attempt('core_trace', lambda:(out/'trace.json').write_text(json.dumps(rows)+'\n'))
    if physics_enabled:
        import numpy as np
        cleanup.attempt('core_physics_archive', lambda:np.savez_compressed(out/'acquisition-physics.npz',**acquisition_states))
        cleanup.export('core_pad_export', pad_steps, out/'acquisition-pad-steps.json.gz')


class EvidenceCleanup:
    def __init__(self, primary_error=None):
        self.primary_error = primary_error
        self.actions = []
        self.errors = []
        self.core_preservation_attempted = False

    def attempt(self, name, action):
        """Attempt every independent preservation action, retaining failures."""
        try:
            value = action()
        except BaseException as error:
            self.errors.append(error)
            self.actions.append(dict(action=name, succeeded=False, error=_error(error)))
            return None
        self.actions.append(dict(action=name, succeeded=True))
        return value

    def export(self, name, writer, target):
        return self.attempt(name, lambda: export_once(writer, target))

    def finish(self, target, *, active_error=None):
        """Write cleanup diagnostics, then propagate the true primary error.

        Called inside the producer's finally block. A primary exception already
        unwinding remains that same exception; secondary cleanup failures become
        notes and a separate receipt. If cleanup alone failed, its first error
        propagates after every registered action was attempted.
        """
        if self.primary_error is None:
            self.primary_error = active_error
        elif active_error is not None and active_error is not self.primary_error:
            self.errors.append(active_error)
            self.actions.append(dict(action='exception_handler', succeeded=False, error=_error(active_error)))
        receipt = dict(schema='doorbench.isaac-evidence-cleanup.v1',
            primary_error=None if self.primary_error is None else _error(self.primary_error),
            cleanup_actions_succeeded=not self.errors, actions=list(self.actions),
            physical_qualification=False, scope='Evidence preservation only; no change to physical result')
        # Never overwrite a previous receipt. If even this write fails, the
        # exception note reaches the producer's original error.txt traceback.
        self.attempt('cleanup_receipt', lambda: write_new_json(target, receipt))
        if self.errors:
            message='Evidence cleanup failures: '+'; '.join(
                type(error).__name__+': '+str(error) for error in self.errors)
            if self.primary_error is not None:
                self.primary_error.add_note(message)
            else:
                self.errors[0].add_note(message)
                raise self.errors[0]
        if self.primary_error is not None and active_error is not self.primary_error:
            raise self.primary_error
        return receipt
