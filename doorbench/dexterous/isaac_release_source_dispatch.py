"""Explicit source-kind dispatch; completed archives retain their old path."""
from pathlib import Path

PAUSED_SOURCE_KIND = 'paused-live-isaac-transfer-v1'


def validate_source_kind(source_kind):
    if source_kind is not None and source_kind != PAUSED_SOURCE_KIND:
        raise ValueError('Unknown explicit Isaac release source kind')
    return source_kind


def admit_release_planning_source(source, *, robot, door_xml, door_usd,
        profile='volar-phalange-v1', source_kind=None, phase_audit_path=None):
    validate_source_kind(source_kind)
    if source_kind is None:
        if phase_audit_path is not None:
            raise ValueError('Paused phase audit requires explicit live source kind')
        from .isaac_release_planning import admit_isaac_release_context
        return admit_isaac_release_context(source, robot=robot, door_xml=door_xml,
            door_usd=door_usd, profile=profile)
    from .isaac_paused_release_context import admit_paused_isaac_release_context
    return admit_paused_isaac_release_context(source, robot=robot, door_xml=door_xml,
        door_usd=door_usd, profile=profile, phase_audit_path=phase_audit_path)


def require_planning_context(context):
    from .isaac_release_planning import IsaacReleasePlanningContext
    if isinstance(context, IsaacReleasePlanningContext):return None
    if getattr(context, 'admission', {}).get('source_kind') != PAUSED_SOURCE_KIND:
        raise ValueError('Explicit freshly admitted actual Isaac source context required')
    from .isaac_paused_release_context import PausedIsaacReleasePlanningContext
    if type(context) is not PausedIsaacReleasePlanningContext:
        raise ValueError('Explicit freshly admitted actual Isaac source context required')
    if context.admission.get('source_kind') != PAUSED_SOURCE_KIND:
        raise ValueError('Paused planning context must retain its distinct source kind')
    return PAUSED_SOURCE_KIND


def source_paths(source_kind):
    validate_source_kind(source_kind)
    if source_kind is None:return (Path(__file__).resolve(),)
    from .isaac_paused_release_context import paused_release_source_paths
    return (Path(__file__).resolve(), *paused_release_source_paths())
