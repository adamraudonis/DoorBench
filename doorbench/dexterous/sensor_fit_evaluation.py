"""Full-history fitting diagnostics; never physical recovery or task success."""
import json
import numpy as np
import torch

from .sensor_actor import native_motor_forces


def evaluate_frozen_fit(model, episodes, names, *, readiness_limits=None):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    scores = {}
    with torch.inference_mode():
        for name, episode in zip(names, episodes, strict=True):
            hidden = None
            sums = {key: 0. for key in ('learned', 'persistence', 'zero')}
            for start in range(0, len(episode), 64):
                length = min(64, len(episode)-start)
                values, target = episode.sequence(start, length)
                pred, hidden = model(**{k: torch.as_tensor(v[None], device=device) for k,v in values.items()}, hidden=hidden)
                pred = pred[0].cpu().numpy()
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite frozen-history predictions')
                for key, estimate in (('learned', pred), ('persistence', episode.numeric['previous_action'][start:start+length]), ('zero', np.zeros_like(target))):
                    sums[key] += float(np.sum((estimate-target)**2, dtype=np.float64))
            scores[name] = {key: value/(len(episode)*episode.dimensions.actions) for key,value in sums.items()}

        nominal = episodes[0]
        if len(nominal) < 251 or nominal.times[0] != 0 or nominal.metadata['physics_dt_s'] != .002:
            raise ValueError('Frozen startup comparison needs the actual nominal cold reset')
        motors = json.loads((nominal.path/'motor-contract.json').read_text())
        caps = np.array([a['force_range'] for a in motors['actuators']])
        body = [i for i,n in enumerate(nominal.layout['action_order']) if not n.startswith(('rh_', 'lh_'))]
        errors = []; hidden = None; previous = np.zeros(nominal.dimensions.actions, np.float32)
        for i in range(251):
            packet = nominal.packet(i)
            packet['previous_action'] = previous
            previous, hidden = model.act(packet, float(nominal.times[i]), hidden)
            force = native_motor_forces(previous, caps)
            target = native_motor_forces(nominal.numeric['previous_action'][i+1], caps)
            errors.append((force-target)[body])
        errors = np.array(errors)
    model.train(was_training)
    startup = dict(body_force_rmse_Nm_first100ms=float(np.sqrt(np.mean(errors[:51]**2))),
        body_force_rmse_Nm_first500ms=float(np.sqrt(np.mean(errors**2))))
    checks = {}
    if readiness_limits:
        # Different summation precision must not promote the baseline itself.
        checks.update({name: scores[name]['learned'] < bound*(1-1e-6) for name,bound in readiness_limits['dataset_mse'].items()})
        checks.update({name: startup[name] < bound*(1-1e-6) for name,bound in readiness_limits['startup'].items()})
    return dict(scope=__doc__, closed_loop_evaluated=False, task_success_rate=None,
        recurrent_history='One reset; complete actual sensor history. Startup uses actor-owned previous commands on nominal recorded sensor states.',
        examples={name: len(episode) for name,episode in zip(names, episodes, strict=True)},
        prediction_mse_normalized_force=scores, startup=startup,
        readiness_checks=checks, relative_comparison_roundoff_margin=1e-6,
        ready_for_independent_runtime_checks=bool(checks) and all(checks.values()),
        limitation='Training data fit only; passing also requires runtime checks before any new physical attempt, and does not establish closed-loop stability.')
