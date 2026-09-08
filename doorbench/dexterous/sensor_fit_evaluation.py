"""Full-history fitting diagnostics; never physical recovery or task success."""
import json
import numpy as np
import torch

from .sensor_actor import native_motor_forces
from .autoregressive_training import actor_history_chunk


def evaluate_frozen_fit(model, episodes, names, *, readiness_limits=None,include_actor_history_sources=False):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    scores = {};owned_scores={}
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
            if include_actor_history_sources:
                if episode.times[0]!=0 or np.any(episode.numeric['previous_action'][0]!=0):
                    raise ValueError('Full actor-history evaluation requires actual zero-command reset')
                owned_hidden=None;previous=torch.zeros((1,episode.dimensions.actions),device=device)
                errors=np.zeros(episode.dimensions.actions)
                for start in range(0,len(episode),64):
                    values,target=episode.sequence(start,min(64,len(episode)-start))
                    owned,owned_hidden,previous,_=actor_history_chunk(model,
                        {k:torch.as_tensor(v[None],device=device) for k,v in values.items()},
                        previous=previous,hidden=owned_hidden)
                    errors+=np.sum((owned[0].cpu().numpy()-target)**2,axis=0,dtype=np.float64)
                motor_names=episode.layout['action_order']
                groups=dict(all=list(range(episode.dimensions.actions)),
                    body=[i for i,n in enumerate(motor_names) if not n.startswith(('rh_','lh_'))])
                motors=json.loads((episode.path/'motor-contract.json').read_text())
                scale=np.array([(a['force_range'][1]-a['force_range'][0])*.5 for a in motors['actuators']])
                owned_scores[name]=dict(examples=len(episode),last_decision_time_s=float(episode.times[len(episode)-1]),
                    normalized_force_mse={group:float(errors[ids].mean()/len(episode)) for group,ids in groups.items()},
                    motor_force_rmse_Nm={group:float(np.sqrt(np.mean(errors[ids]*scale[ids]**2)/len(episode))) for group,ids in groups.items()})

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
        full_source_actor_history=owned_scores,
        actor_history_evaluation_scope=('One zero-command reset per source; actual recorded sensor states, continuous model-owned previous commands and GRU state across all chunks. Extra diagnostic only; original six criteria remain unchanged.'
            if include_actor_history_sources else 'Additional per-source actor history not requested'),
        readiness_checks=checks, relative_comparison_roundoff_margin=1e-6,
        ready_for_independent_runtime_checks=bool(checks) and all(checks.values()),
        limitation='Training data fit only; passing also requires runtime checks before any new physical attempt, and does not establish closed-loop stability.')
