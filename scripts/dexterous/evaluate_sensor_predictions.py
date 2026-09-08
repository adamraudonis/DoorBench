"""Evaluate causal next-force prediction on a recorded qualified episode.

Teacher physical states and previous actions are replayed as observations; this
is teacher-forced prediction, never a student rollout or task success metric.
Report the strong 2 ms previous-command persistence baseline explicitly.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from doorbench.dexterous.sensor_actor import ActorDimensions, SensorActor
from doorbench.dexterous.sensor_demonstrations import SensorDemonstration
from doorbench.dexterous.correction_demonstrations import CorrectionDemonstration
from doorbench.dexterous.motor_contract_identity import SENSOR_ACTOR_CHECKPOINT_SCHEMA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--episode", type=Path)
    dataset.add_argument('--correction-dataset',type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qualification",choices=['operation-report.json','acquisition-report.json'],default='operation-report.json')
    parser.add_argument("--legacy-teacher-receipt",type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument('--reset-observation-run',type=Path)
    args = parser.parse_args()
    if min(args.chunk_length, args.threads) <= 0:
        parser.error("Use positive chunk and thread counts")
    torch.set_num_threads(args.threads)
    if args.correction_dataset and (args.legacy_teacher_receipt or args.reset_observation_run):
        parser.error('Correction observations already contain their actual reset and use a separate admission contract')
    episode = CorrectionDemonstration(args.correction_dataset) if args.correction_dataset else SensorDemonstration(args.episode,qualification=args.qualification,legacy_teacher_receipt=args.legacy_teacher_receipt,reset_observation_run=args.reset_observation_run)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("schema") != SENSOR_ACTOR_CHECKPOINT_SCHEMA or
            payload.get('motor_contract_sha256') != episode.motor_contract_sha256 or
            payload["sensor_layout"] != episode.layout or
            payload["physics_dt_s"] != episode.metadata["physics_dt_s"] or
            payload["dimensions"] != asdict(episode.dimensions)):
        raise ValueError("Prediction checkpoint differs from the physical recording contract")
    model = SensorActor(ActorDimensions(**payload["dimensions"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval().requires_grad_(False)
    hidden = None
    sums = {name: np.zeros(episode.dimensions.actions) for name in ("learned", "persistence", "zero")}
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(episode), args.chunk_length):
            length = min(args.chunk_length, len(episode)-start)
            values, target = episode.sequence(start, length)
            prediction, hidden = model(**{k: torch.from_numpy(v)[None] for k, v in values.items()}, hidden=hidden)
            prediction = prediction[0].numpy()
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite prediction")
            previous = episode.numeric["previous_action"][start:start+length]
            for name, value in (("learned", prediction), ("persistence", previous), ("zero", np.zeros_like(target))):
                sums[name] += np.sum((value-target)**2, axis=0)
    groups = {"all": list(range(episode.dimensions.actions)),
              "body": [i for i,n in enumerate(episode.layout["action_order"]) if not n.startswith(("lh_", "rh_"))],
              "left_hand": [i for i,n in enumerate(episode.layout["action_order"]) if n.startswith("lh_")],
              "right_hand": [i for i,n in enumerate(episode.layout["action_order"]) if n.startswith("rh_")]}
    scores = {name: {group: float(values[indices].mean()/len(episode)) for group,indices in groups.items()}
              for name,values in sums.items()}
    identity = episode.metadata["files"]["actor-sensors.npz"]
    training = {e["files"]["actor-sensors.npz"] for e in payload.get("training_episodes", [])}
    validation = {e["files"]["actor-sensors.npz"] for e in payload.get("validation_episodes", [])}
    report = dict(scope=('Prediction on actual recorded student sensors with counterfactual teacher labels; no corrective physical trajectory was executed' if args.correction_dataset else __doc__), closed_loop_evaluated=False, task_success_rate=None,
        episode=episode.metadata, split="training_episode" if identity in training else "separate_validation_episode" if identity in validation else "unseen_episode",
        examples=len(episode), hidden_state="Reset once; recurrent state carried through every recorded sensor sample",
        prediction_mse_normalized_force=scores,
        learned_to_persistence_mse_ratio={group: scores["learned"][group]/max(scores["persistence"][group], 1e-30) for group in groups},
        elapsed_s=time.monotonic()-started, device="cpu", threads=args.threads,
        checkpoint_sha256=digest(args.checkpoint),
        qualification_file_sha256=({'correction-report.json':digest(args.correction_dataset/'report.json')} if args.correction_dataset else {name:digest(args.episode/name) for name in (
            args.qualification, "mechanical-audit.json", "passive-tendon-audit.json", "motor-contract.json")}),
        limitations=["A single training episode cannot establish generalization.",
            "Recorded previous actions are replayed; newly predicted commands do not feed back into the measurements.",
            "No physical state was advanced and no opening attempt was made by the student."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(split=report["split"], scores=scores, elapsed_s=report["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
