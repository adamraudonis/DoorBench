"""Audit supervised-label coverage of frozen recurrent sampling, without fitting."""
import argparse
import json
from pathlib import Path
import numpy as np

from doorbench.dexterous.recurrent_sampling import sample_windows
from doorbench.dexterous.sensor_training_bundle import verify_bundle, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=verify_bundle(args.dataset_manifest)
    config=manifest['training_settings'];lengths=[r['examples'] for r in manifest['datasets']]
    result=[];arrays={}
    for mode in ('legacy_fixed_burn','prefix_complete_v1'):
        rng=np.random.default_rng(config['seed']);counts=[np.zeros(n,np.int64) for n in lengths]
        for iteration in range(max(5000,config['iterations'])):
            for window in sample_windows(rng,lengths,batch_size=config['batch_size'],supervised_length=config['sequence_length'],
                    burn_in=config['burn_in'],episode_start_probability=config['episode_start_probability'],mode=mode):
                counts[window.episode][window.label_start:window.label_start+window.supervised_length]+=1
            if iteration+1 in (1000,5000):
                rows=[]
                for definition,count in zip(manifest['datasets'],counts,strict=True):
                    name=definition['name'];arrays[f'{mode}_{iteration+1}_{name}']=count.copy()
                    rows.append(dict(name=name,examples=len(count),supervised_events=int(count.sum()),
                        unique_supervised_examples=int(np.count_nonzero(count)),unsupervised_examples=int(np.count_nonzero(count==0)),
                        never_supervised_first128=np.flatnonzero(count[:128]==0).tolist(),
                        minimum_supervisions=int(count.min()),last_example_supervisions=int(count[-1])))
                result.append(dict(mode=mode,optimizer_steps=iteration+1,datasets=rows))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'coverage.npz',**arrays)
    report=dict(scope=__doc__,dataset_manifest_sha256=digest(args.dataset_manifest),seed=config['seed'],
        no_model_training=True,no_new_data=True,results=result,
        structural_defect='Legacy true-start labels cover0..31; random windows begin labels at64 or later. Indices32..63, decisions64..126ms, are impossible to supervise for any random seed or training length.',
        correction='Sample supervised-start index uniformly, warm up only min(start,64) actual preceding observations. All labels become reachable with no invented history.',
        coverage_file_sha256=digest(args.output/'coverage.npz'))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
