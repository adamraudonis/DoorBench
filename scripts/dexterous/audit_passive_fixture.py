"""Recompute retained passive-joint fixture reports from actual trace arrays.

Read-only, no engine import or physical execution. The ideal-static native
failure and incomplete first Isaac export are retained as separate outcomes.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory):
    results=[];hash_errors=[]
    for name in ('joint-friction-fixture-001','joint-friction-fixture-002'):
        manifest=json.loads((directory/(name+'-verified.json')).read_text())
        for relative,digest in manifest.items():
            if sha(directory/name/relative)!=digest:hash_errors.append(name+'/'+relative)
    inertia=.0000027+.017*.0125**2+.0002
    predicted_terminal=(.02-.01)/.05
    predicted_oscillation=.01*.002/(2*inertia-.05*.002)
    for name in ('joint-friction-native-001','joint-friction-native-002','joint-friction-fixture-002/result'):
        root=directory/name;report=json.loads((root/'report.json').read_text());rows=np.load(root/'trace.npz')['rows']
        assert rows.shape==(4,500,5) and np.isfinite(rows).all()
        assert np.allclose(rows[:,:,0],np.arange(1,501)[None]*.002,atol=1e-12,rtol=0)
        excitation=np.array([.005,.02,.005,.02]);assert np.array_equal(rows[:,:250,3],np.repeat(excitation[:,None],250,axis=1))
        assert not rows[:,250:,3].any() and not rows[:2,:,4].any()
        reconstructed={}
        for i,case in enumerate(('static_under','static_over','smooth_under','smooth_over')):
            reconstructed[case]=dict(final_angle_rad=float(rows[i,-1,1]),driven_displacement_rad=float(rows[i,249,1]-.4),
                late_driven_velocity_rad_s=float(np.mean(rows[i,200:250,2])),late_coast_peak_velocity_rad_s=float(np.max(abs(rows[i,450:,2]))))
        checks=dict(all_steps=True,finite=True,no_joint_limit_hit=bool(np.max(abs(rows[:,:,1]))<1.5),
            backend_static_resists_small_load=abs(reconstructed['static_under']['driven_displacement_rad'])<.002,
            backend_viscous_units=abs(reconstructed['static_over']['late_driven_velocity_rad_s']-.2)<.01,
            backend_coast_stops=reconstructed['static_over']['late_coast_peak_velocity_rad_s']<.001)
        prior=np.c_[np.zeros(2),rows[2:,:-1,2]]
        explicit_error=float(np.max(abs(rows[2:,:,4]-(-.05*prior-.01*np.tanh(prior/.001)))))
        source_match=None;properties_match=None
        if name.startswith('joint-friction-fixture'):
            source_match=report['script_sha256']==sha(root.parent/'probe.py')
            properties=np.asarray(report['metadata']['backend_friction_properties'])
            expected=np.array([[[[.01,.01,.05]]],[[[.01,.01,.05]]],[[[0.,0.,0.]]],[[[0.,0.,0.]]]],np.float32)
            properties_match=np.array_equal(properties,expected)
        results.append(dict(name=name,original_passed=report['passed'],recomputed_passed=all(checks.values()),
            report_reproduced=report['metrics']==reconstructed and report['checks']==checks and report['passed']==all(checks.values()),
            metrics=reconstructed,checks=checks,explicit_passive_law_reconstruction_error_Nm=explicit_error,
            source_matches_report=source_match,original_property_readback_matches_float32_SI=properties_match,
            first_loaded_velocity_rad_s=float(rows[1,0,2]),predicted_first_loaded_velocity_rad_s=.002*(.02-.01)/(inertia+.002*.05),
            terminal_velocity_relative_error=float(abs(reconstructed['static_over']['late_driven_velocity_rad_s']/predicted_terminal-1)),
            smooth_coast_amplitude_error_rad_s=float(abs(reconstructed['smooth_over']['late_coast_peak_velocity_rad_s']-predicted_oscillation)),
            hashes={n:sha(root/n) for n in ('trace.npz','report.json')}))
    return dict(schema='doorbench.passive-fixture-independent-review.v1',verification_passed=not hash_errors and all(r['report_reproduced'] and r['explicit_passive_law_reconstruction_error_Nm']<1e-12 and r['source_matches_report'] is not False and r['original_property_readback_matches_float32_SI'] is not False for r in results),
        hash_errors=hash_errors,results=results,original_effective_inertia_kg_m2=inertia,
        predicted_SI_terminal_velocity_rad_s=predicted_terminal,predicted_legacy_saturated_coast_limit_cycle_rad_s=predicted_oscillation,
        incomplete_isaac_001_retained=not (directory/'joint-friction-fixture-001/result/report.json').exists(),
        physical_steps_in_audit=0,source_reports_modified=False,
        limitation='Measured single-hinge SI and qualitative passive behavior; native dry-friction compliance fails the original ideal-static criterion. No full-robot/contact parity claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(a.directory);result['audit_source_sha256']=sha(__file__)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['verification_passed']:raise SystemExit(1)
if __name__=='__main__':main()
