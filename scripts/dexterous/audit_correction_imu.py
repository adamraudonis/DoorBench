"""Compare real actor IMU gyro with independent FK of evaluator measurements.

This is an evidence audit only. No simulator is stepped and the reconstructed
world/body state never becomes an actor feature or training label.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.correction_demonstrations import CorrectionDemonstration
from doorbench.dexterous.offline_teacher_queries import load_query_evidence,sha
from doorbench.dexterous.sensor_contract import SENSOR_KEYS


def rotation(wxyz):
    q=np.asarray(wxyz);return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('corrections','robot','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous IMU audits')
    data=CorrectionDemonstration(args.corrections);queries,contract,physical,source=load_query_evidence(data.path)
    if sha(args.robot)!=data.layout['robot_xml_sha256']:raise ValueError('Wrong analytic model')
    m=mujoco.MjModel.from_xml_path(str(args.robot));d=mujoco.MjData(m)
    joint=np.array([m.joint(name).id for name in contract['joint_order']]);qa=m.jnt_qposadr[joint];va=m.jnt_dofadr[joint]
    imu=data.layout['imu'];body=m.body(imu['body_name']).id;mount=rotation(imu['quaternion_wxyz_body'])
    jp=np.zeros((3,m.nv));jr=jp.copy();errors=[]
    for i in range(1,len(data)):
        root=queries['root_state'][i];d.qpos[:7]=root[:7];d.qpos[qa]=queries['joint_position'][i]
        d.qvel[:3]=root[7:10];d.qvel[3:6]=rotation(root[3:7]).T@root[10:13];d.qvel[va]=queries['joint_velocity'][i]
        mujoco.mj_forward(m,d);mujoco.mj_jacBody(m,d,jp,jr,body)
        gyro=mount.T@d.xmat[body].reshape(3,3).T@(jr@d.qvel)
        packet=data.packet(i)
        stream=SENSOR_KEYS.index('imu_gyro')
        if not packet['sensor_valid'][stream] or abs(packet['sensor_time_s'][stream]-data.times[i])>1e-9:
            raise ValueError('This exact frame audit requires current gyro observations')
        errors.append(gyro-packet['imu_gyro'])
    if not errors:raise ValueError('No eligible post-reset measurements')
    errors=np.asarray(errors)
    result=dict(scope=__doc__,samples=len(errors),last_time_s=float(data.times[-1]),
        maximum_absolute_gyro_error_rad_s=float(abs(errors).max()),gyro_rmse_rad_s=float(np.sqrt(np.mean(errors**2))),
        source=source,correction_report_sha256=sha(args.corrections/'report.json'),robot_sha256=sha(args.robot),
        script_sha256=sha(__file__),physics_steps=0,actor_inputs_modified=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('samples','maximum_absolute_gyro_error_rad_s','gyro_rmse_rad_s')}))


if __name__=='__main__':main()
