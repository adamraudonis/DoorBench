#!/usr/bin/env python3
"""Export physical sensor placements/configuration; no task objects or state."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def export_layout(robot_xml):
    import mujoco
    from doorbench.dexterous.sensor_contract import AngularTaxelGrid, INTERFACE_VERSION
    path = Path(robot_xml)
    spec = mujoco.MjSpec.from_file(str(path))
    model = spec.compile()
    root = ET.fromstring(spec.to_xml())
    plugins = {node.get('name'): node for node in root.findall('./sensor/plugin')}
    sensors = []
    for i in range(model.nsensor):
        name = model.sensor(i).name
        if not name.endswith('_touch'):
            continue
        node = plugins.get(name)
        if node is None or node.get('plugin') != 'mujoco.sensor.touch_grid':
            raise ValueError(f'Unsupported tactile sensor: {name}')
        config = {child.get('key'): child.get('value') for child in node.findall('config')}
        if int(config.get('nchannel', 1)) != 3 or float(config.get('gamma', 0)) != 0:
            raise ValueError('Only three force channels with gamma=0 are currently supported')
        width, height = map(int, config['size'].split())
        grid = AngularTaxelGrid(width, height, tuple(map(float, config['fov'].split())))
        site = int(model.sensor_objid[i])
        if int(model.sensor_dim[i]) != grid.dimension:
            raise ValueError('Compiled native sensor dimensions disagree')
        body = int(model.site_bodyid[site])
        # Fixed sites may live on welded children; importer body mapping must
        # preserve this body frame or explicitly compose the fixed transform.
        quat = model.site_quat[site]
        sensors.append(dict(name=name, body_name=model.body(body).name,
            position_body_m=model.site_pos[site].tolist(), quaternion_xyzw_body=[*quat[1:].tolist(),float(quat[0])],
            width=width, height=height, fov_degrees=list(grid.fov_degrees), gamma=0,
            dimension=grid.dimension))
    if not sensors:
        raise ValueError('No native tactile sensors found')
    cameras = []
    for name in ('left_eye_camera', 'right_eye_camera'):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if index < 0:
            continue
        cameras.append(dict(name=name, body_name=model.body(int(model.cam_bodyid[index])).name,
            position_body_m=model.cam_pos[index].tolist(), quaternion_wxyz_body=model.cam_quat[index].tolist(),
            convention='opengl', fovy_degrees=float(model.cam_fovy[index])))
    imu_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'imu')
    imu = None if imu_index < 0 else dict(body_name=model.body(int(model.site_bodyid[imu_index])).name,
        position_body_m=model.site_pos[imu_index].tolist(), quaternion_wxyz_body=model.site_quat[imu_index].tolist())
    return dict(cameras=cameras, imu=imu, interface_version=INTERFACE_VERSION, robot_xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        channel_order=['z','x','y'], layout='sensor order, then channel, vertical bin, horizontal bin',
        tactile_dimension=sum(row['dimension'] for row in sensors), sensors=sensors,
        joint_order=[model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE],
        action_order=[model.actuator(i).name for i in range(model.nu)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_layout(args.robot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'tactile_dimension': result['tactile_dimension'], 'sensors': len(result['sensors']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
