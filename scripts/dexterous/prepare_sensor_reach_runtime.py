"""Project frozen acquisition joints into the world-state-free reach inputs."""
import argparse,json
from pathlib import Path
from doorbench.dexterous.sensor_reach_runtime import prepare_reach_runtime_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','schedule','motors','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();receipt=prepare_reach_runtime_inputs(a.reference,a.schedule,json.loads(a.motors.read_text()),a.output/'joint-route.json',a.output/'protocol.json')
    (a.output/'input-projection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
