"""Scripted sensor inference must run in the native preparation environment."""
import subprocess
import sys
from pathlib import Path


def test_scripted_runtime_imports_without_torch():
    code='''import builtins
original=builtins.__import__
def guarded(name,*a,**k):
    if name=="torch" or name.startswith("torch."):raise AssertionError("Scripted inference imported Torch")
    return original(name,*a,**k)
builtins.__import__=guarded
from doorbench.dexterous.sensor_acquisition_runtime import SensorAcquisitionBalanceRuntime
from doorbench.dexterous.sensor_reach_runtime import SensorReachBalanceRuntime
from doorbench.dexterous.sensor_contract import ActorDimensions
assert ActorDimensions().shapes["tactile"]==(1344,)
'''
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True)


def test_existing_actor_dimensions_import_remains_compatible():
    from doorbench.dexterous.sensor_actor import ActorDimensions as legacy
    from doorbench.dexterous.sensor_contract import ActorDimensions
    assert legacy is ActorDimensions
