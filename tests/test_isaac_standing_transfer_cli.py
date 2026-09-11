"""Exercise CLI validation with only the launcher argument provider stubbed."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def command(extra):
    stub="""import sys,types,runpy
app=types.ModuleType('isaaclab.app')
class Launcher:
 @staticmethod
 def add_app_launcher_args(p):
  p.add_argument('--device',default='cpu');p.add_argument('--headless',action='store_true');p.add_argument('--enable_cameras',action='store_true')
app.AppLauncher=Launcher
sys.modules['isaaclab.app']=app
sys.argv=['isaac_opening.py']+sys.argv[1:]
runpy.run_path('scripts/dexterous/isaac_opening.py',run_name='__main__')
"""
    return [sys.executable,'-c',stub,'--robot-usd','missing','--door-usd','missing','--motors','missing','--reference','missing','--output','missing-output','--native-robot','missing','--acquisition','--operate-after-acquisition','--acquisition-stance-profile','landed-foot-v1','--standing-transfer-route','missing','--seconds','50','--validate-arguments-only']+extra


def test_standing_transfer_preflight_and_incompatible_modes():
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    result=subprocess.run(command([]),cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['physics_started'] is False
    result=subprocess.run(command(['--standing-transfer-hybrid-support']),cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['physics_started'] is False
    for extra in [['--seconds','40'],['--full-opening'],['--standing-transfer-start-seconds','nan']]:
        result=subprocess.run(command(extra),cwd=ROOT,env=env,capture_output=True,text=True)
        assert result.returncode!=0
