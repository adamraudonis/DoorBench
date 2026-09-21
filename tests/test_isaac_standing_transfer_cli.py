"""Exercise CLI validation with only the launcher argument provider stubbed."""
import json,os,subprocess,sys
import ast
from types import SimpleNamespace
import pytest
from doorbench.dexterous.transfer_preload import PROFILES
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


@pytest.mark.parametrize('profile',tuple(PROFILES))
def test_existing_preload_profiles_require_explicit_transfer(profile):
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    argv=command(['--standing-transfer-preload-profile',profile])
    result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    index=argv.index('--standing-transfer-route');del argv[index:index+2]
    result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    assert (result.returncode==0)==(profile=='maintain')
    if profile!='maintain':assert 'preload profile override requires an explicit transfer route' in result.stderr


def test_preload_default_and_constructor_plumbing_match_existing_contract():
    tree=ast.parse((ROOT/'scripts/dexterous/isaac_opening.py').read_text())
    declaration=next(node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and any(isinstance(arg,ast.Constant) and arg.value=='--standing-transfer-preload-profile' for arg in node.args))
    assert next(k.value.value for k in declaration.keywords if k.arg=='default')=='maintain'
    constructor=next(node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name) and node.func.id=='StandingTransferTeacher')
    captured=[]
    def teacher(*args,**kwargs):captured.append(kwargs)
    for profile in PROFILES:
        scope=dict(StandingTransferTeacher=teacher,operation=None,motors=None,
            a=SimpleNamespace(standing_transfer_route='route.json',standing_transfer_start_seconds=28.,
                standing_transfer_preload_profile=profile,standing_transfer_hybrid_support=True,
                standing_transfer_support_load_target=6.))
        eval(compile(ast.Expression(constructor),'<real-producer-constructor>','eval'),scope)
        assert captured[-1]['preload_profile']==profile
        assert captured[-1]['fixed_pad_tracking'] is False and captured[-1]['attained_arm_tracking'] is True
        assert captured[-1]['handoff_seconds']==1. and captured[-1]['support_load_target']==6.
