"""Native warning handlers survive nesting and exceptional exits."""
import mujoco
import pytest
from doorbench.native_warnings import capture_native_warnings


def test_capture_forwards_nests_and_restores_original_handler():
    previous=mujoco.get_mju_user_warning();forwarded=[]
    def installed(message):forwarded.append(message)
    mujoco.set_mju_user_warning(installed)
    try:
        with capture_native_warnings() as outer:
            mujoco.get_mju_user_warning()('first')
            with pytest.raises(ValueError):
                with capture_native_warnings() as inner:
                    mujoco.get_mju_user_warning()('nested')
                    raise ValueError('interrupt native run')
            mujoco.get_mju_user_warning()('last')
        assert outer==['first','nested','last'] and inner==['nested']
        assert forwarded==outer and mujoco.get_mju_user_warning() is installed
    finally:mujoco.set_mju_user_warning(previous)


def test_qa_cannot_sign_off_a_counter_free_native_warning(monkeypatch):
    from doorbench import qa
    def trial(*args):
        mujoco.get_mju_user_warning()('Linesearch objective is not convex')
        return {'checks':{'fixture':True},'metrics':{},'signed_off':True}
    monkeypatch.setattr(qa,'_run_qa',trial)
    result=qa.run_qa({},'/nonexistent/fixture',{}, {},{})
    assert not result['signed_off']
    assert not result['checks']['native_warning_messages_absent']
    assert result['metrics']['native_warning_messages']==['Linesearch objective is not convex']
