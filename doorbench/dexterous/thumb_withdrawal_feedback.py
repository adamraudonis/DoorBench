"""Compatibility wrapper for the original screened thumb feedback protocol."""
from .finger_withdrawal_feedback import FingerWithdrawalFeedback


class ThumbWithdrawalFeedback(FingerWithdrawalFeedback):
    def __init__(self,teacher,local_point):
        super().__init__(teacher,local_point,digit='th')

    def force(self,forces,t,goal,elapsed):
        result,info=super().force(forces,t,goal,elapsed)
        return result,{'thumb_'+name:value for name,value in info.items()}
