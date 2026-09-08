"""Own-pad force map with thumb opposition/orientation motors left to posture.

The original independent THJ1/THJ2 transmission is retained. Other fingers keep
their original coupled motor projection. Omitting THJ3/4/5 pressure torques is
an explicit control allocation choice, not a change to robot mechanics.
"""
import numpy as np
from .robot_digit_force import RobotDigitForce


class RobotThumbFlexionForce(RobotDigitForce):
    def __init__(self,robot_only_model,joint_names,action_names,transmission):
        super().__init__(robot_only_model,joint_names,action_names,transmission)
        columns=np.array([self.names.index(n) for n in ('rh_THJ2','rh_THJ1')])
        rows=np.flatnonzero(np.any(abs(self.matrix[:,columns])>0,axis=1))
        A=self.matrix[np.ix_(rows,columns)]
        if len(rows)!=2 or set(self.actions[r] for r in rows)!={'rh_A_THJ1','rh_A_THJ2'} or np.linalg.matrix_rank(A)!=2:
            raise ValueError('Original two independent thumb flexion motors required')
        if np.any(np.delete(self.matrix[rows],columns,axis=1)!=0):
            raise ValueError('Thumb pressure cannot couple into opposition or other joints')
        self.groups['th']=(columns,rows,A,self.groups['th'][3])
