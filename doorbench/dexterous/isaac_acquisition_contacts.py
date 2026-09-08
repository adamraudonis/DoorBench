"""Evaluator-only contact admission for the explicitly scoped Door55 grasp."""
import math
import re


def acquisition_hand_contact_counts(sensor_paths,filter_paths,occupied):
    """Count actual occupied hand patches; only RH digits may touch the handle.

    Admission here is not grasp qualification: all loaded digit patches still
    require the independent surface, straight-lever and opposed-pad audit.
    """
    hand=unintended=0
    for patch in occupied:
        i,j=patch['sensor'],patch['filter']
        if type(i) is not int or type(j) is not int or not 0<=i<len(sensor_paths) or not 0<=j<len(filter_paths[i]):
            raise ValueError('Invalid recorded contact row')
        force,distance=float(patch['force_N']),float(patch['distance_m'])
        if not math.isfinite(force) or not math.isfinite(distance) or force<0:
            raise ValueError('Invalid measured contact force or separation')
        name=sensor_paths[i].rsplit('/',1)[-1]
        if not name.startswith(('rh_','lh_')) or not (distance<=0 or force>1e-8):continue
        hand+=1
        digit=re.fullmatch(r'rh_(ff|mf|rf|lf|th)(knuckle|proximal|middle|distal|base|hub|metacarpal)',name)
        allowed=digit is not None and filter_paths[i][j]=='/World/Door/Articulation/leaf_handle'
        unintended+=int(not allowed)
    return dict(hand_contact_count=hand,unintended_hand_contact_count=unintended)
