"""Shared material partition for the original flexural PVC strip model."""
import math


def strip_layout(spec):
    leaf=spec['leaf'];width=float(leaf['strip_width']);height=float(leaf['height'])
    count=int(leaf['count']);segments=max(2,math.ceil(height/.30))
    pitch=(spec['opening']['width']-.02-width)/max(count-1,1)
    if count>1 and 2*pitch<width-1e-8:
        raise ValueError('Two-layer strip curtain contains overlapping strips in the same layer')
    return {'count':count,'segments':segments,'segment_length':height/segments,
            'pitch':pitch,'width':width,'height':height,'thickness':float(leaf['thickness']),
            'neighbor_overlap_fraction':1-pitch/width if count>1 else 0.}


def segment_name(strip,index):
    return f'strip_{strip}' if index==0 else f'strip_{strip}_segment_{index}'
