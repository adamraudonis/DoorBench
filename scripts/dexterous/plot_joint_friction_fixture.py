#!/usr/bin/env python3
"""Plot retained actual fixture traces; no simulation or image synthesis."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
native=np.load(a.fixtures/'joint-friction-native-002/trace.npz')['rows']
isaac=np.load(a.fixtures/'joint-friction-fixture-002/result/trace.npz')['rows']
fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
series=[('Isaac native friction',isaac[1],'#087f8c'),
        ('MuJoCo native friction',native[1],'#526170'),
        ('Isaac old smooth approximation',isaac[3],'#c64d3c')]
for label,rows,color in series:
    axes[0].plot(rows[:,0],rows[:,2],label=label,color=color,lw=1.3)
    mask=rows[:,0]>=.98
    axes[1].plot(rows[mask,0],rows[mask,2],color=color,lw=1.3,marker='o',ms=3)
axes[0].axvline(.5,color='#697386',lw=.9,ls='--')
axes[0].text(.515,.185,'Applied torque\nremoved',fontsize=8,va='top')
axes[0].set(xlim=(.46,1),title='The old approximation keeps oscillating')
axes[1].set(xlim=(.979,1.001),ylim=(-.08,.08),title='Final 20 ms: actual 2 ms samples')
axes[1].set_xticks([.98,.985,.99,.995,1.])
for ax in axes:
    ax.set(xlabel='Simulation time (s)',ylabel='Joint velocity (rad/s)')
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Isolated finger joint · original inertia, 0.01 Nm friction, 0.05 Nm·s/rad damping',fontsize=11)
axes[0].legend(loc='upper right',fontsize=7.5)
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=160)
