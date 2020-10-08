
'''
Figure: demonstration of a simple point correspondence algorithm.
'''


import os
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
import lmfree2d as lm



#(0) Load data:
dirREPO   = lm.get_repository_path()
names     = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
name      = names[0]
fnameCSV  = os.path.join(dirREPO, 'Data', name, 'contours_sroc.csv')
r         = lm.read_csv(fnameCSV)
i0,i1     = 1, 2
r0,r1     = r[i0], r[i1]



#(1) Process:
np.random.seed(0)
### shuffle:
rA0,rA1   = lm.shuffle_points(r0), lm.shuffle_points(r1)
### reorder:
rB0,rB1   = [lm.reorder_points(r, optimum_order=True, ensure_clockwise=True)  for r in [rA0,rA1]]
### optimum roll correspondence:
rD0       = rB0.copy()
rD1       = lm.corresp_roll(rB1, rB0)
### intermediary roll:
rC0       = rB0.copy()
rC1       = np.roll(rB1, 75, axis=0)



#(1) Plot:
plt.close('all')
lm.set_matplotlib_rcparams()
fig = plt.figure(figsize=(10,3))
# create axes:
axw,axh   = 0.25, 0.95
axx       = np.linspace(0, 1, 5)[:4]
AX        = [plt.axes([x,0,axw,axh])  for x in axx]
ax0,ax1,ax2,ax3 = AX
# plot:
handles   = lm.plot_correspondence(ax0, rA0, rA1)
lm.plot_correspondence(ax1, rB0, rB1)
lm.plot_correspondence(ax2, rC0, rC1)
lm.plot_correspondence(ax3, rD0, rD1)
# panel labels:
[ax.text(0.52, 0.99, '(%s)' %chr(97+i), size=14, ha='center', transform=ax.transAxes)   for i,ax in enumerate(AX)]
labels    = ['Original', 'Ordered', 'Rolled', 'Optimum Roll']
[ax.text(0.5, 0.35, f'{s}', size=12, ha='center', transform=ax.transAxes, zorder=10, bbox=dict(facecolor='w', alpha=0.8))   for ax,s in zip(AX,labels)]
# legend:
labels    = ['Contour Points A', 'Contour Points B', 'Initial Point A', 'Initial Point B', 'Correspondence Line']
leg       = ax0.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.66, 0.6))
plt.setp(leg.get_texts(), size=8)
plt.show()



#(2) Save (or display) figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'corresp.pdf')
plt.savefig(fnamePDF)

